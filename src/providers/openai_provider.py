"""Adapt OpenAI Responses conversations to the vault harness contract."""

from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import ResponseInputParam, ToolParam
from openai.types.responses.response_input_param import ResponseInputItemParam

from src.harness import run_in_harness
from src.providers.llm_provider import (
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolResult,
    validate_tool_results,
)
from src.tracing import Trace


def tool_declarations(tools: Sequence[ToolDefinition]) -> list[ToolParam]:
    """Return strict OpenAI function declarations for neutral tool definitions."""
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
            "strict": True,
        }
        for tool in tools
    ]


class OpenAIProvider:
    """Supply independent Responses sessions using an injected or owned client.

    Model selection belongs to this provider. A client created here uses the
    OpenAI SDK's environment configuration and is closed on context exit or close.
    Injected clients remain caller-owned and are never closed by this provider.

    Raises:
        openai.OpenAIError: If client creation fails.
    """

    def __init__(
        self, client: OpenAI | None = None, *, model: str = "gpt-6-luna"
    ) -> None:
        self._owns_client = client is None
        self.client = client if client is not None else OpenAI()
        self.model = model

    def __enter__(self) -> "OpenAIProvider":
        """Return this provider for context-managed owned-client cleanup."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close an owned client on normal exit or failure."""
        self.close()

    def close(self) -> None:
        """Close the owned client, leaving an injected client open."""
        if self._owns_client:
            self.client.close()

    def start_session(
        self, system_prompt: str, user_prompt: str, tools: Sequence[ToolDefinition]
    ) -> "OpenAISession":
        """Return an independent Responses history with strict function tools."""
        return OpenAISession(self.client, self.model, system_prompt, user_prompt, tools)


class OpenAISession:
    """Retain native Responses output items for one harness conversation."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        user_prompt: str,
        tools: Sequence[ToolDefinition],
    ) -> None:
        self._client = client
        self.model = model
        self._tools = tool_declarations(tools)
        self._conversation: ResponseInputParam = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        self.prompt_messages = (("developer", system_prompt), ("user", user_prompt))
        self._pending: tuple[ToolCall, ...] = ()

    @property
    def initial_request(self) -> dict[str, object]:
        """Return native request arguments for snapshotting before invocation."""
        return {
            "model": self.model,
            "tools": self._tools,
            "input": self._conversation,
        }

    def invoke(self) -> ModelResponse:
        """Return normalized output and preserve all native history items.

        Noncompleted responses and unsupported output types carry a validation
        error while retaining text, usage, and function calls for tracing.

        Raises:
            ValueError: If earlier tool calls still require results.
            openai.OpenAIError: If the API invocation fails.
        """
        if self._pending:
            raise ValueError("Pending tool calls require results before invocation")
        response = self._client.responses.create(
            model=self.model, tools=self._tools, input=self._conversation
        )
        calls = tuple(
            ToolCall(item.call_id, item.name, item.arguments)
            for item in response.output
            if item.type == "function_call"
        )
        error = None
        if response.status != "completed":
            error = f"API response {response.id} is {response.status}"
        else:
            for item in response.output:
                if item.type not in {"message", "reasoning", "function_call"}:
                    error = f"Unsupported response output type: {item.type}"
                    break
        self._conversation.extend(
            cast(ResponseInputItemParam, item.to_dict()) for item in response.output
        )
        self._pending = calls
        usage = response.usage
        return ModelResponse(
            model=response.model,
            response_id=response.id,
            text=response.output_text,
            tool_calls=calls,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            cached_tokens=usage.input_tokens_details.cached_tokens if usage else None,
            cache_write_tokens=(
                usage.input_tokens_details.cache_write_tokens if usage else None
            ),
            reasoning_tokens=(
                usage.output_tokens_details.reasoning_tokens if usage else None
            ),
            validation_error=error,
        )

    def add_tool_results(self, results: Sequence[ToolResult]) -> None:
        """Append exact outputs with the corresponding Responses call IDs.

        Raises:
            ValueError: If the batch does not match pending calls in order.
        """
        validate_tool_results(self._pending, results)
        self._conversation.extend(
            {"type": "function_call_output", "call_id": r.call_id, "output": r.output}
            for r in results
        )
        self._pending = ()


def main() -> None:
    """Print an OpenAI harness answer and its trace for the sample vault query.

    Load environment configuration from dotenv and locate the bundled vault
    relative to this module. The owned client closes on success or failure, and
    invocation traces are printed even when the harness raises.

    Raises:
        openai.OpenAIError: If client creation or an API invocation fails.
        RuntimeError: If the harness cannot produce a valid final answer.
    """
    load_dotenv()
    trace = Trace()
    with OpenAIProvider() as provider:
        try:
            output = run_in_harness(
                provider,
                user_prompt=(
                    "Help me locate the file called "
                    "'Agentic Software Engineering Factory'."
                ),
                vault_path=str(
                    Path(__file__).resolve().parents[2] / "tests/test_vault"
                ),
                trace=trace,
            )
            print("Final output:")
            print(output)
        finally:
            print()
            print(trace)


if __name__ == "__main__":
    main()
