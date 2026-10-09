"""Adapt Claude Messages client-tool conversations to the harness contract."""

import json
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import cast

from anthropic import Anthropic
from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam
from dotenv import load_dotenv

from src.harness import run_in_harness
from src.providers.llm_provider import (
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolResult,
    validate_tool_results,
)
from src.tracing import Trace


class ClaudeProvider:
    """Supply Claude client-tool sessions with explicit model and output budget.

    SDK-created clients use environment configuration and are closed by context
    exit or close. Injected clients remain caller-owned. max_tokens applies to
    each API response rather than the full harness run.

    Raises:
        ValueError: If max_tokens is not positive.
        anthropic.AnthropicError: If client creation fails.
    """

    def __init__(
        self,
        client: Anthropic | None = None,
        *,
        model: str,
        max_tokens: int,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self._owns_client = client is None
        self.client = client if client is not None else Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def __enter__(self) -> "ClaudeProvider":
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
    ) -> "ClaudeSession":
        """Return independent Messages history and translated tool schemas."""
        return ClaudeSession(
            self.client, self.model, self.max_tokens, system_prompt, user_prompt, tools
        )


class ClaudeSession:
    """Retain complete Claude content blocks and ordered client-tool results."""

    def __init__(
        self,
        client: Anthropic,
        model: str,
        max_tokens: int,
        system_prompt: str,
        user_prompt: str,
        tools: Sequence[ToolDefinition],
    ) -> None:
        self._client = client
        self.model = model
        self._max_tokens = max_tokens
        self._system = system_prompt
        self._tools: list[ToolParam] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]
        self._messages: list[MessageParam] = [{"role": "user", "content": user_prompt}]
        self.prompt_messages = (("system", system_prompt), ("user", user_prompt))
        self._pending: tuple[ToolCall, ...] = ()

    @property
    def initial_request(self) -> dict[str, object]:
        """Return native Messages request arguments as snapshot-compatible data."""
        return {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "system": self._system,
            "tools": self._tools,
            "messages": self._messages,
        }

    def invoke(self) -> ModelResponse:
        """Return normalized client calls, text, usage, and stop validation.

        Only text, thinking, redacted thinking, and client tool_use blocks are
        supported. tool_use requires calls; end_turn requires no calls. Other
        stop reasons carry validation errors, including truncated responses.
        Cache read/write counts are included in normalized total input tokens.

        Raises:
            ValueError: If earlier tool calls still require results.
            anthropic.AnthropicError: If the API invocation fails.
        """
        if self._pending:
            raise ValueError("Pending tool calls require results before invocation")
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            system=self._system,
            tools=self._tools,
            messages=self._messages,
        )
        calls = tuple(
            ToolCall(block.id, block.name, json.dumps(block.input, ensure_ascii=False))
            for block in response.content
            if block.type == "tool_use"
        )
        error = None
        if response.stop_reason not in {"tool_use", "end_turn"}:
            error = f"API response {response.id} stopped with {response.stop_reason}"
        elif (response.stop_reason == "tool_use") != bool(calls):
            error = f"API response has inconsistent stop reason: {response.stop_reason}"
        else:
            for block in response.content:
                if block.type not in {
                    "text",
                    "tool_use",
                    "thinking",
                    "redacted_thinking",
                }:
                    error = f"Unsupported response content type: {block.type}"
                    break
        self._messages.append(
            cast(
                MessageParam,
                {
                    "role": "assistant",
                    "content": [block.model_dump() for block in response.content],
                },
            )
        )
        self._pending = calls
        usage = response.usage
        cache_read = usage.cache_read_input_tokens
        cache_write = usage.cache_creation_input_tokens
        input_tokens = usage.input_tokens + (cache_read or 0) + (cache_write or 0)
        return ModelResponse(
            model=response.model,
            response_id=response.id,
            text="".join(b.text for b in response.content if b.type == "text"),
            tool_calls=calls,
            input_tokens=input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=input_tokens + usage.output_tokens,
            cached_tokens=cache_read,
            cache_write_tokens=cache_write,
            validation_error=error,
        )

    def add_tool_results(self, results: Sequence[ToolResult]) -> None:
        """Append one user message with all matching tool_result blocks.

        Dispatcher error envelopes stay unchanged and are flagged as tool errors.
        Empty and non-JSON strings are retained as ordinary tool output.

        Raises:
            ValueError: If the batch does not match pending calls in order.
        """
        validate_tool_results(self._pending, results)
        blocks: list[ToolResultBlockParam] = []
        for result in results:
            try:
                envelope = json.loads(result.output)
            except json.JSONDecodeError:
                envelope = None
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result.call_id,
                    "content": result.output,
                    "is_error": isinstance(envelope, dict)
                    and envelope.get("ok") is False,
                }
            )
        self._messages.append({"role": "user", "content": blocks})
        self._pending = ()


def main() -> None:
    """Print a Claude harness answer and its trace for the sample vault query.

    Load environment configuration from dotenv and use Claude Haiku 5.5 with a
    4096-token response budget. The bundled vault is located relative to this
    module. The owned client closes on success or failure, and invocation traces
    are printed even when the harness raises.

    Raises:
        anthropic.AnthropicError: If client creation or an API invocation fails.
        RuntimeError: If the harness cannot produce a valid final answer.
    """
    load_dotenv()
    trace = Trace()
    with ClaudeProvider(model="claude-haiku-5-5", max_tokens=4096) as provider:
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
