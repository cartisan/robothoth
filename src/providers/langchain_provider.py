"""Adapt tool-capable LangChain chat models without replacing the harness loop."""

import json
import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable

from src.harness import run_in_harness
from src.providers.llm_provider import (
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolResult,
    validate_tool_results,
)
from src.tracing import Trace


class LangChainProvider:
    """Supply independent conversations through a caller-owned chat model.

    The model must support bind_tools and synchronous invoke. model_label is used
    when response metadata omits the actual model and for failed-call traces.
    Provider credentials and SDK settings belong to the supplied chat model.
    """

    def __init__(self, chat_model: BaseChatModel, *, model_label: str) -> None:
        self.chat_model = chat_model
        self.model = model_label

    def start_session(
        self, system_prompt: str, user_prompt: str, tools: Sequence[ToolDefinition]
    ) -> "LangChainSession":
        """Return fresh message history with the registered schemas bound.

        Raises:
            NotImplementedError: If the supplied chat model lacks tool support.
            Exception: Original model binding errors propagate unchanged.
        """
        declarations: list[dict[str, object]] = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in tools
        ]
        bound = self.chat_model.bind_tools(declarations)
        return LangChainSession(
            bound, self.model, system_prompt, user_prompt, declarations
        )


class LangChainSession:
    """Retain complete AI messages and append matching tool-result messages."""

    def __init__(
        self,
        bound_model: Runnable,
        model: str,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, object]],
    ) -> None:
        self._bound_model = bound_model
        self.model = model
        self._tools = tools
        self._messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        self.prompt_messages = (("system", system_prompt), ("user", user_prompt))
        self._pending: tuple[ToolCall, ...] = ()

    @property
    def initial_request(self) -> dict[str, object]:
        """Return plain message snapshots and the bound tool declarations."""
        return {
            "model": self.model,
            "tools": self._tools,
            "messages": [message.model_dump(mode="json") for message in self._messages],
        }

    def invoke(self) -> ModelResponse:
        """Return normalized AI message output, tool requests, and usage.

        Original tool argument strings are retained when available, including
        invalid JSON; otherwise argument objects are serialized for dispatch.
        When valid and invalid calls coexist, native call IDs establish order;
        absence of a recoverable order yields a validation error. Truncation,
        refusal, unsupported content, and non-AI responses are also rejected.
        Unknown usage remains None; normalized cache breakdowns are not added
        again to LangChain's input or total counts.

        Raises:
            ValueError: If earlier tool calls still require results.
            Exception: Original invocation exceptions propagate unchanged.
        """
        if self._pending:
            raise ValueError("Pending tool calls require results before invocation")
        message = self._bound_model.invoke(list(self._messages))
        if not isinstance(message, AIMessage):
            return ModelResponse(
                model=self.model,
                validation_error="LangChain response must be AIMessage",
            )
        calls = [
            ToolCall(
                c["id"] or "", c["name"], json.dumps(c["args"], ensure_ascii=False)
            )
            for c in message.tool_calls
        ]
        calls.extend(
            ToolCall(c["id"] or "", c["name"] or "", c["args"] or "")
            for c in message.invalid_tool_calls
        )
        error = None
        raw_calls = message.additional_kwargs.get("tool_calls")
        if not isinstance(raw_calls, list):
            raw_calls = message.content if isinstance(message.content, list) else []
        raw_arguments: dict[str, str] = {}
        for raw in raw_calls:
            if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
                continue
            function = raw.get("function")
            arguments = (
                function.get("arguments")
                if isinstance(function, dict)
                else raw.get("args")
            )
            if isinstance(arguments, str):
                raw_arguments[raw["id"]] = arguments
        calls = [
            ToolCall(c.call_id, c.name, raw_arguments.get(c.call_id, c.arguments))
            for c in calls
        ]
        if message.tool_calls and message.invalid_tool_calls:
            order = [
                c["id"]
                for c in raw_calls
                if isinstance(c, dict) and isinstance(c.get("id"), str)
            ]
            if len(order) != len(calls) or set(order) != {c.call_id for c in calls}:
                error = "Cannot recover order of valid and invalid LangChain tool calls"
            else:
                calls.sort(key=lambda c: order.index(c.call_id))
        text_parts: list[str] = []
        if isinstance(message.content, str):
            text_parts.append(message.content)
        else:
            for block in message.content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif block.get("type") in {"text", "output_text"}:
                    text_parts.append(str(block.get("text", "")))
                elif block.get("type") not in {
                    "reasoning",
                    "thinking",
                    "redacted_thinking",
                    "tool_use",
                    "tool_call",
                }:
                    error = f"Unsupported LangChain content type: {block.get('type')}"
        metadata = message.response_metadata
        finish = metadata.get("finish_reason", metadata.get("stop_reason"))
        if finish is not None and finish not in {
            "stop",
            "end_turn",
            "tool_calls",
            "tool_use",
            "function_call",
            "stop_sequence",
        }:
            error = f"LangChain response stopped with {finish}"
        elif finish in {"tool_calls", "tool_use", "function_call"} and not calls:
            error = f"LangChain response has no calls for stop reason {finish}"
        if message.additional_kwargs.get("refusal"):
            error = "LangChain response contains a refusal"
        model = metadata.get("model_name") or metadata.get("model")
        self._messages.append(message)
        self._pending = tuple(calls)
        usage = message.usage_metadata
        input_details = usage.get("input_token_details", {}) if usage else {}
        output_details = usage.get("output_token_details", {}) if usage else {}
        return ModelResponse(
            model=model if isinstance(model, str) else self.model,
            response_id=message.id,
            text="".join(text_parts),
            tool_calls=self._pending,
            input_tokens=usage["input_tokens"] if usage else None,
            output_tokens=usage["output_tokens"] if usage else None,
            total_tokens=usage["total_tokens"] if usage else None,
            cached_tokens=input_details.get("cache_read"),
            cache_write_tokens=input_details.get("cache_creation"),
            reasoning_tokens=output_details.get("reasoning"),
            validation_error=error,
        )

    def add_tool_results(self, results: Sequence[ToolResult]) -> None:
        """Append each exact result as a ToolMessage with its matching call ID.

        Raises:
            ValueError: If the batch does not match pending calls in order.
        """
        validate_tool_results(self._pending, results)
        self._messages.extend(
            ToolMessage(content=r.output, tool_call_id=r.call_id) for r in results
        )
        self._pending = ()


def main() -> None:
    """Print a Kimi K2.6 harness answer and trace through LangChain.

    Load KIMI_API_KEY from the environment or dotenv. Use Moonshot's Chat
    Completions endpoint in instant mode, with a 4096-token response budget.
    The bundled vault is located relative to this module. The synchronous HTTP
    client closes on success or failure, and invocation traces are printed even
    when the harness raises.

    Raises:
        KeyError: If KIMI_API_KEY is missing.
        ValueError: If chat model configuration is invalid.
        openai.OpenAIError: If an API invocation fails.
        RuntimeError: If the harness cannot produce a valid final answer.
    """
    import httpx2 as httpx
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    load_dotenv()
    trace = Trace()
    with httpx.Client(timeout=60.0) as http_client:
        chat_model = ChatOpenAI(
            model="kimi-k2.6",
            api_key=SecretStr(os.environ["KIMI_API_KEY"]),
            base_url="https://api.moonshot.ai/v1",
            use_responses_api=False,
            temperature=0.6,
            max_completion_tokens=4096,
            extra_body={"thinking": {"type": "disabled"}},
            http_client=http_client,
        )
        provider = LangChainProvider(chat_model, model_label="kimi-k2.6")
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
