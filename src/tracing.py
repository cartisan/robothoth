"""Capture API requests, responses, tool results, and run metrics."""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage


def message_text(message: AIMessage) -> tuple[str, str | None]:
    """Return supported assistant text and any unsupported content error.

    Text blocks are concatenated in order. Reasoning and tool blocks contribute
    no final text; unknown blocks make the response invalid.
    """
    if isinstance(message.content, str):
        return message.content, None
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            kind = block.get("type")
            if kind in {"text", "output_text"}:
                parts.append(str(block.get("text", "")))
            elif kind not in {
                "reasoning",
                "thinking",
                "redacted_thinking",
                "tool_use",
                "tool_call",
                "function_call",
            }:
                return "".join(parts), f"Unsupported LangChain content type: {kind}"
        else:
            return "".join(parts), "Unsupported LangChain content block"
    return "".join(parts), None


def message_tool_calls(message: AIMessage) -> tuple[list["ToolCallTrace"], str | None]:
    """Return calls with original argument strings and a call-order error.

    Valid and malformed calls use native call order when mixed. If their order
    cannot be recovered unambiguously, the error prevents dispatch.
    """
    calls = [
        ToolCallTrace(
            c.get("id") or "",
            c.get("name") or "",
            json.dumps(c.get("args"), ensure_ascii=False),
        )
        for c in message.tool_calls
    ]
    calls.extend(
        ToolCallTrace(c.get("id") or "", c.get("name") or "", c.get("args") or "")
        for c in message.invalid_tool_calls
    )
    raw_calls = message.additional_kwargs.get("tool_calls")
    if not isinstance(raw_calls, list):
        raw_calls = message.content if isinstance(message.content, list) else []
    raw_arguments: dict[str, str] = {}
    order: list[str] = []
    for raw in raw_calls:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            continue
        call_id = raw["id"]
        order.append(call_id)
        function = raw.get("function")
        arguments = (
            function.get("arguments") if isinstance(function, dict) else raw.get("args")
        )
        if isinstance(arguments, str):
            raw_arguments[call_id] = arguments
    for call in calls:
        call.arguments = raw_arguments.get(call.call_id, call.arguments)
    if message.tool_calls and message.invalid_tool_calls:
        if (
            len(order) != len(calls)
            or len(set(order)) != len(order)
            or set(order) != {c.call_id for c in calls}
        ):
            return (
                calls,
                "Cannot recover order of valid and invalid LangChain tool calls",
            )
        calls.sort(key=lambda call: order.index(call.call_id))
    return calls, None


@dataclass(frozen=True)
class RequestTrace:
    """Retain an independent snapshot of all supplied API request arguments.

    Explicit prompt messages govern display when supplied. Otherwise, text is
    extracted from the input argument. Nested argument values are copied on
    construction so later conversation or tool declaration changes cannot alter
    the recorded request.
    """

    arguments: dict[str, object]
    prompt_messages: list[tuple[str, str]] | None = None

    def __post_init__(self) -> None:
        """Detach arguments and prompt messages from mutable caller data."""
        object.__setattr__(self, "arguments", deepcopy(self.arguments))
        object.__setattr__(self, "prompt_messages", deepcopy(self.prompt_messages))

    def __str__(self) -> str:
        """Return prompt message roles and text in order on one line.

        Explicit prompt messages are displayed when provided. Otherwise,
        string content and input/output text blocks are displayed. Non-message
        inputs and non-text blocks are omitted. A string input is shown as a
        user message. JSON quoting escapes line breaks and tabs in message text;
        model, tools, and other request arguments are retained but not displayed.
        """
        if self.prompt_messages is not None:
            return " | ".join(
                f"{role}: {json.dumps(text, ensure_ascii=False)}"
                for role, text in self.prompt_messages
            )
        request_input = self.arguments.get("input", [])
        if isinstance(request_input, str):
            return f"user: {json.dumps(request_input, ensure_ascii=False)}"
        messages: list[str] = []
        if isinstance(request_input, list):
            for item in request_input:
                if not isinstance(item, dict) or "role" not in item:
                    continue
                role = item["role"]
                content = item.get("content")
                if not isinstance(role, str):
                    continue
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "".join(
                        block["text"]
                        for block in content
                        if isinstance(block, dict)
                        and block.get("type") in {"input_text", "output_text"}
                        and isinstance(block.get("text"), str)
                    )
                else:
                    continue
                messages.append(f"{role}: {json.dumps(text, ensure_ascii=False)}")
        return " | ".join(messages)


@dataclass
class ToolCallTrace:
    """Record a requested function and its exact returned output.

    Arguments retain original JSON strings when available or serialized
    LangChain argument objects. Output is the unmodified dispatcher envelope,
    including tool errors, or None if no result returned.
    Requests rejected before dispatch and dispatches that raise have no output.
    """

    call_id: str
    name: str
    arguments: str
    output: str | None = None


@dataclass(frozen=True)
class CallTrace:
    """Record model output, token usage, and latency for one API invocation.

    Token counts are unknown when the API supplies no usage. Cached and
    cache-write tokens are input breakdowns; reasoning tokens are an output
    breakdown. Elapsed time covers the model invocation and excludes tool
    execution.
    Output contains all requested function calls in response order as
    ``name(arguments_json)``, or response text when no functions are requested.
    Tool requests take precedence over accompanying text. Invocations that fail
    before returning a response have no response ID or output and unknown usage.
    Each instance owns its tool call list, in response order. Tool records retain
    results as execution completes, even if a later tool or API invocation fails.
    """

    model: str
    elapsed_seconds: float
    response_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    output: str | None = None
    tool_calls: list[ToolCallTrace] = field(default_factory=list)

    def record_tool_result(self, index: int, output: str) -> None:
        """Retain the exact result for the tool at its response-order index.

        Empty strings and error envelopes are retained without modification.

        Raises:
            IndexError: If the index does not identify a requested tool.
        """
        self.tool_calls[index].output = output

    def __str__(self) -> str:
        """Return model metrics, response output, and requested tool results.

        Latency is shown in seconds with three decimal places. Missing usage
        appears as ``unknown`` and absent output as ``(no output)``. Line breaks
        within model output and tool results are replaced with spaces for display;
        stored text remains unchanged. Each tool result appears on a separate
        line with its name.
        Missing results appear as ``(not executed)``; empty results stay empty.
        """
        cost = (
            f"{self.total_tokens} tokens"
            if self.total_tokens is not None
            else "unknown"
        )
        output = self.output if self.output is not None else "(no output)"
        output = " ".join(output.splitlines())
        summary = (
            f"Model: {self.model}\t"
            f"Latency: {self.elapsed_seconds:.3f}s\t"
            f"Total cost: {cost}\t"
            f"Output: {output}"
        )
        for tool in self.tool_calls:
            result = (
                " ".join(tool.output.splitlines())
                if tool.output is not None
                else "(not executed)"
            )
            summary += f"\n\tName: {tool.name}\tResult: {result}"
        return summary


@dataclass
class Trace:
    """Collect API invocation metrics for a run, including failed invocations.

    Each instance owns its call list. Costs represent tokens rather than money,
    and latency represents API invocation time rather than whole-run time.
    The initial request snapshot is separate from API response metrics and is
    absent until a valid run attempts its first API invocation.
    """

    calls: list[CallTrace] = field(default_factory=list)
    initial_request: RequestTrace | None = None

    def total_cost(self) -> int | None:
        """Return total tokens, or None if any invocation has unknown usage.

        An empty trace costs zero. Token breakdowns are not added separately.
        """
        total = 0
        for call in self.calls:
            if call.total_tokens is None:
                return None
            total += call.total_tokens
        return total

    def total_latency(self) -> float:
        """Return summed API elapsed seconds, or zero for an empty trace."""
        return sum((call.elapsed_seconds for call in self.calls), 0.0)

    def record_initial_request(
        self,
        model: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tool_binding_kwargs: dict[str, Any],
        prompt_messages: list[tuple[str, str]],
    ) -> None:
        """Replace the initial request with a snapshot of model call inputs.

        Tools, serialized messages, binding options, and display prompt messages
        are copied. Any prior call traces remain when the trace is reused.
        """
        self.initial_request = RequestTrace(
            {
                "model": model,
                "tools": tools,
                "messages": messages,
                "tool_binding_kwargs": tool_binding_kwargs,
            },
            prompt_messages,
        )

    def record_failure(self, model: str, elapsed_seconds: float) -> CallTrace:
        """Append and return a failed invocation trace with unknown usage.

        The caller supplies the requested model and API elapsed time. No response
        ID or output is recorded because the invocation returned no response.
        """
        call = CallTrace(model=model, elapsed_seconds=elapsed_seconds)
        self.calls.append(call)
        return call

    def record_response(
        self,
        response: AIMessage,
        elapsed_seconds: float,
        model_label: str,
        tool_calls: list[ToolCallTrace],
        text: str,
    ) -> CallTrace:
        """Append and return a trace of response metrics and requested tools.

        The caller supplies API elapsed time. Output, usage, and tool requests
        are retained regardless of response status; response validation remains
        the caller's responsibility. Missing usage stays unknown.
        """
        output = (
            "\n".join(f"{item.name}({item.arguments})" for item in tool_calls)
            if tool_calls
            else text or None
        )
        usage = response.usage_metadata
        input_details = usage.get("input_token_details", {}) if usage else {}
        output_details = usage.get("output_token_details", {}) if usage else {}
        metadata = response.response_metadata
        model = metadata.get("model_name") or metadata.get("model")
        call = CallTrace(
            model=model if isinstance(model, str) else model_label,
            response_id=response.id,
            elapsed_seconds=elapsed_seconds,
            input_tokens=usage.get("input_tokens") if usage else None,
            output_tokens=usage.get("output_tokens") if usage else None,
            cached_tokens=input_details.get("cache_read"),
            cache_write_tokens=input_details.get("cache_creation"),
            reasoning_tokens=output_details.get("reasoning"),
            total_tokens=usage.get("total_tokens") if usage else None,
            output=output,
            tool_calls=tool_calls,
        )
        self.calls.append(call)
        return call

    def __str__(self) -> str:
        """Return the initial prompt, API call traces, and aggregate metrics.

        Unknown usage is shown explicitly. Model output and tool results are
        flattened only for display; stored output keeps its original line breaks.
        An empty trace displays zero total tokens and API latency.
        """
        lines = []
        if self.initial_request is not None:
            lines.append(f"Initial request: {self.initial_request}")
        lines.extend(
            f"API call {number}: {call}"
            for number, call in enumerate(self.calls, start=1)
        )
        total = self.total_cost()
        lines.append(f"Total tokens: {total if total is not None else 'unknown'}")
        lines.append(f"Total API latency: {self.total_latency():.3f}s")
        return "\n".join(lines)
