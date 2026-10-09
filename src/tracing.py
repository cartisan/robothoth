"""Capture API requests, responses, tool results, and run metrics."""

import json
from copy import deepcopy
from dataclasses import dataclass, field

from src.providers.llm_provider import ModelResponse


@dataclass(frozen=True)
class RequestTrace:
    """Retain an independent snapshot of all supplied API request arguments.

    Explicit prompt messages govern display when supplied. Otherwise, text is
    extracted from the input argument. Nested argument values are copied on
    construction so later conversation or tool declaration changes cannot alter
    the recorded request.
    """

    arguments: dict[str, object]
    prompt_messages: tuple[tuple[str, str], ...] | None = None

    def __post_init__(self) -> None:
        """Detach the stored arguments from the caller's mutable request data."""
        object.__setattr__(self, "arguments", deepcopy(self.arguments))

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

    Arguments retain original JSON strings or serialized provider argument
    objects. Output is the unmodified dispatcher envelope, including tool errors,
    or None if no result returned.
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
    breakdown. Elapsed time includes SDK retries and response normalization but
    excludes tool execution.
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
        arguments: dict[str, object],
        prompt_messages: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        """Replace the initial request with an independent argument snapshot.

        Prompt messages provide display text independently of the native request.
        Existing API call traces remain available when the trace is reused.
        """
        self.initial_request = RequestTrace(arguments, prompt_messages)

    def record_failure(self, model: str, elapsed_seconds: float) -> CallTrace:
        """Append and return a failed invocation trace with unknown usage.

        The caller supplies the requested model and API elapsed time. No response
        ID or output is recorded because the invocation returned no response.
        """
        call = CallTrace(model=model, elapsed_seconds=elapsed_seconds)
        self.calls.append(call)
        return call

    def record_response(
        self, response: ModelResponse, elapsed_seconds: float
    ) -> CallTrace:
        """Append and return a trace of response metrics and requested tools.

        The caller supplies API elapsed time. Output, usage, and tool requests
        are retained regardless of response status; response validation remains
        the caller's responsibility. Missing usage stays unknown.
        """
        function_calls = response.tool_calls
        output = (
            "\n".join(f"{item.name}({item.arguments})" for item in function_calls)
            if function_calls
            else response.text or None
        )
        call = CallTrace(
            model=response.model,
            response_id=response.response_id,
            elapsed_seconds=elapsed_seconds,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_tokens=response.cached_tokens,
            cache_write_tokens=response.cache_write_tokens,
            reasoning_tokens=response.reasoning_tokens,
            total_tokens=response.total_tokens,
            output=output,
            tool_calls=[
                ToolCallTrace(
                    call_id=item.call_id, name=item.name, arguments=item.arguments
                )
                for item in function_calls
            ],
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
