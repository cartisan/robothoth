import json
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import ResponseInputParam, ToolParam
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming,
)
from openai.types.responses.response_input_param import ResponseInputItemParam

from src.tools.registry import registry

VAULT_HOME = "/Users/leonid/code/robothoth/tests/test_vault"

tools: list[ToolParam] = cast(list[ToolParam], registry.declarations())

SYSTEM_PROMPT = """You are an assistant that helps navigate an Obsidian note vault.
The vault is located under a vault path on the local machine,
and is managed by your harness.
You only need to operate on file paths relative to that vault path.

Help the user with the following query:
"""

user_prompt = "Help me locate the file called 'Agentic Software Engineering Factory'."


@dataclass(frozen=True)
class RequestTrace:
    """Retain an independent snapshot of all supplied API request arguments.

    Nested argument values are copied on construction so later conversation or
    tool declaration changes cannot alter the recorded request.
    """

    arguments: dict[str, object]

    def __post_init__(self) -> None:
        """Detach the stored arguments from the caller's mutable request data."""
        object.__setattr__(self, "arguments", deepcopy(self.arguments))

    def __str__(self) -> str:
        """Return prompt message roles and text in order on one line.

        String content and input/output text blocks are displayed. Non-message
        inputs and non-text blocks are omitted. A string input is shown as a
        user message. JSON quoting escapes line breaks and tabs in message text;
        model, tools, and other request arguments are retained but not displayed.
        """
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

    Arguments retain the API's original JSON string. Output is the unmodified
    dispatcher envelope, including tool errors, or None if no result returned.
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
    breakdown. Elapsed time includes SDK retries but excludes tool execution.
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


def run(
    client: OpenAI,
    *,
    user_prompt: str,
    vault_path: str,
    trace: Trace,
    model: str = "gpt-6-luna",
    max_api_calls: int = 20,
) -> str:
    """Return final assistant text after resolving requested vault tool calls.

    All response output is retained in the conversation, and each function
    result is paired with its call ID. Text accompanying function calls is
    intermediate; completion requires text without function calls. Tool error
    envelopes are returned to the model for resolution. Metrics are appended
    to the supplied trace and remain available if the run fails. Each call trace
    retains requested functions, including their unmodified argument JSON, or
    response text when no functions are requested. Output is retained even for
    responses rejected by validation or the call limit. Structured tool records
    on the requesting API trace retain call IDs, names, original arguments, and
    exact returned envelopes. Results remain available if a later call fails;
    requests without a returned result have None as their output.
    The initial request snapshots every supplied API argument before invocation,
    including all prompt messages. A valid run replaces this snapshot, while
    existing call traces are appended to; an invalid call limit leaves it intact.

    The positive call limit counts API invocations, including the final text
    request, but not internal SDK retries. Tools requested by the last allowed
    response are not executed because no subsequent request can consume them.

    Raises:
        ValueError: If ``max_api_calls`` is not positive.
        RuntimeError: If a response is not completed, contains an unsupported
            output type, contains neither text nor function calls, or requests
            more tools at the call limit.
        openai.OpenAIError: If an API invocation fails.
    """
    if max_api_calls <= 0:
        raise ValueError("max_api_calls must be positive")

    # noinspection bad-assignment
    input_list: ResponseInputParam = [
        {"role": "developer", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    request_arguments: ResponseCreateParamsNonStreaming = {
        "model": model,
        "tools": tools,
        "input": input_list,
    }
    trace.initial_request = RequestTrace(dict(request_arguments))

    for call_number in range(1, max_api_calls + 1):
        started = perf_counter()
        try:
            # TODO: I don't like that this works in a loop only because we
            # change input_list, hidden in these arguments.
            response = client.responses.create(**request_arguments)
        except Exception:
            trace.calls.append(
                CallTrace(model=model, elapsed_seconds=perf_counter() - started)
            )
            raise
        elapsed = perf_counter() - started
        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        output = (
            "\n".join(f"{item.name}({item.arguments})" for item in function_calls)
            if function_calls
            else response.output_text or None
        )
        usage = response.usage
        trace.calls.append(
            CallTrace(
                model=response.model,
                response_id=response.id,
                elapsed_seconds=elapsed,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                cached_tokens=(
                    usage.input_tokens_details.cached_tokens if usage else None
                ),
                cache_write_tokens=(
                    usage.input_tokens_details.cache_write_tokens if usage else None
                ),
                reasoning_tokens=(
                    usage.output_tokens_details.reasoning_tokens if usage else None
                ),
                total_tokens=usage.total_tokens if usage else None,
                output=output,
                tool_calls=[
                    ToolCallTrace(
                        call_id=item.call_id,
                        name=item.name,
                        arguments=item.arguments,
                    )
                    for item in function_calls
                ],
            )
        )

        if response.status != "completed":
            raise RuntimeError(f"API response {response.id} is {response.status}")
        for item in response.output:
            if item.type not in {"message", "reasoning", "function_call"}:
                raise RuntimeError(f"Unsupported response output type: {item.type}")

        input_list.extend(
            cast(ResponseInputItemParam, item.to_dict()) for item in response.output
        )
        if not function_calls:
            if not response.output_text:
                raise RuntimeError(
                    "API response contains neither text nor function calls"
                )
            return response.output_text
        if call_number == max_api_calls:
            raise RuntimeError(f"API call limit ({max_api_calls}) reached")

        for item, tool_trace in zip(function_calls, trace.calls[-1].tool_calls):
            tool_result = registry.dispatch(
                name=item.name,
                arguments_json=item.arguments,
                vault_path=vault_path,
            )
            tool_trace.output = tool_result
            # noinspection bad-argument-type
            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": tool_result,
                }
            )

    raise AssertionError("Positive API call limit exhausted without a result or error")


def main() -> None:
    """Print the example vault answer, initial prompt, call outputs, and metrics.

    Environment variables are loaded from dotenv before creating the client.
    Metrics are printed even if the run fails; unknown usage is shown explicitly.
    The initial prompt is printed on one line before API call traces.

    Raises:
        openai.OpenAIError: If client creation or an API invocation fails.
        RuntimeError: If the run cannot produce final text within its call limit.
    """
    load_dotenv()
    trace = Trace()
    with OpenAI() as client:
        try:
            output = run(
                client, user_prompt=user_prompt, vault_path=VAULT_HOME, trace=trace
            )
            print("Final output:")
            print(output)
        finally:
            if trace.initial_request is not None:
                print(f"Initial request: {trace.initial_request}")
            for number, call in enumerate(trace.calls, start=1):
                print(f"API call {number}: {call}")
            total = trace.total_cost()
            print(f"Total tokens: {total if total is not None else 'unknown'}")
            print(f"Total API latency: {trace.total_latency():.3f}s")


if __name__ == "__main__":
    main()
