from dataclasses import dataclass, field
from time import perf_counter
from typing import cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import ResponseInputParam, ToolParam
from openai.types.responses.response_input_param import ResponseInputItemParam

from src.tools import dispatch

VAULT_HOME = "/Users/leonid/code/robothoth/tests/test_vault"

tools: list[ToolParam] = [
    {
        "type": "function",
        "name": "list_notes",
        "description": """Return all notes as sorted vault-relative Markdown paths
under an optional directory.

The directory is interpreted relative to ``vault_path`` and searched
recursively. Only regular Markdown files inside the vault are returned.

Raises:
    FileNotFoundError: If the vault or requested directory does not exist.
    ValueError: If ``path`` is absolute, escapes the vault, or is not a
        directory inside the vault.""",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "An optional path relative to the vault home"
                    "for which to return notes.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        "strict": None,
    },
]

system_prompt = """You are an assistant that helps navigate an Obsidian note vault.
The vault is located under a vault path on the local machine,
and is managed by your harness.
You only need to operate on file paths relative to that vault path.

Help the user with the following query:
"""

user_prompt = "Help me locate the file called 'Agentic Software Engineering Factory'."


@dataclass(frozen=True)
class CallTrace:
    """Record token usage and elapsed seconds for one API invocation.

    Token counts are unknown when the API supplies no usage. Cached and
    cache-write tokens are input breakdowns; reasoning tokens are an output
    breakdown. Elapsed time includes SDK retries but excludes tool execution.
    Failed invocations have no response ID and unknown token counts.
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


@dataclass
class Trace:
    """Collect API invocation metrics for a run, including failed invocations.

    Each instance owns its call list. Costs represent tokens rather than money,
    and latency represents API invocation time rather than whole-run time.
    """

    calls: list[CallTrace] = field(default_factory=list)

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
    to the supplied trace and remain available if the run fails.

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

    input_list: ResponseInputParam = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    for call_number in range(1, max_api_calls + 1):
        started = perf_counter()
        try:
            response = client.responses.create(
                model=model, tools=tools, input=input_list
            )
        except Exception:
            trace.calls.append(
                CallTrace(model=model, elapsed_seconds=perf_counter() - started)
            )
            raise
        elapsed = perf_counter() - started
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
        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        if not function_calls:
            if not response.output_text:
                raise RuntimeError(
                    "API response contains neither text nor function calls"
                )
            return response.output_text
        if call_number == max_api_calls:
            raise RuntimeError(f"API call limit ({max_api_calls}) reached")

        for item in function_calls:
            tool_result = dispatch.dispatch_tool_call(
                name=item.name,
                arguments_json=item.arguments,
                vault_path=vault_path,
            )
            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": tool_result,
                }
            )

    raise AssertionError("Positive API call limit exhausted without a result or error")


def main() -> None:
    """Print the example vault answer and per-call and aggregate API metrics.

    Environment variables are loaded from dotenv before creating the client.
    Metrics are printed even if the run fails; unknown usage is shown explicitly.

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
            for number, call in enumerate(trace.calls, start=1):
                print(f"API call {number}: {call}")
            total = trace.total_cost()
            print(f"Total tokens: {total if total is not None else 'unknown'}")
            print(f"Total API latency: {trace.total_latency():.3f}s")


if __name__ == "__main__":
    main()
