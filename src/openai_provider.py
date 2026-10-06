from typing import cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import ResponseInputParam, ToolParam
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming,
)
from openai.types.responses.response_input_param import ResponseInputItemParam

from src.tools.registry import registry
from src.tracing import Trace

VAULT_HOME = "/Users/leonid/code/robothoth/tests/test_vault"

tools: list[ToolParam] = cast(list[ToolParam], registry.declarations())

SYSTEM_PROMPT = """You are an assistant that helps navigate an Obsidian note vault.
The vault is located under a vault path on the local machine,
and is managed by your harness.
You only need to operate on file paths relative to that vault path.

Help the user with the following query:
"""

user_prompt = "Help me locate the file called 'Agentic Software Engineering Factory'."


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
    trace.record_initial_request(dict(request_arguments))

    for call_number in range(1, max_api_calls + 1):
        # TODO: I don't like that this works in a loop only because we
        # change input_list, hidden in these arguments.
        response, call_trace = trace.invoke(client, request_arguments)
        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]

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

        for index, item in enumerate(function_calls):
            tool_result = registry.dispatch(
                name=item.name,
                arguments_json=item.arguments,
                vault_path=vault_path,
            )
            call_trace.record_tool_result(index, tool_result)
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
            print(trace)


if __name__ == "__main__":
    main()
