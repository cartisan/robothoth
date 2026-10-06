from time import perf_counter
from typing import cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import ResponseInputParam
from openai.types.responses.response_input_param import ResponseInputItemParam

from src.tools.registry import registry
from src.tracing import Trace

VAULT_HOME = "/Users/leonid/code/robothoth/tests/test_vault"

SYSTEM_PROMPT = """You are an assistant that helps navigate an Obsidian note vault.
The vault is located under a vault path on the local machine,
and is managed by your harness.
You only need to operate on file paths relative to that vault path.

Help the user with the following query:
"""


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

    Tools operate within ``vault_path``; their results, including error envelopes,
    are returned to the model with conversation history. Text accompanying tool
    calls is intermediate; completion requires text without tool calls.

    The supplied trace snapshots the initial request and appends API metrics and
    tool results, retaining them on failure. Valid runs replace the initial
    snapshot; invalid call limits leave the trace unchanged.

    ``max_api_calls`` counts API invocations, including the final text request,
    but excludes SDK retries. Tools from the last allowed response are executed
    and traced before a call-limit error is raised.

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
    conversation: ResponseInputParam = [
        {"role": "developer", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    tools = registry.openai_tool_declarations()
    trace.record_initial_request(
        {"model": model, "tools": tools, "input": conversation}
    )

    for call_number in range(1, max_api_calls + 1):
        started = perf_counter()
        try:
            response = client.responses.create(
                model=model, tools=tools, input=conversation
            )
        except Exception:
            trace.record_failure(
                model=model, elapsed_seconds=perf_counter() - started
            )
            raise
        call_trace = trace.record_response(
            response, elapsed_seconds=perf_counter() - started
        )
        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]

        if response.status != "completed":
            raise RuntimeError(f"API response {response.id} is {response.status}")
        for item in response.output:
            if item.type not in {"message", "reasoning", "function_call"}:
                raise RuntimeError(f"Unsupported response output type: {item.type}")
        if not function_calls:
            if not response.output_text:
                raise RuntimeError(
                    "API response contains neither text nor function calls"
                )
            return response.output_text

        # Add API response to conversation history of model calls
        conversation.extend(
            cast(ResponseInputItemParam, item.to_dict()) for item in response.output
        )

        # execute the requested tools, record them in trace and add them to
        # conversation history for model
        for index, item in enumerate(function_calls):
            tool_result = registry.dispatch(
                name=item.name,
                arguments_json=item.arguments,
                vault_path=vault_path,
            )
            call_trace.record_tool_result(index, tool_result)
            # noinspection bad-argument-type
            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": tool_result,
                }
            )

    raise RuntimeError(f"API call limit ({max_api_calls}) reached")


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
    user_prompt = (
        "Help me locate the file called 'Agentic Software Engineering Factory'."
    )

    trace = Trace()
    with OpenAI() as client:
        try:
            output = run(
                client, user_prompt=user_prompt, vault_path=VAULT_HOME, trace=trace
            )
            print("Final output:")
            print(output)
        finally:
            print()
            print(trace)


if __name__ == "__main__":
    main()
