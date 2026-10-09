"""Run vault tool conversations independently of model APIs."""

from time import perf_counter

from src.providers.llm_provider import LLMProvider, ToolResult
from src.tools.registry import registry
from src.tracing import Trace

SYSTEM_PROMPT = """You are an assistant that helps navigate an Obsidian note vault.
The vault is located under a vault path on the local machine,
and is managed by your harness.
You only need to operate on file paths relative to that vault path.

Help the user with the following query:
"""


def run_in_harness(
    provider: LLMProvider,
    *,
    user_prompt: str,
    vault_path: str,
    trace: Trace,
    max_api_calls: int = 20,
) -> str:
    """Return final assistant text after resolving registered vault tool calls.

    Tools operate within vault_path. Their exact results, including error
    envelopes, are returned through the provider session. Text accompanying tool
    calls is intermediate; completion requires nonempty text without tool calls.

    Valid runs replace the initial trace snapshot and append invocation metrics
    and tool results, retaining them on failure. Invalid call limits leave the
    trace unchanged. Latency measures provider invocations, including SDK retries
    and response normalization, but excludes tool execution.

    max_api_calls counts invocations including final completion, not SDK retries.
    Tools from the last allowed response execute before a call-limit error.

    Raises:
        ValueError: If max_api_calls is not positive.
        RuntimeError: If a response is invalid, has missing or duplicate tool IDs,
            contains neither text nor tools, or requires calls beyond the limit.
        Exception: Original provider and unexpected dispatch exceptions propagate.
    """
    if max_api_calls <= 0:
        raise ValueError("max_api_calls must be positive")

    session = provider.start_session(
        SYSTEM_PROMPT, user_prompt, registry.tool_definitions()
    )
    trace.record_initial_request(session.initial_request, session.prompt_messages)

    for _ in range(max_api_calls):
        started = perf_counter()
        try:
            response = session.invoke()
        except Exception:
            trace.record_failure(session.model, perf_counter() - started)
            raise
        call_trace = trace.record_response(response, perf_counter() - started)
        if response.validation_error:
            raise RuntimeError(response.validation_error)
        calls = response.tool_calls
        if any(not call.call_id or not call.name for call in calls) or len(
            {call.call_id for call in calls}
        ) != len(calls):
            raise RuntimeError("Tool calls require names and unique nonempty IDs")
        if not calls:
            if not response.text:
                raise RuntimeError(
                    "API response contains neither text nor function calls"
                )
            return response.text

        results: list[ToolResult] = []
        for index, call in enumerate(calls):
            output = registry.dispatch(
                name=call.name, arguments_json=call.arguments, vault_path=vault_path
            )
            call_trace.record_tool_result(index, output)
            results.append(ToolResult(call.call_id, output))
        session.add_tool_results(results)

    raise RuntimeError(f"API call limit ({max_api_calls}) reached")
