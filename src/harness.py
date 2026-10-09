"""Run vault tool conversations with LangChain chat models."""

from time import perf_counter
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.tools.registry import registry
from src.tracing import Trace, message_text, message_tool_calls

SYSTEM_PROMPT = """You are an assistant that helps navigate an Obsidian note vault.
The vault is located under a vault path on the local machine,
and is managed by your harness.
You only need to operate on file paths relative to that vault path.

Help the user with the following query:
"""


def run_in_harness(
    chat_model: BaseChatModel,
    *,
    user_prompt: str,
    vault_path: str,
    trace: Trace,
    max_api_calls: int = 20,
    model_label: str | None = None,
    tool_binding_kwargs: dict[str, Any] | None = None,
) -> str:
    """Return final assistant text after resolving registered vault tool calls.

    Each run binds the registry tools and owns its conversation history. Tool
    outputs, including error envelopes, are returned exactly as ToolMessages.
    Completion requires nonempty text without calls. The call limit includes
    final completion, and tools from the last allowed response still execute.
    Invocation traces retain usage and completed tool results on failure.

    Args:
        chat_model: A synchronous LangChain model supporting tool binding.
        user_prompt: The user's vault question.
        vault_path: Root used by registered tools.
        trace: Collector for request, response, and tool metrics.
        max_api_calls: Maximum model invocations in this run.
        model_label: Label for failed invocations and responses without model
            metadata. Defaults to the chat model's class name.
        tool_binding_kwargs: Optional arguments passed to bind_tools, such as
            strict=True for supporting models.

    Raises:
        ValueError: If max_api_calls is not positive.
        RuntimeError: If a response is invalid or the call limit is reached.
        Exception: Original binding, invocation, and unexpected dispatch errors
            propagate unchanged.
    """
    if max_api_calls <= 0:
        raise ValueError("max_api_calls must be positive")
    tools = registry.tool_definitions()
    bound = chat_model.bind_tools(tools, **(tool_binding_kwargs or {}))
    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    label = model_label or type(chat_model).__name__
    trace.record_initial_request(
        model=label,
        tools=tools,
        messages=[m.model_dump(mode="json") for m in messages],
        tool_binding_kwargs=tool_binding_kwargs or {},
        prompt_messages=[("system", SYSTEM_PROMPT), ("user", user_prompt)],
    )

    for _ in range(max_api_calls):
        started = perf_counter()
        try:
            response = bound.invoke(list(messages))
        except Exception:
            trace.record_failure(label, perf_counter() - started)
            raise
        elapsed = perf_counter() - started
        if not isinstance(response, AIMessage):
            trace.record_failure(label, elapsed)
            raise RuntimeError("LangChain response must be AIMessage")
        calls, error = message_tool_calls(response)
        answer, content_error = message_text(response)
        call_trace = trace.record_response(response, elapsed, label, calls, answer)
        error = error or content_error
        metadata = response.response_metadata
        status = metadata.get("status")
        if status is not None and status not in {"completed", "complete"}:
            error = f"LangChain response status is {status}"
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
        if response.additional_kwargs.get("refusal"):
            error = "LangChain response contains a refusal"
        if error:
            raise RuntimeError(error)
        if any(not call.call_id or not call.name for call in calls) or len(
            {call.call_id for call in calls}
        ) != len(calls):
            raise RuntimeError("Tool calls require names and unique nonempty IDs")
        if not calls:
            if not answer:
                raise RuntimeError(
                    "API response contains neither text nor function calls"
                )
            return answer
        messages.append(response)
        for index, call in enumerate(calls):
            output = registry.dispatch(
                name=call.name, arguments_json=call.arguments, vault_path=vault_path
            )
            call_trace.record_tool_result(index, output)
            messages.append(ToolMessage(content=output, tool_call_id=call.call_id))

    raise RuntimeError(f"API call limit ({max_api_calls}) reached")
