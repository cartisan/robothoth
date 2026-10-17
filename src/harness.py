"""Run vault tool conversations with LangChain chat models."""

from dataclasses import dataclass
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
from src.utils import SystemPrompt

SYSTEM_PROMPT = SystemPrompt(
    version=1,
    text="""You are an assistant that helps navigate an Obsidian note vault.
The vault is located under a vault path on the local machine,
and is managed by your harness.
You only need to operate on file paths relative to that vault path.

Help the user with the following query:
""",
)


@dataclass(frozen=True)
class HarnessResult:
    """Retain a valid final answer and the fresh trace belonging to its run."""

    final_answer: str
    trace: Trace


class HarnessError(RuntimeError):
    """Retain a failed run's partial trace and chained operational exception."""

    def __init__(self, message: str, trace: Trace) -> None:
        """Expose the available trace even when binding or execution fails."""
        super().__init__(message)
        self.trace = trace


def run_in_harness(
    chat_model: BaseChatModel,
    user_prompt: str,
    vault_path: str,
    system_prompt: SystemPrompt = SYSTEM_PROMPT,
    max_api_calls: int = 20,
    model_label: str | None = None,
    tool_binding_kwargs: dict[str, Any] | None = None,
) -> HarnessResult:
    """Return final assistant text and its independent invocation trace.

    Each run binds the registry tools and owns its conversation history. Tool
    outputs, including error envelopes, are returned exactly as ToolMessages.
    Completion requires nonempty text without calls. The call limit includes
    final completion, and tools from the last allowed response still execute.
    Invocation traces retain usage and completed tool results on failure.

    Args:
        chat_model: A synchronous LangChain model supporting tool binding.
        user_prompt: The user's vault question.
        vault_path: Root used by registered tools.
        system_prompt: Versioned instructions replacing the default vault prompt.
        max_api_calls: Maximum model invocations in this run.
        model_label: Label for failed invocations and responses without model
            metadata. Defaults to the chat model's class name.
        tool_binding_kwargs: Optional arguments passed to bind_tools, such as
            strict=True for supporting models.

    Raises:
        ValueError: If the user prompt, vault path, or a supplied model label are
            blank, or the invocation limit is not positive.
            Validation occurs before tool binding or execution.
        HarnessError: If binding, invocation, dispatch, response validation, or
            the call limit fails. The error retains the partial trace and chains
            the original exception.
    """
    validate_harness_options(
        user_prompt=user_prompt,
        vault_path=vault_path,
        max_api_calls=max_api_calls,
        model_label=model_label,
    )
    trace = Trace()
    try:
        return _execute(
            chat_model=chat_model,
            user_prompt=user_prompt,
            vault_path=vault_path,
            system_prompt=system_prompt,
            max_api_calls=max_api_calls,
            model_label=model_label,
            tool_binding_kwargs=tool_binding_kwargs,
            trace=trace,
        )
    except Exception as exc:
        raise HarnessError(str(exc), trace) from exc


def validate_harness_options(
    user_prompt: str,
    vault_path: str,
    max_api_calls: int,
    model_label: str | None,
) -> None:
    """Accept nonblank conversation text and a positive invocation limit.

    A supplied model label must also be nonblank. Argument types are checked
    statically by the annotated interface; filesystem access is left to tools.

    Raises:
        ValueError: If required text or a supplied label is blank, or the
            invocation limit is not positive.
    """
    for name, value in (
        ("user_prompt", user_prompt),
        ("vault_path", vault_path),
    ):
        if not value.strip():
            raise ValueError(f"{name} must be nonblank text")
    if max_api_calls <= 0:
        raise ValueError("max_api_calls must be a positive integer")
    if model_label is not None and not model_label.strip():
        raise ValueError("model_label must be nonblank text or None")


def _execute(
    chat_model: BaseChatModel,
    user_prompt: str,
    vault_path: str,
    system_prompt: SystemPrompt,
    max_api_calls: int,
    model_label: str | None,
    tool_binding_kwargs: dict[str, Any] | None,
    trace: Trace,
) -> HarnessResult:
    """Return a completed conversation, retaining partial execution in trace.

    Raises:
        Exception: If model binding, execution, or response validation fails.
    """
    tools = registry.tool_definitions()
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt.text),
        HumanMessage(content=user_prompt),
    ]
    label = model_label or type(chat_model).__name__
    trace.record_initial_request(
        model=label,
        tools=tools,
        messages=[m.model_dump(mode="json") for m in messages],
        tool_binding_kwargs=tool_binding_kwargs or {},
        prompt_messages=[("system", system_prompt.text), ("user", user_prompt)],
    )
    bound = chat_model.bind_tools(tools, **(tool_binding_kwargs or {}))

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
            if not answer.strip():
                raise RuntimeError(
                    "API response contains neither text nor function calls"
                )
            return HarnessResult(answer, trace)
        messages.append(response)
        for index, call in enumerate(calls):
            output = registry.dispatch(
                name=call.name, arguments_json=call.arguments, vault_path=vault_path
            )
            call_trace.record_tool_result(index, output)
            messages.append(ToolMessage(content=output, tool_call_id=call.call_id))

    raise RuntimeError(f"API call limit ({max_api_calls}) reached")
