"""Exercise the direct LangChain harness with scripted model responses."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

import src.harness as harness
from src.tools.registry import registry
from src.utils import SystemPrompt


class ScriptedChatModel(BaseChatModel):
    """Return scripted messages or errors while capturing each model request."""

    responses: list[Any]
    requests: list[list[BaseMessage]] = Field(default_factory=list)
    declarations: list[dict[str, Any]] = Field(default_factory=list)
    binding_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        """Return the scripted model's LangChain type label."""
        return "scripted"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        """Capture normalized declarations and binding options for this model."""
        self.declarations = [convert_to_openai_tool(tool) for tool in tools]
        self.binding_kwargs = kwargs
        if tool_choice is not None:
            self.binding_kwargs["tool_choice"] = tool_choice
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Return the next response after recording a detached message history.

        Raises:
            Exception: If the next scripted entry is an exception.
        """
        index = len(self.requests)
        self.requests.append(deepcopy(messages))
        item = self.responses[index]
        if isinstance(item, Exception):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


def test_rounds_preserve_messages_results_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain complete AI messages, ordered exact results, and token details."""
    first = AIMessage(
        id="response",
        content=[
            {"type": "reasoning", "reasoning": "opaque"},
            {"type": "text", "text": "Checking"},
            {"type": "function_call", "name": "list_notes"},
        ],
        tool_calls=[
            {"name": "list_notes", "args": {"path": None}, "id": "a"},
            {"name": "read_note", "args": {"notepath": "Beta.md"}, "id": "b"},
        ],
        response_metadata={
            "model_name": "returned-model",
            "finish_reason": "tool_calls",
        },
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_token_details": {"cache_read": 30, "cache_creation": 10},
            "output_token_details": {"reasoning": 5},
        },
    )
    model = ScriptedChatModel(responses=[first, AIMessage(content="Done")])
    outputs = ['{"ok":true,"result":[]}', "note\ncontents"]
    dispatcher = MagicMock(side_effect=outputs)
    monkeypatch.setattr(harness.registry, "dispatch", dispatcher)
    ticks = iter([1.0, 3.0, 100.0, 103.0])
    monkeypatch.setattr(harness, "perf_counter", lambda: next(ticks))
    result = harness.run_in_harness(
        model,
        user_prompt="Find",
        vault_path="vault",
        model_label="requested",
        tool_binding_kwargs={"strict": True},
    )
    assert result.final_answer == "Done"
    trace = result.trace
    assert model.binding_kwargs == {"strict": True}
    assert len(model.declarations) == len(registry.tool_definitions())
    assert model.declarations[0]["function"] == registry.tool_definitions()[0]
    assert model.requests[1][2].content == first.content
    results = model.requests[1][3:]
    assert all(isinstance(result, ToolMessage) for result in results)
    assert [
        result.tool_call_id for result in results if isinstance(result, ToolMessage)
    ] == ["a", "b"]
    assert [result.content for result in results] == outputs
    assert dispatcher.call_count == 2
    assert trace.total_latency() == 5.0
    call = trace.calls[0]
    assert (call.model, call.response_id, call.input_tokens, call.total_tokens) == (
        "returned-model",
        "response",
        100,
        120,
    )
    assert (call.cached_tokens, call.cache_write_tokens, call.reasoning_tokens) == (
        30,
        10,
        5,
    )
    assert [tool.output for tool in call.tool_calls] == outputs
    assert trace.calls[1].model == "requested"
    assert trace.total_cost() is None
    assert trace.initial_request is not None
    assert len(trace.initial_request.arguments["messages"]) == 2  # type: ignore[arg-type]


def test_malformed_arguments_reach_dispatcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send malformed JSON unchanged and return its exact error envelope."""
    first = AIMessage(
        content="",
        invalid_tool_calls=[
            {"name": "list_notes", "args": "{", "id": "bad", "error": "bad JSON"}
        ],
    )
    model = ScriptedChatModel(responses=[first, AIMessage(content="Done")])
    output = '{"ok":false,"error":{"code":"invalid_arguments"}}'
    dispatcher = MagicMock(return_value=output)
    monkeypatch.setattr(harness.registry, "dispatch", dispatcher)
    result = harness.run_in_harness(model, user_prompt="Find", vault_path="vault")
    assert result.final_answer == "Done"
    trace = result.trace
    dispatcher.assert_called_once_with(
        name="list_notes", arguments_json="{", vault_path="vault"
    )
    assert model.requests[1][-1].content == output
    assert trace.calls[0].tool_calls[0].arguments == "{"


def test_mixed_calls_follow_raw_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute mixed parsed and malformed calls in native response order."""
    first = AIMessage(
        content="",
        tool_calls=[{"name": "list_notes", "args": {"path": None}, "id": "good"}],
        invalid_tool_calls=[
            {"name": "read_note", "args": "{", "id": "bad", "error": "bad"}
        ],
        additional_kwargs={
            "tool_calls": [
                {"id": "bad", "function": {"arguments": "{"}},
                {"id": "good", "function": {"arguments": '{"path":null}'}},
            ]
        },
    )
    model = ScriptedChatModel(responses=[first, AIMessage(content="Done")])
    dispatcher = MagicMock(side_effect=["bad result", "good result"])
    monkeypatch.setattr(harness.registry, "dispatch", dispatcher)
    result = harness.run_in_harness(model, user_prompt="Find", vault_path="vault")
    assert result.final_answer == "Done"
    trace = result.trace
    assert [call.name for call in trace.calls[0].tool_calls] == [
        "read_note",
        "list_notes",
    ]
    assert [call.arguments for call in trace.calls[0].tool_calls] == [
        "{",
        '{"path":null}',
    ]


def test_mixed_responses_calls_match_call_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatch Responses calls in output order using call IDs, not item IDs."""
    first = AIMessage(
        content=[
            {"type": "reasoning", "id": "reasoning-item"},
            {
                "type": "function_call",
                "id": "bad-item",
                "call_id": "bad-call",
                "name": "read_note",
                "arguments": "{",
            },
            {
                "type": "function_call",
                "id": "good-item",
                "call_id": "good-call",
                "name": "list_notes",
                "arguments": '{ "path": null }',
            },
        ],
        tool_calls=[{"name": "list_notes", "args": {"path": None}, "id": "good-call"}],
        invalid_tool_calls=[
            {"name": "read_note", "args": "{", "id": "bad-call", "error": "bad"}
        ],
    )
    model = ScriptedChatModel(responses=[first, AIMessage(content="Done")])
    dispatcher = MagicMock(side_effect=["bad result", "good result"])
    monkeypatch.setattr(harness.registry, "dispatch", dispatcher)
    result = harness.run_in_harness(model, user_prompt="Find", vault_path="vault")
    assert result.final_answer == "Done"
    trace = result.trace
    assert [call.call_id for call in trace.calls[0].tool_calls] == [
        "bad-call",
        "good-call",
    ]
    assert [call.arguments for call in trace.calls[0].tool_calls] == [
        "{",
        '{ "path": null }',
    ]
    assert [call.kwargs for call in dispatcher.call_args_list] == [
        {"name": "read_note", "arguments_json": "{", "vault_path": "vault"},
        {
            "name": "list_notes",
            "arguments_json": '{ "path": null }',
            "vault_path": "vault",
        },
    ]
    assert [
        message.tool_call_id
        for message in model.requests[1][3:]
        if isinstance(message, ToolMessage)
    ] == [
        "bad-call",
        "good-call",
    ]


@pytest.mark.parametrize(
    "response",
    [
        AIMessage(content="Partial", response_metadata={"finish_reason": "length"}),
        AIMessage(content="Partial", response_metadata={"status": "incomplete"}),
        AIMessage(content="Partial", response_metadata={"stop_reason": "max_tokens"}),
        AIMessage(content="Partial", additional_kwargs={"refusal": "No"}),
        AIMessage(content=[{"type": "image", "url": "unsupported"}]),
        AIMessage(content="", response_metadata={"finish_reason": "tool_calls"}),
        AIMessage(content=""),
        AIMessage(
            content="", tool_calls=[{"name": "list_notes", "args": {}, "id": ""}]
        ),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "list_notes", "args": {}, "id": "a"},
                {"name": "list_notes", "args": {}, "id": "a"},
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "list_notes", "args": {}, "id": "a"}],
            invalid_tool_calls=[
                {"name": "read_note", "args": "{", "id": "b", "error": "bad"}
            ],
        ),
    ],
)
def test_invalid_response_traced_before_rejection(
    response: AIMessage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trace invalid responses without dispatching their requested tools."""
    dispatcher = MagicMock()
    monkeypatch.setattr(harness.registry, "dispatch", dispatcher)
    with pytest.raises(harness.HarnessError) as caught:
        harness.run_in_harness(ScriptedChatModel(responses=[response]), "Find", "vault")
    assert len(caught.value.trace.calls) == 1
    assert isinstance(caught.value.__cause__, RuntimeError)
    dispatcher.assert_not_called()


def test_run_isolation_custom_prompt_and_call_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep traces and histories separate and execute last allowed tool calls."""
    model = ScriptedChatModel(
        responses=[AIMessage(content="Done"), AIMessage(content="Done")]
    )
    results = [
        harness.run_in_harness(
            chat_model=model,
            user_prompt=prompt,
            vault_path="vault",
            system_prompt=SystemPrompt(version=2, text="Custom instructions"),
        )
        for prompt in ("First", "Second")
    ]
    assert [request[1].content for request in model.requests] == ["First", "Second"]
    assert all(len(request) == 2 for request in model.requests)
    assert all(
        request[0].content == "Custom instructions" for request in model.requests
    )
    assert results[0].trace is not results[1].trace
    assert all(len(result.trace.calls) == 1 for result in results)
    initial = results[0].trace.initial_request
    assert initial is not None and "Custom instructions" in str(initial)

    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="", tool_calls=[{"name": "list_notes", "args": {}, "id": "a"}]
            )
        ]
    )
    monkeypatch.setattr(harness.registry, "dispatch", MagicMock(return_value="saved"))
    with pytest.raises(
        harness.HarnessError, match=r"call limit \(1\) reached"
    ) as caught:
        harness.run_in_harness(model, "Find", "vault", max_api_calls=1)
    assert caught.value.trace.calls[0].tool_calls[0].output == "saved"
    assert len(model.requests) == 1


@pytest.mark.parametrize(argnames="limit", argvalues=[0, -1])
def test_invalid_limit_has_no_side_effect(limit: int) -> None:
    """Reject invalid limits before tool binding or invocation."""
    model = ScriptedChatModel(responses=[])
    with pytest.raises(expected_exception=ValueError, match="positive"):
        harness.run_in_harness(
            chat_model=model,
            user_prompt="Find",
            vault_path="vault",
            max_api_calls=limit,
        )
    assert model.declarations == []
    assert model.requests == []


@pytest.mark.parametrize(
    "changes",
    [
        {"user_prompt": " "},
        {"vault_path": ""},
        {"model_label": ""},
    ],
)
def test_invalid_arguments_rejected_before_binding(changes: dict[str, Any]) -> None:
    model = MagicMock()
    arguments = {"user_prompt": "Find", "vault_path": "vault", **changes}
    with pytest.raises(ValueError):
        harness.run_in_harness(model, **arguments)
    model.bind_tools.assert_not_called()


def test_binding_failure_retains_initial_request() -> None:
    error = ValueError("cannot bind")
    model = MagicMock()
    model.bind_tools.side_effect = error
    with pytest.raises(harness.HarnessError) as caught:
        harness.run_in_harness(model, "Find", "vault")
    assert caught.value.__cause__ is error
    assert caught.value.trace.initial_request is not None
    assert caught.value.trace.calls == []


def test_invocation_and_dispatch_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve prior usage and partial tool outputs with chained exceptions."""
    error = ConnectionError("offline")
    first = AIMessage(
        content="",
        tool_calls=[{"name": "list_notes", "args": {}, "id": "a"}],
        usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    )
    model = ScriptedChatModel(responses=[first, error])
    monkeypatch.setattr(harness.registry, "dispatch", MagicMock(return_value="saved"))
    with pytest.raises(harness.HarnessError) as raised:
        harness.run_in_harness(model, "Find", "vault", model_label="requested")
    assert raised.value.__cause__ is error
    trace = raised.value.trace
    assert trace.calls[0].tool_calls[0].output == "saved"
    assert trace.calls[1].model == "requested"
    assert trace.calls[1].response_id is None
    assert trace.total_cost() is None

    failure = OSError("broken dispatch")
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "list_notes", "args": {}, "id": "a"},
                    {"name": "read_note", "args": {}, "id": "b"},
                ],
            )
        ]
    )
    monkeypatch.setattr(
        harness.registry, "dispatch", MagicMock(side_effect=["saved", failure])
    )
    with pytest.raises(harness.HarnessError) as raised:
        harness.run_in_harness(model, "Find", "vault")
    assert raised.value.__cause__ is failure
    assert [call.output for call in raised.value.trace.calls[0].tool_calls] == [
        "saved",
        None,
    ]


def test_non_ai_response_records_failure() -> None:
    model = MagicMock()
    model.bind_tools.return_value.invoke.return_value = "unexpected"
    with pytest.raises(harness.HarnessError, match="AIMessage") as caught:
        harness.run_in_harness(model, "Find", "vault")
    assert len(caught.value.trace.calls) == 1


def test_whitespace_answer_is_invalid() -> None:
    with pytest.raises(harness.HarnessError):
        harness.run_in_harness(
            ScriptedChatModel(responses=[AIMessage(content=" ")]), "Find", "vault"
        )
