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
from src.tracing import Trace


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
    trace = Trace()
    assert (
        harness.run_in_harness(
            model,
            user_prompt="Find",
            vault_path="vault",
            trace=trace,
            model_label="requested",
            tool_binding_kwargs={"strict": True},
        )
        == "Done"
    )
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
    trace = Trace()
    assert (
        harness.run_in_harness(
            model, user_prompt="Find", vault_path="vault", trace=trace
        )
        == "Done"
    )
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
    trace = Trace()
    assert (
        harness.run_in_harness(
            model, user_prompt="Find", vault_path="vault", trace=trace
        )
        == "Done"
    )
    assert [call.name for call in trace.calls[0].tool_calls] == [
        "read_note",
        "list_notes",
    ]
    assert [call.arguments for call in trace.calls[0].tool_calls] == [
        "{",
        '{"path":null}',
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
    trace = Trace()
    with pytest.raises(RuntimeError):
        harness.run_in_harness(
            ScriptedChatModel(responses=[response]),
            user_prompt="Find",
            vault_path="vault",
            trace=trace,
        )
    assert len(trace.calls) == 1
    dispatcher.assert_not_called()


def test_run_isolation_and_call_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep histories separate and execute last allowed tool calls."""
    model = ScriptedChatModel(
        responses=[AIMessage(content="Done"), AIMessage(content="Done")]
    )
    trace = Trace()
    for prompt in ("First", "Second"):
        assert (
            harness.run_in_harness(
                model, user_prompt=prompt, vault_path="vault", trace=trace
            )
            == "Done"
        )
    assert [request[1].content for request in model.requests] == ["First", "Second"]
    assert len(trace.calls) == 2
    assert trace.initial_request is not None and 'user: "Second"' in str(
        trace.initial_request
    )

    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="", tool_calls=[{"name": "list_notes", "args": {}, "id": "a"}]
            )
        ]
    )
    monkeypatch.setattr(harness.registry, "dispatch", MagicMock(return_value="saved"))
    limited = Trace()
    with pytest.raises(RuntimeError, match=r"call limit \(1\) reached"):
        harness.run_in_harness(
            model,
            user_prompt="Find",
            vault_path="vault",
            trace=limited,
            max_api_calls=1,
        )
    assert limited.calls[0].tool_calls[0].output == "saved"
    assert len(model.requests) == 1


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limit_has_no_side_effect(limit: int) -> None:
    """Reject invalid limits before tool binding or trace mutation."""
    model = ScriptedChatModel(responses=[])
    trace = Trace()
    trace.record_initial_request(
        model="old",
        tools=[],
        messages=[],
        tool_binding_kwargs={},
        prompt_messages=[("user", "Old")],
    )
    initial = trace.initial_request
    with pytest.raises(ValueError, match="positive"):
        harness.run_in_harness(
            model,
            user_prompt="Find",
            vault_path="vault",
            trace=trace,
            max_api_calls=limit,
        )
    assert model.declarations == []
    assert trace.initial_request is initial


def test_invocation_and_dispatch_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve prior usage and partial tool outputs on exact exceptions."""
    error = ConnectionError("offline")
    first = AIMessage(
        content="",
        tool_calls=[{"name": "list_notes", "args": {}, "id": "a"}],
        usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    )
    model = ScriptedChatModel(responses=[first, error])
    monkeypatch.setattr(harness.registry, "dispatch", MagicMock(return_value="saved"))
    trace = Trace()
    with pytest.raises(ConnectionError) as raised:
        harness.run_in_harness(
            model,
            user_prompt="Find",
            vault_path="vault",
            trace=trace,
            model_label="requested",
        )
    assert raised.value is error
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
    trace = Trace()
    with pytest.raises(OSError) as dispatch_raised:
        harness.run_in_harness(
            model, user_prompt="Find", vault_path="vault", trace=trace
        )
    assert dispatch_raised.value is failure
    assert [call.output for call in trace.calls[0].tool_calls] == ["saved", None]
