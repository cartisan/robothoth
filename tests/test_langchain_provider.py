"""Verify tool-capable chat model integration without provider-specific packages."""

import json
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

import src.harness as harness
from src.providers.langchain_provider import LangChainProvider
from src.providers.llm_provider import ToolResult
from src.tools.registry import registry
from src.tracing import Trace


class ScriptedChatModel(BaseChatModel):
    """Return supplied AI messages and capture bound tools and request history."""

    responses: list[AIMessage]
    requests: list[list[BaseMessage]] = Field(default_factory=list)
    declarations: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        """Return the label required by the LangChain chat model contract."""
        return "scripted"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        """Bind schemas through LangChain's actual tool conversion helper."""
        self.declarations = [convert_to_openai_tool(tool) for tool in tools]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Capture native history and return the next scripted message."""
        response = self.responses[len(self.requests)]
        self.requests.append(deepcopy(messages))
        return ChatResult(generations=[ChatGeneration(message=response)])


def test_native_messages_tools_and_usage_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve reasoning metadata while translating ordered calls and results."""
    first = AIMessage(
        id="response",
        content=[
            {"type": "reasoning", "reasoning": "private", "signature": "opaque"},
            {"type": "text", "text": "Checking"},
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
    provider = LangChainProvider(model, model_label="requested-model")
    outputs = ['{"ok":true,"result":[]}', "note\ncontents"]
    monkeypatch.setattr(harness.registry, "dispatch", MagicMock(side_effect=outputs))
    trace = Trace()
    assert (
        harness.run_in_harness(
            provider, user_prompt="Find", vault_path="vault", trace=trace
        )
        == "Done"
    )
    assert len(model.declarations) == len(registry.tool_definitions())
    for declaration, tool in zip(
        model.declarations, registry.tool_definitions(), strict=True
    ):
        assert declaration["function"] == {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
    history = model.requests[1]
    assert history[2].content == first.content
    assert history[2].response_metadata == first.response_metadata
    results = history[3:]
    assert all(isinstance(result, ToolMessage) for result in results)
    assert [
        result.tool_call_id for result in results if isinstance(result, ToolMessage)
    ] == ["a", "b"]
    assert [result.content for result in results] == outputs
    call = trace.calls[0]
    assert call.model == "returned-model"
    assert call.input_tokens == 100
    assert call.total_tokens == 120
    assert call.cached_tokens == 30
    assert call.cache_write_tokens == 10
    assert call.reasoning_tokens == 5
    assert call.response_id == "response"
    assert trace.calls[1].model == "requested-model"
    assert trace.total_cost() is None
    assert trace.initial_request is not None
    assert len(trace.initial_request.arguments["messages"]) == 2  # type: ignore[arg-type]
    assert 'user: "Find"' in str(trace.initial_request)


def test_malformed_arguments_are_returned_to_the_model() -> None:
    """Retain invalid JSON and send dispatcher errors as matching tool messages."""
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "name": "list_notes",
                        "args": "{",
                        "id": "bad",
                        "error": "bad JSON",
                    }
                ],
            ),
            AIMessage(content="Retry"),
        ]
    )
    trace = Trace()
    assert (
        harness.run_in_harness(
            LangChainProvider(model, model_label="test"),
            user_prompt="Find",
            vault_path="vault",
            trace=trace,
        )
        == "Retry"
    )
    result = model.requests[1][-1]
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "bad"
    assert isinstance(result.content, str)
    assert json.loads(result.content)["error"]["code"] == "invalid_arguments"
    assert trace.calls[0].tool_calls[0].arguments == "{"


def test_mixed_valid_and_invalid_calls_recover_native_order() -> None:
    """Follow original call order rather than LangChain's split call lists."""
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": "bad",
                            "type": "function",
                            "function": {"name": "list_notes", "arguments": "{"},
                        },
                        {
                            "id": "good",
                            "type": "function",
                            "function": {
                                "name": "list_notes",
                                "arguments": '{"path":null}',
                            },
                        },
                    ]
                },
            )
        ]
    )
    session = LangChainProvider(model, model_label="test").start_session(
        "System", "User", registry.tool_definitions()
    )
    response = session.invoke()
    assert response.validation_error is None
    assert [call.call_id for call in response.tool_calls] == ["bad", "good"]
    assert response.tool_calls[0].arguments == "{"
    assert response.tool_calls[1].arguments == '{"path":null}'


@pytest.mark.parametrize(
    "response",
    [
        AIMessage(content="Partial", response_metadata={"finish_reason": "length"}),
        AIMessage(content="Partial", response_metadata={"stop_reason": "max_tokens"}),
        AIMessage(
            content="Partial", response_metadata={"finish_reason": "content_filter"}
        ),
        AIMessage(content="Partial", additional_kwargs={"refusal": "Refused"}),
        AIMessage(content="", response_metadata={"finish_reason": "tool_calls"}),
        AIMessage(content=[{"type": "image", "url": "unsupported"}]),
        AIMessage(
            content="",
            tool_calls=[{"name": "list_notes", "args": {}, "id": "a"}],
            invalid_tool_calls=[
                {"name": "list_notes", "args": "{", "id": "b", "error": "bad"}
            ],
        ),
    ],
)
def test_invalid_output_is_traced_before_rejection(response: AIMessage) -> None:
    """Reject invalid stops, refusals, content, or unrecoverable call order."""
    model = ScriptedChatModel(responses=[response])
    trace = Trace()
    with pytest.raises(RuntimeError):
        harness.run_in_harness(
            LangChainProvider(model, model_label="test"),
            user_prompt="Find",
            vault_path="vault",
            trace=trace,
        )
    assert len(trace.calls) == 1
    assert len(model.requests) == 1


def test_session_isolation_and_result_validation() -> None:
    """Keep messages separate and reject missing or reordered result batches."""
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="", tool_calls=[{"name": "list_notes", "args": {}, "id": "a"}]
            ),
            AIMessage(content="Done"),
        ]
    )
    provider = LangChainProvider(model, model_label="test")
    first = provider.start_session("System", "First", registry.tool_definitions())
    second = provider.start_session("System", "Second", registry.tool_definitions())
    first.invoke()
    with pytest.raises(ValueError, match="Pending"):
        first.invoke()
    with pytest.raises(ValueError, match="match"):
        first.add_tool_results([ToolResult("wrong", "")])
    first.add_tool_results([ToolResult("a", "")])
    assert first.invoke().text == "Done"
    assert len(second.initial_request["messages"]) == 2  # type: ignore[arg-type]
    assert len(model.requests) == 2
    assert model.requests[1][-1].content == ""


def test_wrong_message_type_and_invocation_errors() -> None:
    """Normalize wrong return types and propagate original invocation exceptions."""
    model = MagicMock(spec=BaseChatModel)
    model.bind_tools.return_value = RunnableLambda(lambda _: "wrong")
    provider = LangChainProvider(model, model_label="test")
    trace = Trace()
    with pytest.raises(RuntimeError, match="AIMessage"):
        harness.run_in_harness(
            provider, user_prompt="Find", vault_path="vault", trace=trace
        )
    assert len(trace.calls) == 1
    error = ConnectionError("offline")

    def fail(_: object) -> AIMessage:
        """Raise the original provider error.

        Raises:
            ConnectionError: Always, using the captured instance.
        """
        raise error

    model.bind_tools.return_value = RunnableLambda(fail)
    with pytest.raises(ConnectionError) as raised:
        harness.run_in_harness(
            provider, user_prompt="Find", vault_path="vault", trace=trace
        )
    assert raised.value is error
    assert trace.calls[1].model == "test"
    assert trace.calls[1].response_id is None
    assert trace.total_cost() is None


def test_model_without_tool_support_is_rejected_before_invocation() -> None:
    """Expose tool-binding failures without claiming an API invocation happened."""
    model = MagicMock(spec=BaseChatModel)
    model.bind_tools.side_effect = NotImplementedError("tools unsupported")
    trace = Trace()
    with pytest.raises(NotImplementedError, match="tools unsupported"):
        harness.run_in_harness(
            LangChainProvider(model, model_label="test"),
            user_prompt="Find",
            vault_path="vault",
            trace=trace,
        )
    assert trace.calls == []
    assert trace.initial_request is None
