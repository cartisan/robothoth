"""Verify Claude wire translation, continuation, validation, and usage."""

from copy import deepcopy
from typing import cast
from unittest.mock import MagicMock

import httpx2 as httpx
import pytest
from anthropic import APIConnectionError
from anthropic.types import Message

import src.harness as harness
import src.providers.claude_provider as adapter
from src.providers.llm_provider import ToolResult
from src.tools.registry import registry
from src.tracing import Trace


def message(*blocks: dict[str, object], stop_reason: str = "end_turn") -> Message:
    """Return a Messages response with cache usage and supplied content blocks."""
    return Message.model_validate(
        {
            "id": "response",
            "type": "message",
            "role": "assistant",
            "model": "returned-claude",
            "content": list(blocks),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 10,
            },
        }
    )


def tool(call_id: str, name: str = "list_notes") -> dict[str, object]:
    """Return a Claude client-tool block with a nullable path argument."""
    return {"type": "tool_use", "id": call_id, "name": name, "input": {"path": None}}


def test_multiple_tools_and_reasoning_preserve_native_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch tool results immediately after complete assistant content blocks."""
    first = message(
        {"type": "thinking", "thinking": "Private", "signature": "signed"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "Checking"},
        tool("a"),
        tool("b"),
        stop_reason="tool_use",
    )
    responses = iter([first, message({"type": "text", "text": "Done"})])
    requests: list[dict[str, object]] = []

    def create(**kwargs: object) -> Message:
        """Snapshot the request and return the next scripted Claude response."""
        requests.append(deepcopy(kwargs))
        return next(responses)

    client = MagicMock()
    client.messages.create.side_effect = create
    outputs = ['{"ok":true,"result":[]}', '{"ok":false,"error":{"code":"not_found"}}']
    monkeypatch.setattr(harness.registry, "dispatch", MagicMock(side_effect=outputs))
    provider = adapter.ClaudeProvider(client, model="requested-claude", max_tokens=2048)
    trace = Trace()
    assert (
        harness.run_in_harness(
            provider, user_prompt="Find", vault_path="vault", trace=trace
        )
        == "Done"
    )
    assert requests[0]["system"] == harness.SYSTEM_PROMPT
    assert requests[0]["model"] == "requested-claude"
    assert requests[0]["max_tokens"] == 2048
    declarations = cast(list[dict[str, object]], requests[0]["tools"])
    for declaration, definition in zip(
        declarations, registry.tool_definitions(), strict=True
    ):
        assert declaration == {
            "name": definition.name,
            "description": definition.description,
            "input_schema": definition.input_schema,
        }
    history = cast(list[dict[str, object]], requests[1]["messages"])
    assert history[1] == {
        "role": "assistant",
        "content": [b.model_dump() for b in first.content],
    }
    assert history[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "a",
                "content": outputs[0],
                "is_error": False,
            },
            {
                "type": "tool_result",
                "tool_use_id": "b",
                "content": outputs[1],
                "is_error": True,
            },
        ],
    }
    assert trace.initial_request is not None
    assert trace.initial_request.arguments == requests[0]
    assert "system: " in str(trace.initial_request)
    call = trace.calls[0]
    assert call.model == "returned-claude"
    assert call.input_tokens == 140
    assert call.total_tokens == 160
    assert call.cached_tokens == 30
    assert call.cache_write_tokens == 10
    assert call.reasoning_tokens is None
    assert trace.total_cost() == 320
    assert [c.output for c in call.tool_calls] == outputs


@pytest.mark.parametrize(
    "stop", ["max_tokens", "refusal", "pause_turn", "stop_sequence"]
)
def test_noncompleted_stop_reason_retains_trace_without_dispatch(
    monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    """Reject partial output before tools execute, retaining usage and text."""
    client = MagicMock()
    client.messages.create.return_value = message(
        tool("a"), {"type": "text", "text": "Partial"}, stop_reason=stop
    )
    dispatch = MagicMock()
    monkeypatch.setattr(harness.registry, "dispatch", dispatch)
    trace = Trace()
    with pytest.raises(RuntimeError, match=stop):
        harness.run_in_harness(
            adapter.ClaudeProvider(client, model="claude", max_tokens=100),
            user_prompt="Find",
            vault_path="vault",
            trace=trace,
        )
    dispatch.assert_not_called()
    assert trace.calls[0].total_tokens == 160
    assert trace.calls[0].tool_calls[0].output is None


@pytest.mark.parametrize(
    "response",
    [
        message({"type": "text", "text": "Partial"}, stop_reason="tool_use"),
        message(tool("a"), stop_reason="end_turn"),
        message(
            {
                "type": "server_tool_use",
                "id": "server",
                "name": "web_search",
                "input": {},
            }
        ),
    ],
)
def test_inconsistent_or_unsupported_responses_are_errors(response: Message) -> None:
    """Return traceable validation errors for malformed stops or server tools."""
    client = MagicMock()
    client.messages.create.return_value = response
    session = adapter.ClaudeProvider(
        client, model="claude", max_tokens=100
    ).start_session("System", "User", registry.tool_definitions())
    normalized = session.invoke()
    assert normalized.validation_error is not None
    assert normalized.total_tokens == 160


def test_sdk_failure_is_traced_and_propagated() -> None:
    """Keep original Anthropic exceptions and record failed-call metrics."""
    error = APIConnectionError(request=httpx.Request("POST", "https://example.test"))
    client = MagicMock()
    client.messages.create.side_effect = error
    trace = Trace()
    with pytest.raises(APIConnectionError) as raised:
        harness.run_in_harness(
            adapter.ClaudeProvider(client, model="requested", max_tokens=100),
            user_prompt="Find",
            vault_path="vault",
            trace=trace,
        )
    assert raised.value is error
    assert trace.calls[0].model == "requested"
    assert trace.total_cost() is None


def test_session_isolation_and_result_validation() -> None:
    """Reject mismatched batches and retain arbitrary exact output strings."""
    client = MagicMock()
    client.messages.create.side_effect = [
        message(tool("a"), tool("b"), stop_reason="tool_use"),
        message({"type": "text", "text": "Done"}),
    ]
    provider = adapter.ClaudeProvider(client, model="claude", max_tokens=100)
    first = provider.start_session("System", "First", registry.tool_definitions())
    second = provider.start_session("System", "Second", registry.tool_definitions())
    first.invoke()
    with pytest.raises(ValueError, match="Pending"):
        first.invoke()
    with pytest.raises(ValueError, match="match"):
        first.add_tool_results([ToolResult("b", ""), ToolResult("a", "raw")])
    first.add_tool_results([ToolResult("a", ""), ToolResult("b", "raw\ntext")])
    assert first.invoke().text == "Done"
    history = cast(list[dict[str, object]], second.initial_request["messages"])
    assert history == [{"role": "user", "content": "Second"}]
    assert client.messages.create.call_count == 2


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_budget_does_not_create_client(
    monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    """Validate output budgets before client creation."""
    factory = MagicMock()
    monkeypatch.setattr(adapter, "Anthropic", factory)
    with pytest.raises(ValueError, match="positive"):
        adapter.ClaudeProvider(model="claude", max_tokens=limit)
    factory.assert_not_called()
