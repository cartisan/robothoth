"""Verify native client ownership and session isolation across adapters."""

from typing import cast
from unittest.mock import MagicMock

import pytest
from openai.types.responses import Response, ResponseFunctionToolCall

import src.providers.claude_provider as claude
import src.providers.openai_provider as openai
from src.providers.llm_provider import ToolResult
from src.tools.registry import registry


@pytest.mark.parametrize("kind", ["openai", "claude"])
@pytest.mark.parametrize("fails", [False, True])
def test_owned_client_closes_on_context_exit(
    monkeypatch: pytest.MonkeyPatch, kind: str, fails: bool
) -> None:
    """Close SDK-created clients on both normal and exceptional context exit."""
    client = MagicMock()
    factory = MagicMock(return_value=client)
    error = RuntimeError("run failed")
    if kind == "openai":
        monkeypatch.setattr(openai, "OpenAI", factory)
        provider: openai.OpenAIProvider | claude.ClaudeProvider = (
            openai.OpenAIProvider()
        )
    else:
        monkeypatch.setattr(claude, "Anthropic", factory)
        provider = claude.ClaudeProvider(model="claude", max_tokens=100)
    if fails:
        with pytest.raises(RuntimeError) as raised:
            with provider:
                raise error
        assert raised.value is error
    else:
        with provider as entered:
            assert entered is provider
    factory.assert_called_once_with()
    client.close.assert_called_once_with()


@pytest.mark.parametrize("kind", ["openai", "claude"])
def test_injected_client_remains_open(kind: str) -> None:
    """Leave caller-owned SDK clients open on context exit and explicit close."""
    client = MagicMock()
    if kind == "openai":
        provider: openai.OpenAIProvider | claude.ClaudeProvider = openai.OpenAIProvider(
            client
        )
    else:
        provider = claude.ClaudeProvider(client, model="claude", max_tokens=100)
    with provider:
        pass
    provider.close()
    client.close.assert_not_called()


def test_openai_sessions_keep_history_separate_and_validate_results() -> None:
    """Reject unmatched tool results without contaminating another conversation."""
    client = MagicMock()
    client.responses.create.return_value = Response.model_construct(
        id="response",
        model="returned",
        status="completed",
        usage=None,
        output=[
            ResponseFunctionToolCall(
                type="function_call", name="list_notes", call_id="a", arguments="{}"
            )
        ],
    )
    provider = openai.OpenAIProvider(client)
    first = provider.start_session("System", "First", registry.tool_definitions())
    second = provider.start_session("System", "Second", registry.tool_definitions())
    first.invoke()
    with pytest.raises(ValueError, match="Pending"):
        first.invoke()
    with pytest.raises(ValueError, match="match"):
        first.add_tool_results([ToolResult("wrong", "raw")])
    first.add_tool_results([ToolResult("a", "")])
    with pytest.raises(ValueError, match="match"):
        first.add_tool_results([ToolResult("a", "")])
    first_history = cast(list[dict[str, object]], first.initial_request["input"])
    second_history = cast(list[dict[str, object]], second.initial_request["input"])
    assert first_history[-1] == {
        "type": "function_call_output",
        "call_id": "a",
        "output": "",
    }
    assert second_history == [
        {"role": "developer", "content": "System"},
        {"role": "user", "content": "Second"},
    ]
    assert client.responses.create.call_count == 1
