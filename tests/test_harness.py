"""Exercise orchestration without any provider SDK or wire-format assumptions."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

import src.harness as harness
from src.providers.llm_provider import (
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from src.tracing import Trace


@dataclass
class ScriptedSession:
    """Return scripted normalized responses and retain ordered result batches."""

    responses: list[ModelResponse | Exception]
    model: str = "requested-model"
    initial_request: dict[str, object] = field(
        default_factory=lambda: {"provider_payload": {"prompt": "Question"}}
    )
    prompt_messages: tuple[tuple[str, str], ...] = (("user", "Question"),)
    batches: list[list[ToolResult]] = field(default_factory=list)
    invocations: int = 0

    def invoke(self) -> ModelResponse:
        """Return the next response or raise its scripted exception.

        Raises:
            Exception: If the next script entry is an exception.
        """
        item = self.responses[self.invocations]
        self.invocations += 1
        if isinstance(item, Exception):
            raise item
        return item

    def add_tool_results(self, results: Sequence[ToolResult]) -> None:
        """Retain the exact ordered result batch supplied by the harness."""
        self.batches.append(list(results))


@dataclass
class ScriptedProvider:
    """Return a fresh scripted session and retain the supplied neutral tools."""

    responses: list[ModelResponse | Exception]
    sessions: list[ScriptedSession] = field(default_factory=list)
    tools: Sequence[ToolDefinition] = ()

    def start_session(
        self, system_prompt: str, user_prompt: str, tools: Sequence[ToolDefinition]
    ) -> ScriptedSession:
        """Return independent state with the supplied initial prompt."""
        self.tools = tools
        session = ScriptedSession(
            list(self.responses),
            prompt_messages=(("system", system_prompt), ("user", user_prompt)),
        )
        self.sessions.append(session)
        return session


def test_generic_loop_retains_order_results_and_invocation_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use neutral responses, preserving exact results and excluding dispatch time."""
    first = ModelResponse(
        model="returned-model",
        text="Working",
        total_tokens=12,
        tool_calls=(
            ToolCall("a", "read_note", '{"notepath":"A.md"}'),
            ToolCall("b", "list_notes", "{"),
        ),
    )
    provider = ScriptedProvider([first, ModelResponse("returned-model", text="Done")])
    dispatcher = MagicMock(side_effect=['{"ok":true,"result":"note"}', ""])
    monkeypatch.setattr(harness.registry, "dispatch", dispatcher)
    ticks = iter([1.0, 3.0, 100.0, 103.0])
    monkeypatch.setattr(harness, "perf_counter", lambda: next(ticks))
    trace = Trace()
    assert (
        harness.run_in_harness(
            provider, user_prompt="Find it", vault_path="vault", trace=trace
        )
        == "Done"
    )
    session = provider.sessions[0]
    assert session.batches == [
        [ToolResult("a", '{"ok":true,"result":"note"}'), ToolResult("b", "")]
    ]
    assert provider.tools == harness.registry.tool_definitions()
    assert trace.total_latency() == 5.0
    assert trace.calls[0].output == 'read_note({"notepath":"A.md"})\nlist_notes({)'
    assert trace.calls[0].tool_calls[1].output == ""
    assert trace.initial_request is not None
    assert trace.initial_request.arguments == session.initial_request
    session.initial_request.clear()
    assert "provider_payload" in trace.initial_request.arguments
    assert 'user: "Find it"' in str(trace)


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limit_does_not_create_session_or_change_trace(limit: int) -> None:
    """Reject invalid limits before any provider or trace side effect."""
    provider = ScriptedProvider([])
    trace = Trace()
    trace.record_initial_request({"input": "Old"})
    initial = trace.initial_request
    with pytest.raises(ValueError, match="positive"):
        harness.run_in_harness(
            provider,
            user_prompt="Find",
            vault_path="vault",
            trace=trace,
            max_api_calls=limit,
        )
    assert provider.sessions == []
    assert trace.initial_request is initial
    assert trace.calls == []


@pytest.mark.parametrize(
    "response,match",
    [
        (
            ModelResponse("model", text="Partial", validation_error="truncated"),
            "truncated",
        ),
        (ModelResponse("model"), "neither text nor function calls"),
        (ModelResponse("model", tool_calls=(ToolCall("", "list_notes", "{}"),)), "IDs"),
        (ModelResponse("model", tool_calls=(ToolCall("a", "", "{}"),)), "names"),
        (
            ModelResponse(
                "model",
                tool_calls=(
                    ToolCall("a", "list_notes", "{}"),
                    ToolCall("a", "read_note", "{}"),
                ),
            ),
            "unique",
        ),
    ],
)
def test_invalid_responses_are_traced_before_rejection(
    monkeypatch: pytest.MonkeyPatch, response: ModelResponse, match: str
) -> None:
    """Retain invalid response usage and requests without dispatching tools."""
    provider = ScriptedProvider([response])
    dispatcher = MagicMock()
    monkeypatch.setattr(harness.registry, "dispatch", dispatcher)
    trace = Trace()
    with pytest.raises(RuntimeError, match=match):
        harness.run_in_harness(
            provider, user_prompt="Find", vault_path="vault", trace=trace
        )
    dispatcher.assert_not_called()
    assert len(trace.calls) == 1
    assert len(trace.calls[0].tool_calls) == len(response.tool_calls)


def test_last_allowed_tools_execute_and_remain_in_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute final-round tools without making another model invocation."""
    provider = ScriptedProvider(
        [ModelResponse("model", tool_calls=(ToolCall("a", "list_notes", "{}"),))]
    )
    monkeypatch.setattr(harness.registry, "dispatch", MagicMock(return_value="result"))
    trace = Trace()
    with pytest.raises(RuntimeError, match=r"call limit \(1\) reached"):
        harness.run_in_harness(
            provider,
            user_prompt="Find",
            vault_path="vault",
            trace=trace,
            max_api_calls=1,
        )
    assert provider.sessions[0].invocations == 1
    assert trace.calls[0].tool_calls[0].output == "result"
    assert provider.sessions[0].batches == [[ToolResult("a", "result")]]


def test_dispatch_failure_retains_all_calls_and_completed_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop dispatch on failure while preserving preceding outputs and requests."""
    error = OSError("broken dispatch")
    provider = ScriptedProvider(
        [
            ModelResponse(
                "model",
                tool_calls=(
                    ToolCall("a", "list_notes", "{}"),
                    ToolCall("b", "read_note", "{}"),
                    ToolCall("c", "read_note", "{}"),
                ),
            )
        ]
    )
    dispatcher = MagicMock(side_effect=["saved", error])
    monkeypatch.setattr(harness.registry, "dispatch", dispatcher)
    trace = Trace()
    with pytest.raises(OSError) as raised:
        harness.run_in_harness(
            provider, user_prompt="Find", vault_path="vault", trace=trace
        )
    assert raised.value is error
    assert [c.output for c in trace.calls[0].tool_calls] == ["saved", None, None]
    assert dispatcher.call_count == 2
    assert provider.sessions[0].batches == []


def test_failure_keeps_original_exception_and_earlier_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trace failed latency with the requested model and propagate exact errors."""
    error = ConnectionError("offline")
    provider = ScriptedProvider(
        [
            ModelResponse(
                "returned-model",
                total_tokens=12,
                tool_calls=(ToolCall("a", "list_notes", "{}"),),
            ),
            error,
        ]
    )
    monkeypatch.setattr(harness.registry, "dispatch", MagicMock(return_value="saved"))
    ticks = iter([1.0, 2.0, 10.0, 13.0])
    monkeypatch.setattr(harness, "perf_counter", lambda: next(ticks))
    trace = Trace()
    with pytest.raises(ConnectionError) as raised:
        harness.run_in_harness(
            provider, user_prompt="Find", vault_path="vault", trace=trace
        )
    assert raised.value is error
    assert trace.calls[0].total_tokens == 12
    assert trace.calls[0].tool_calls[0].output == "saved"
    assert trace.calls[1].model == "requested-model"
    assert trace.calls[1].response_id is None
    assert trace.total_cost() is None
    assert trace.total_latency() == 4.0


def test_reused_provider_and_trace_start_independent_sessions() -> None:
    """Replace initial prompts while retaining previous trace calls."""
    provider = ScriptedProvider([ModelResponse("model", text="Done")])
    trace = Trace()
    for prompt in ("First", "Second"):
        harness.run_in_harness(
            provider, user_prompt=prompt, vault_path="vault", trace=trace
        )
    assert provider.sessions[0] is not provider.sessions[1]
    assert len(trace.calls) == 2
    assert trace.initial_request is not None
    assert 'user: "Second"' in str(trace.initial_request)
