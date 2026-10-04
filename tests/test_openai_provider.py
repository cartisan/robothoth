import json
from copy import deepcopy
from typing import cast
from unittest.mock import MagicMock

import httpx2 as httpx
import pytest
from openai import APIConnectionError
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseUsage,
)
from openai.types.responses.response_output_item import ResponseOutputItem

import src.openai_provider as provider


def make_response(
    *items: ResponseOutputItem,
    text: str = "",
    usage: bool = True,
    status: str = "completed",
) -> Response:
    """Return a synthetic SDK response with optional text and known usage."""
    output = list(items)
    if text:
        output.append(
            ResponseOutputMessage(
                id="message",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(type="output_text", text=text, annotations=[])
                ],
            )
        )
    return Response.model_construct(
        id="response",
        model="returned-model",
        status=status,
        output=output,
        usage=(
            ResponseUsage.model_validate(
                {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "input_tokens_details": {
                        "cached_tokens": 30,
                        "cache_write_tokens": 10,
                    },
                    "output_tokens_details": {"reasoning_tokens": 5},
                }
            )
            if usage
            else None
        ),
    )


def tool_call(call_id: str, arguments: str = "{}") -> ResponseFunctionToolCall:
    """Return a list_notes function call with the supplied ID and arguments."""
    return ResponseFunctionToolCall(
        type="function_call",
        name="list_notes",
        arguments=arguments,
        call_id=call_id,
    )


def test_immediate_text_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return immediate text and record usage and API-only elapsed time."""
    client = MagicMock()
    client.responses.create.return_value = make_response(text="Done")
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(provider, "perf_counter", lambda: next(ticks))
    trace = provider.Trace()

    assert (
        provider.run(
            client,
            user_prompt="Question",
            vault_path="vault",
            trace=trace,
            model="requested-model",
            max_api_calls=1,
        )
        == "Done"
    )

    client.responses.create.assert_called_once()
    assert client.responses.create.call_args.kwargs["model"] == "requested-model"
    assert trace.calls == [
        provider.CallTrace(
            model="returned-model",
            response_id="response",
            elapsed_seconds=2.5,
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            cached_tokens=30,
            cache_write_tokens=10,
            reasoning_tokens=5,
        )
    ]
    assert trace.total_cost() == 120
    assert trace.total_latency() == 2.5


def test_multiple_rounds_preserve_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every tool call and preserve reasoning, text, and matching IDs."""
    reasoning = ResponseReasoningItem(
        id="reasoning", type="reasoning", summary=[], encrypted_content="opaque"
    )
    responses = iter(
        [
            make_response(reasoning, tool_call("a"), tool_call("b"), text="Checking"),
            make_response(tool_call("c")),
            make_response(text="Found it"),
        ]
    )
    requests: list[dict[str, object]] = []

    def create(**kwargs: object) -> Response:
        """Capture a request snapshot and return the next scripted response."""
        requests.append(deepcopy(kwargs))
        return next(responses)

    client = MagicMock()
    client.responses.create.side_effect = create
    dispatcher = MagicMock(return_value='{"ok": true, "result": []}')
    monkeypatch.setattr(provider.dispatch, "dispatch_tool_call", dispatcher)
    ticks = iter([1.0, 2.0, 50.0, 52.0, 90.0, 93.0])
    monkeypatch.setattr(provider, "perf_counter", lambda: next(ticks))
    trace = provider.Trace()

    assert (
        provider.run(client, user_prompt="Find it", vault_path="vault", trace=trace)
        == "Found it"
    )

    assert len(requests) == 3
    initial = cast(list[dict[str, object]], requests[0]["input"])
    assert initial == [
        {"role": "developer", "content": provider.system_prompt},
        {"role": "user", "content": "Find it"},
    ]
    second = cast(list[dict[str, object]], requests[1]["input"])
    assert second[2] == reasoning.to_dict()
    assert [item["type"] for item in second[2:]] == [
        "reasoning",
        "function_call",
        "function_call",
        "message",
        "function_call_output",
        "function_call_output",
    ]
    third = cast(list[dict[str, object]], requests[2]["input"])
    assert third[: len(second)] == second
    assert [
        item["call_id"] for item in third if item.get("type") == "function_call_output"
    ] == ["a", "b", "c"]
    assert dispatcher.call_count == 3
    dispatcher.assert_called_with(
        name="list_notes", arguments_json="{}", vault_path="vault"
    )
    assert trace.total_cost() == 360
    assert trace.total_latency() == 6.0


def test_tool_errors_are_sent_to_model() -> None:
    """Preserve dispatcher error envelopes for malformed model arguments."""
    requests: list[dict[str, object]] = []
    responses = iter(
        [make_response(tool_call("bad", "{")), make_response(text="Please retry")]
    )

    def create(**kwargs: object) -> Response:
        """Capture the next request and return a scripted response."""
        requests.append(deepcopy(kwargs))
        return next(responses)

    client = MagicMock()
    client.responses.create.side_effect = create
    provider.run(client, user_prompt="Find", vault_path="vault", trace=provider.Trace())
    history = cast(list[dict[str, object]], requests[1]["input"])
    result = json.loads(cast(str, history[-1]["output"]))
    assert history[-1]["call_id"] == "bad"
    assert result["error"]["code"] == "invalid_arguments"


def test_call_limit_stops_before_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop at the configured API cap without dispatching the last tools."""
    client = MagicMock()
    client.responses.create.return_value = make_response(tool_call("again"))
    dispatcher = MagicMock(return_value="{}")
    monkeypatch.setattr(provider.dispatch, "dispatch_tool_call", dispatcher)
    trace = provider.Trace()
    with pytest.raises(RuntimeError, match="call limit \\(2\\) reached"):
        provider.run(
            client, user_prompt="Find", vault_path="vault", trace=trace, max_api_calls=2
        )
    assert client.responses.create.call_count == 2
    assert dispatcher.call_count == 1
    assert len(trace.calls) == 2


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limit_makes_no_requests(limit: int) -> None:
    """Reject nonpositive limits before invoking the API."""
    client = MagicMock()
    trace = provider.Trace()
    with pytest.raises(ValueError, match="positive"):
        provider.run(
            client,
            user_prompt="Find",
            vault_path="vault",
            trace=trace,
            max_api_calls=limit,
        )
    client.responses.create.assert_not_called()
    assert trace.calls == []


@pytest.mark.parametrize("status", ["failed", "incomplete", "cancelled"])
def test_noncompleted_responses_are_errors(status: str) -> None:
    """Reject noncompleted responses even when they contain partial text."""
    client = MagicMock()
    client.responses.create.return_value = make_response(text="Partial", status=status)
    trace = provider.Trace()
    with pytest.raises(RuntimeError, match=status):
        provider.run(client, user_prompt="Find", vault_path="vault", trace=trace)
    assert trace.total_cost() == 120
    client.responses.create.assert_called_once()


def test_empty_response_is_error() -> None:
    """Reject a completed response without text or function calls."""
    client = MagicMock()
    client.responses.create.return_value = make_response()
    with pytest.raises(RuntimeError, match="neither text nor function calls"):
        provider.run(
            client, user_prompt="Find", vault_path="vault", trace=provider.Trace()
        )
    client.responses.create.assert_called_once()


def test_unsupported_tool_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unsupported tool calls before dispatch or final-text completion."""
    from openai.types.responses import ResponseCustomToolCall

    client = MagicMock()
    client.responses.create.return_value = make_response(
        tool_call("valid"),
        ResponseCustomToolCall(
            type="custom_tool_call", call_id="custom", name="custom", input="data"
        ),
        text="Partial",
    )
    dispatcher = MagicMock()
    monkeypatch.setattr(provider.dispatch, "dispatch_tool_call", dispatcher)
    with pytest.raises(RuntimeError, match="Unsupported.*custom_tool_call"):
        provider.run(
            client, user_prompt="Find", vault_path="vault", trace=provider.Trace()
        )
    dispatcher.assert_not_called()


def test_sdk_failure_retains_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain earlier usage and failed-call latency while propagating SDK errors."""
    client = MagicMock()
    error = APIConnectionError(request=httpx.Request("POST", "https://example.test"))
    client.responses.create.side_effect = [make_response(tool_call("a")), error]
    monkeypatch.setattr(provider.dispatch, "dispatch_tool_call", MagicMock())
    ticks = iter([1.0, 2.0, 10.0, 13.0])
    monkeypatch.setattr(provider, "perf_counter", lambda: next(ticks))
    trace = provider.Trace()
    with pytest.raises(APIConnectionError) as raised:
        provider.run(client, user_prompt="Find", vault_path="vault", trace=trace)
    assert raised.value is error
    assert len(trace.calls) == 2
    assert trace.calls[0].total_tokens == 120
    assert trace.calls[1].response_id is None
    assert trace.calls[1].total_tokens is None
    assert trace.total_cost() is None
    assert trace.total_latency() == 4.0


def test_missing_usage_and_independent_traces() -> None:
    """Keep missing usage unknown and avoid sharing call lists between traces."""
    trace = provider.Trace()
    other = provider.Trace()
    assert trace.total_cost() == 0
    assert trace.total_latency() == 0.0
    client = MagicMock()
    client.responses.create.return_value = make_response(text="Done", usage=False)
    provider.run(client, user_prompt="Find", vault_path="vault", trace=trace)
    assert trace.total_cost() is None
    assert trace.calls[0].input_tokens is None
    assert other.calls == []
    assert other.total_cost() == 0


@pytest.mark.parametrize("fails", [False, True])
def test_main_prints_metrics_and_closes_client(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fails: bool
) -> None:
    """Load dotenv in main and print metrics on successful and failed runs."""
    dotenv = MagicMock()
    client = MagicMock()
    if fails:
        client.responses.create.side_effect = APIConnectionError(
            request=httpx.Request("POST", "https://example.test")
        )
    else:
        client.responses.create.return_value = make_response(text="Final answer")
    factory = MagicMock()
    factory.return_value.__enter__.return_value = client
    monkeypatch.setattr(provider, "load_dotenv", dotenv)
    monkeypatch.setattr(provider, "OpenAI", factory)
    if fails:
        with pytest.raises(APIConnectionError):
            provider.main()
    else:
        provider.main()
    dotenv.assert_called_once_with()
    factory.return_value.__exit__.assert_called_once()
    output = capsys.readouterr().out
    assert "API call 1:" in output
    assert "Total API latency:" in output
    assert f"Total tokens: {'unknown' if fails else '120'}" in output
    if not fails:
        assert "Final answer" in output
