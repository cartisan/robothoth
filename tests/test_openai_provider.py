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

import src.harness as provider
import src.main as example
import src.providers.openai_provider as openai_adapter
import src.tracing as tracing
from src.tools.registry import registry


def run_with_openai(
    client: MagicMock,
    *,
    user_prompt: str,
    vault_path: str,
    trace: tracing.Trace,
    model: str = "gpt-6-luna",
    max_api_calls: int = 20,
) -> str:
    """Return a harness answer using an injected scripted OpenAI client.

    Raises:
        Exception: Harness and scripted client failures propagate unchanged.
    """
    return provider.run_in_harness(
        openai_adapter.OpenAIProvider(client, model=model),
        user_prompt=user_prompt,
        vault_path=vault_path,
        trace=trace,
        max_api_calls=max_api_calls,
    )


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


def test_provider_uses_registered_declarations() -> None:
    """Advertise every registered vault function with its derived schema."""
    client = MagicMock()
    client.responses.create.return_value = make_response(text="Done")
    run_with_openai(
        client, user_prompt="Question", vault_path="vault", trace=tracing.Trace()
    )
    assert client.responses.create.call_args.kwargs[
        "tools"
    ] == openai_adapter.tool_declarations(registry.tool_definitions())


def test_immediate_text_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return immediate text and record usage and API-only elapsed time."""
    client = MagicMock()
    client.responses.create.return_value = make_response(text="Done")
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(provider, "perf_counter", lambda: next(ticks))
    trace = tracing.Trace()

    assert (
        run_with_openai(
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
        tracing.CallTrace(
            model="returned-model",
            response_id="response",
            elapsed_seconds=2.5,
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            cached_tokens=30,
            cache_write_tokens=10,
            reasoning_tokens=5,
            output="Done",
        )
    ]
    assert trace.total_cost() == 120
    assert trace.total_latency() == 2.5
    assert trace.initial_request is not None
    assert trace.initial_request.arguments == {
        "model": "requested-model",
        "tools": openai_adapter.tool_declarations(registry.tool_definitions()),
        "input": [
            {"role": "developer", "content": provider.SYSTEM_PROMPT},
            {"role": "user", "content": "Question"},
        ],
    }


def test_multiple_rounds_preserve_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every tool call and preserve reasoning, text, and matching IDs."""
    reasoning = ResponseReasoningItem(
        id="reasoning", type="reasoning", summary=[], encrypted_content="opaque"
    )
    responses = iter(
        [
            make_response(
                reasoning,
                tool_call("a", '{ "path": null }'),
                ResponseFunctionToolCall(
                    type="function_call",
                    name="read_note",
                    arguments='{"notepath": "Beta.md"}',
                    call_id="b",
                ),
                text="Checking",
            ),
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
    tool_results = ['{"ok": true, "result": []}', "note\ncontents", ""]
    dispatcher = MagicMock(side_effect=tool_results)
    monkeypatch.setattr(provider.registry, "dispatch", dispatcher)
    ticks = iter([1.0, 2.0, 50.0, 52.0, 90.0, 93.0])
    monkeypatch.setattr(provider, "perf_counter", lambda: next(ticks))
    trace = tracing.Trace()

    assert (
        run_with_openai(client, user_prompt="Find it", vault_path="vault", trace=trace)
        == "Found it"
    )

    assert len(requests) == 3
    assert trace.initial_request is not None
    assert trace.initial_request.arguments == requests[0]
    initial = cast(list[dict[str, object]], requests[0]["input"])
    assert initial == [
        {"role": "developer", "content": provider.SYSTEM_PROMPT},
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
    assert [call.output for call in trace.calls] == [
        'list_notes({ "path": null })\nread_note({"notepath": "Beta.md"})',
        "list_notes({})",
        "Found it",
    ]
    assert trace.calls[0].tool_calls == [
        tracing.ToolCallTrace("a", "list_notes", '{ "path": null }', tool_results[0]),
        tracing.ToolCallTrace(
            "b", "read_note", '{"notepath": "Beta.md"}', tool_results[1]
        ),
    ]
    assert trace.calls[1].tool_calls == [
        tracing.ToolCallTrace("c", "list_notes", "{}", tool_results[2])
    ]
    assert trace.calls[2].tool_calls == []
    assert [
        item["output"] for item in third if item.get("type") == "function_call_output"
    ] == tool_results


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
    trace = tracing.Trace()
    run_with_openai(client, user_prompt="Find", vault_path="vault", trace=trace)
    history = cast(list[dict[str, object]], requests[1]["input"])
    result = json.loads(cast(str, history[-1]["output"]))
    assert history[-1]["call_id"] == "bad"
    assert result["error"]["code"] == "invalid_arguments"
    assert trace.calls[0].output == "list_notes({)"
    assert trace.calls[0].tool_calls == [
        tracing.ToolCallTrace(
            "bad", "list_notes", "{", cast(str, history[-1]["output"])
        )
    ]


def test_call_limit_retains_last_tool_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute and trace the last tools without exceeding the API call cap."""
    client = MagicMock()
    client.responses.create.return_value = make_response(tool_call("again"))
    dispatcher = MagicMock(return_value="{}")
    monkeypatch.setattr(provider.registry, "dispatch", dispatcher)
    trace = tracing.Trace()
    with pytest.raises(RuntimeError, match="call limit \\(2\\) reached"):
        run_with_openai(
            client, user_prompt="Find", vault_path="vault", trace=trace, max_api_calls=2
        )
    assert client.responses.create.call_count == 2
    assert dispatcher.call_count == 2
    assert len(trace.calls) == 2
    assert trace.calls[-1].output == "list_notes({})"
    assert trace.calls[0].tool_calls[0].output == "{}"
    assert trace.calls[-1].tool_calls[0].output == "{}"


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limit_makes_no_requests(limit: int) -> None:
    """Reject nonpositive limits before invoking the API."""
    client = MagicMock()
    trace = tracing.Trace()
    with pytest.raises(ValueError, match="positive"):
        run_with_openai(
            client,
            user_prompt="Find",
            vault_path="vault",
            trace=trace,
            max_api_calls=limit,
        )
    client.responses.create.assert_not_called()
    assert trace.calls == []
    assert trace.initial_request is None


@pytest.mark.parametrize("status", ["failed", "incomplete", "cancelled"])
def test_noncompleted_responses_are_errors(status: str) -> None:
    """Reject noncompleted responses even when they contain partial text."""
    client = MagicMock()
    client.responses.create.return_value = make_response(text="Partial", status=status)
    trace = tracing.Trace()
    with pytest.raises(RuntimeError, match=status):
        run_with_openai(client, user_prompt="Find", vault_path="vault", trace=trace)
    assert trace.total_cost() == 120
    assert trace.calls[0].output == "Partial"
    client.responses.create.assert_called_once()


def test_empty_response_is_error() -> None:
    """Reject a completed response without text or function calls."""
    client = MagicMock()
    client.responses.create.return_value = make_response()
    trace = tracing.Trace()
    with pytest.raises(RuntimeError, match="neither text nor function calls"):
        run_with_openai(client, user_prompt="Find", vault_path="vault", trace=trace)
    client.responses.create.assert_called_once()
    assert trace.calls[0].output is None


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
    monkeypatch.setattr(provider.registry, "dispatch", dispatcher)
    trace = tracing.Trace()
    with pytest.raises(RuntimeError, match="Unsupported.*custom_tool_call"):
        run_with_openai(client, user_prompt="Find", vault_path="vault", trace=trace)
    dispatcher.assert_not_called()
    assert trace.calls[0].output == "list_notes({})"
    assert trace.calls[0].tool_calls == [
        tracing.ToolCallTrace("valid", "list_notes", "{}")
    ]


def test_sdk_failure_retains_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain earlier usage and failed-call latency while propagating SDK errors."""
    client = MagicMock()
    error = APIConnectionError(request=httpx.Request("POST", "https://example.test"))
    client.responses.create.side_effect = [make_response(tool_call("a")), error]
    monkeypatch.setattr(provider.registry, "dispatch", MagicMock(return_value="result"))
    ticks = iter([1.0, 2.0, 10.0, 13.0])
    monkeypatch.setattr(provider, "perf_counter", lambda: next(ticks))
    trace = tracing.Trace()
    with pytest.raises(APIConnectionError) as raised:
        run_with_openai(client, user_prompt="Find", vault_path="vault", trace=trace)
    assert raised.value is error
    assert len(trace.calls) == 2
    assert trace.calls[0].total_tokens == 120
    assert trace.calls[0].output == "list_notes({})"
    assert trace.calls[1].response_id is None
    assert trace.calls[1].total_tokens is None
    assert trace.calls[1].output is None
    assert trace.total_cost() is None
    assert trace.total_latency() == 4.0
    assert trace.calls[0].tool_calls[0].output == "result"
    assert trace.calls[1].tool_calls == []
    assert trace.initial_request is not None
    assert trace.initial_request.arguments["input"] == [
        {"role": "developer", "content": provider.SYSTEM_PROMPT},
        {"role": "user", "content": "Find"},
    ]


def test_reused_trace_replaces_initial_request() -> None:
    """Replace the prompt for valid runs and preserve it for invalid limits."""
    client = MagicMock()
    client.responses.create.return_value = make_response(text="Done")
    trace = tracing.Trace()
    run_with_openai(client, user_prompt="First", vault_path="vault", trace=trace)
    first = trace.initial_request
    run_with_openai(client, user_prompt="Second", vault_path="vault", trace=trace)
    assert trace.initial_request is not None
    assert trace.initial_request is not first
    assert 'user: "Second"' in str(trace.initial_request)
    assert len(trace.calls) == 2
    current = trace.initial_request
    with pytest.raises(ValueError):
        run_with_openai(
            client,
            user_prompt="Invalid",
            vault_path="vault",
            trace=trace,
            max_api_calls=0,
        )
    assert trace.initial_request is current
    assert len(trace.calls) == 2


def test_dispatch_failure_retains_earlier_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep completed results and all requests when a later dispatch raises."""
    client = MagicMock()
    client.responses.create.return_value = make_response(
        tool_call("a"), tool_call("b"), tool_call("c")
    )
    error = RuntimeError("dispatch failed")
    dispatcher = MagicMock(side_effect=["saved result", error])
    monkeypatch.setattr(provider.registry, "dispatch", dispatcher)
    trace = tracing.Trace()

    with pytest.raises(RuntimeError, match="dispatch failed") as raised:
        run_with_openai(client, user_prompt="Find", vault_path="vault", trace=trace)

    assert raised.value is error
    assert trace.calls[0].tool_calls == [
        tracing.ToolCallTrace("a", "list_notes", "{}", "saved result"),
        tracing.ToolCallTrace("b", "list_notes", "{}"),
        tracing.ToolCallTrace("c", "list_notes", "{}"),
    ]
    assert dispatcher.call_count == 2
    client.responses.create.assert_called_once()


def test_missing_usage_and_independent_traces() -> None:
    """Keep missing usage unknown and avoid sharing call lists between traces."""
    trace = tracing.Trace()
    other = tracing.Trace()
    assert trace.total_cost() == 0
    assert trace.total_latency() == 0.0
    client = MagicMock()
    client.responses.create.return_value = make_response(text="Done", usage=False)
    run_with_openai(client, user_prompt="Find", vault_path="vault", trace=trace)
    assert trace.total_cost() is None
    assert trace.calls[0].input_tokens is None
    assert trace.calls[0].output == "Done"
    assert other.calls == []
    assert other.initial_request is None
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
    factory.return_value = client
    monkeypatch.setattr(example, "load_dotenv", dotenv)
    monkeypatch.setattr(openai_adapter, "OpenAI", factory)
    if fails:
        with pytest.raises(APIConnectionError):
            example.main()
    else:
        example.main()
    dotenv.assert_called_once_with()
    client.close.assert_called_once()
    output = capsys.readouterr().out
    initial_line = next(
        line for line in output.splitlines() if line.startswith("Initial request:")
    )
    expected_prompt = (
        "Help me locate the file called 'Agentic Software Engineering Factory'."
    )
    assert initial_line == (
        "Initial request: "
        f"developer: {json.dumps(provider.SYSTEM_PROMPT, ensure_ascii=False)} | "
        f"user: {json.dumps(expected_prompt, ensure_ascii=False)}"
    )
    assert output.index("Initial request:") < output.index("API call 1:")
    assert "API call 1:" in output
    assert "Model:" in output
    assert "Latency:" in output
    assert f"Total cost: {'unknown' if fails else '120 tokens'}" in output
    assert f"Output: {'(no output)' if fails else 'Final answer'}" in output
    assert "Total API latency:" in output
    assert f"Total tokens: {'unknown' if fails else '120'}" in output
    if not fails:
        assert "Final answer" in output
