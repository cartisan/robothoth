from copy import deepcopy
from typing import cast

import pytest

import src.tracing as tracing
from src.providers.llm_provider import ModelResponse, ToolCall


def test_record_response_without_a_client() -> None:
    """Retain tool requests and missing usage before response validation."""
    response = ModelResponse(
        response_id="response",
        model="returned-model",
        validation_error="incomplete",
        tool_calls=(ToolCall("a", "list_notes", '{ "path": null }'),),
    )
    trace = tracing.Trace()
    call = trace.record_response(response, elapsed_seconds=2.5)
    assert trace.calls == [call]
    assert call.response_id == "response"
    assert call.model == "returned-model"
    assert call.elapsed_seconds == 2.5
    assert call.total_tokens is None
    assert call.output == 'list_notes({ "path": null })'
    assert call.tool_calls == [
        tracing.ToolCallTrace("a", "list_notes", '{ "path": null }')
    ]
    call.record_tool_result(0, "result")
    assert trace.calls[0].tool_calls[0].output == "result"


def test_record_failure_without_a_client() -> None:
    """Record supplied failure metrics without a response or client."""
    trace = tracing.Trace()
    call = trace.record_failure(model="requested-model", elapsed_seconds=3)
    assert trace.calls == [call]
    assert call == tracing.CallTrace(model="requested-model", elapsed_seconds=3)
    assert trace.total_cost() is None
    assert trace.total_latency() == 3


def test_request_trace_snapshots_arguments_and_formats_all_messages() -> None:
    """Copy all request data and print every message's text on one line."""
    arguments: dict[str, object] = {
        "model": "hidden-model",
        "tools": [{"name": "hidden-tool"}],
        "input": [
            {"role": "developer", "content": 'first\nline\tquoted "text"'},
            {"role": "user", "content": "Question"},
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Earlier "},
                    {"type": "input_image", "image_url": "hidden-image"},
                    {"type": "input_text", "text": "answer"},
                ],
            },
            {"role": "user", "content": "Follow-up"},
            {"type": "function_call", "name": "hidden-call"},
        ],
    }
    expected = deepcopy(arguments)
    request = tracing.RequestTrace(arguments)
    cast(list[dict[str, object]], arguments["input"])[0]["content"] = "changed"
    cast(list[dict[str, object]], arguments["tools"])[0]["name"] = "changed"
    assert request.arguments == expected
    assert str(request) == (
        'developer: "first\\nline\\tquoted \\"text\\"" | user: "Question" | '
        'assistant: "Earlier answer" | user: "Follow-up"'
    )
    assert len(str(request).splitlines()) == 1


def test_request_trace_formats_string_input() -> None:
    """Display a shorthand API input as user text without other arguments."""
    assert str(tracing.RequestTrace({"input": "Hello\nworld", "model": "hidden"})) == (
        'user: "Hello\\nworld"'
    )


@pytest.mark.parametrize(
    ("total_tokens", "output", "expected_cost", "expected_output"),
    [
        (120, "Done", "120 tokens", "Done"),
        (0, "", "0 tokens", ""),
        (None, None, "unknown", "(no output)"),
        (
            None,
            "list_notes({})\nread_note({})",
            "unknown",
            "list_notes({}) read_note({})",
        ),
        (120, "first\r\nsecond\rthird", "120 tokens", "first second third"),
    ],
)
def test_call_trace_string(
    total_tokens: int | None,
    output: str | None,
    expected_cost: str,
    expected_output: str,
) -> None:
    """Display token cost, rounded latency, and available output explicitly."""
    call = tracing.CallTrace(
        model="test-model",
        elapsed_seconds=1.23456,
        total_tokens=total_tokens,
        output=output,
    )
    assert str(call) == (
        "Model: test-model\t"
        "Latency: 1.235s\t"
        f"Total cost: {expected_cost}\t"
        f"Output: {expected_output}"
    )
    assert call.output == output


def test_tool_trace_string_and_independent_lists() -> None:
    """Flatten displayed tool results and distinguish empty from missing output."""
    call = tracing.CallTrace(model="model", elapsed_seconds=0)
    other = tracing.CallTrace(model="model", elapsed_seconds=0)
    call.tool_calls.extend(
        [
            tracing.ToolCallTrace("a", "read_note", "{}", "first\nsecond"),
            tracing.ToolCallTrace("b", "read_note", "{}", ""),
            tracing.ToolCallTrace("c", "read_note", "{}"),
        ]
    )
    assert other.tool_calls == []
    assert str(call).split("\n", 1)[1] == (
        "\tName: read_note\tResult: first second\n"
        "\tName: read_note\tResult: \n"
        "\tName: read_note\tResult: (not executed)"
    )
    assert call.tool_calls[0].output == "first\nsecond"


def test_empty_trace_string() -> None:
    """Display zero metrics when no API request has been recorded."""
    assert str(tracing.Trace()) == ("Total tokens: 0\nTotal API latency: 0.000s")


def test_trace_formatting_preserves_stored_output() -> None:
    """Flatten model text for display without changing the caller's output."""
    output = "First line\nSecond line"
    call = tracing.CallTrace(model="model", elapsed_seconds=1, output=output)
    trace = tracing.Trace(calls=[call])
    assert "Output: First line Second line\nTotal tokens: unknown" in str(trace)
    assert call.output == output
