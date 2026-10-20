"""Repeat summaries expose the evidence behind scores, failures, and errors."""

from typing import Any

import pytest

from src.evaluation.reporting import (
    refresh_summary,
    repeat_token_usage,
    summary_markdown,
)


def render_repeat(result: dict[str, Any]) -> str:
    """Render a one-repeat report with its computed aggregate counts."""
    result = {
        "elapsed_seconds": 0.0,
        "candidate_tokens": None,
        "judge_tokens": None,
        "total_tokens": None,
        **result,
    }
    report = {
        "run_id": "example-run",
        "case_ids": [result["case_id"]],
        "repeats": 1,
        "expected_repeats": 1,
        "results": [result],
        "elapsed_seconds": 12.5,
    }
    refresh_summary(report=report)
    return summary_markdown(report=report)


def test_summary_explains_every_check_and_judge_criterion() -> None:
    explanation = (
        "The answer covers the main idea | but omits escalation.\n"
        "A different phrasing would be fine. <Missing essential content>"
    )
    summary = render_repeat(
        result={
            "case_id": "mixed-requirements",
            "repeat": 2,
            "status": "failed",
            "elapsed_seconds": 2.25,
            "candidate_tokens": 100,
            "judge_tokens": 50,
            "total_tokens": 150,
            "tool_checks": [
                {
                    "criterion_id": "required_tools[0]",
                    "passed": True,
                    "explanation": "Found successful list_notes call.",
                    "calls": [
                        {
                            "invocation_index": 0,
                            "tool_index": 0,
                            "call_id": "failed-read",
                        },
                        {
                            "invocation_index": 1,
                            "tool_index": 2,
                            "call_id": "successful-read",
                        },
                    ],
                },
                {
                    "criterion_id": "required_note_reads[0]",
                    "passed": False,
                    "explanation": "Missing successful read of area/Note.md.",
                    "calls": [],
                },
            ],
            "scores": [
                {
                    "criterion_id": "ideal_answer",
                    "score": 3,
                    "explanation": explanation,
                },
                {
                    "criterion_id": "quality.concision",
                    "score": 4,
                    "explanation": "The answer is concise.",
                },
                {
                    "criterion_id": "quality.task_usefulness",
                    "score": 5,
                    "explanation": "Directly usable.",
                },
            ],
            "error": None,
        }
    )
    assert "### mixed-requirements, repeat 2: failed" in summary
    assert "Total cost: 150 tokens (candidate: 100 tokens; judge: 50 tokens)" in summary
    assert "Total time: 12.500 s" in summary
    assert "Cost: 150 tokens (candidate: 100 tokens; judge: 50 tokens)" in summary
    assert "Time: 2.250 s" in summary
    assert "| required_tools[0] | passed | Found successful list_notes call." in summary
    assert "invocation 1, tool 1 (failed-read)" in summary
    assert "invocation 2, tool 3 (successful-read)" in summary
    assert (
        "| required_note_reads[0] | failed | "
        "Missing successful read of area/Note.md. | None |"
    ) in summary
    assert "| ideal_answer | 3 / 5 | failed |" in summary
    assert "| quality.concision | 4 / 5 | passed | The answer is concise. |" in summary
    assert "| quality.task_usefulness | 5 / 5 | passed | Directly usable. |" in summary
    assert (
        "The answer covers the main idea \\| but omits escalation.<br>"
        "A different phrasing would be fine. &lt;Missing essential content&gt;"
    ) in summary


@pytest.mark.parametrize(
    argnames="status,error_type,message,cause,unavailable",
    argvalues=[
        (
            "harness_error",
            "HarnessError",
            "Execution failed",
            {"type": "ConnectionError", "message": "Provider unavailable"},
            "Not graded because harness execution failed.",
        ),
        (
            "judge_error",
            "JudgeError",
            "Duplicate criterion IDs",
            None,
            "No valid judge ratings available.",
        ),
    ],
)
def test_summary_distinguishes_errors_from_scored_failures(
    status: str,
    error_type: str,
    message: str,
    cause: dict[str, str] | None,
    unavailable: str,
) -> None:
    summary = render_repeat(
        result={
            "case_id": "error-case",
            "repeat": 1,
            "status": status,
            "tool_checks": [],
            "scores": [],
            "error": {"type": error_type, "message": message, "cause": cause},
        }
    )
    assert "No tool checks configured." in summary
    assert unavailable in summary
    assert f"Error: {error_type}: {message}" in summary
    assert "Pass rate: 0 / 0 (null)" in summary
    assert "| Criterion |" not in summary
    if cause:
        assert "Caused by: ConnectionError: Provider unavailable" in summary


@pytest.mark.parametrize(
    argnames="status,candidate_usage,raw_judge,candidate_tokens,judge_tokens,total_tokens",
    argvalues=[
        ("passed", [30, 70], {"usage_metadata": {"total_tokens": 50}}, 100, 50, 150),
        ("judge_error", [30], {"usage_metadata": {"total_tokens": 15}}, 30, 15, 45),
        ("harness_error", [20, None], None, None, 0, None),
        ("harness_error", [], None, 0, 0, 0),
        ("judge_error", [20], None, 20, None, None),
        ("passed", [20], {"usage_metadata": None}, 20, None, None),
        ("passed", [20], {"usage_metadata": {"total_tokens": True}}, 20, None, None),
        ("passed", [None], {"usage_metadata": {"total_tokens": 10}}, None, 10, None),
    ],
)
def test_repeat_cost_counts_both_models_and_preserves_unknown_usage(
    status: str,
    candidate_usage: list[int | None],
    raw_judge: Any,
    candidate_tokens: int | None,
    judge_tokens: int | None,
    total_tokens: int | None,
) -> None:
    usage = repeat_token_usage(
        result={
            "status": status,
            "trace": {"calls": [{"total_tokens": value} for value in candidate_usage]},
            "judge_calls": []
            if status == "harness_error"
            else [{"stage": "content", "raw_response": raw_judge}],
        }
    )
    assert usage == {
        "candidate_tokens": candidate_tokens,
        "judge_tokens": judge_tokens,
        "total_tokens": total_tokens,
    }


@pytest.mark.parametrize(
    argnames="tokens,expected",
    argvalues=[([10, 20, 30], 60), ([10, 20], 30), ([10, None], None), ([], 0)],
)
def test_cost_sums_all_attempted_calls_and_preserves_unknown_usage(
    tokens: list[int | None],
    expected: int | None,
) -> None:
    """Include earlier responses on failed grades and unknown failed-call costs."""
    stages = ["content", "concision", "task_usefulness"]
    result = repeat_token_usage(
        result={
            "trace": {"calls": []},
            "judge_calls": [
                {
                    "stage": stages[index],
                    "raw_response": (
                        {"usage_metadata": {"total_tokens": value}}
                        if value is not None
                        else None
                    ),
                }
                for index, value in enumerate(tokens)
            ],
        }
    )
    assert result == {
        "candidate_tokens": 0,
        "judge_tokens": expected,
        "total_tokens": expected,
    }
