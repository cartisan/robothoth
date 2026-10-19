"""Deterministic requirements use successful envelopes and all recorded attempts."""

import json
from itertools import permutations

import pytest
from pydantic import ValidationError

from src.evaluation.tool_checks import evaluate_tool_checks
from src.gold_dataset.validate import ToolChecks, load_dataset
from src.tracing import CallTrace, ToolCallTrace, Trace


def attempt(
    name: str,
    output: str | None = '{"ok":true,"result":[]}',
    path: str = "area/Note.md",
) -> ToolCallTrace:
    return ToolCallTrace(name, name, json.dumps({"notepath": path}), output)


def traced(*attempts: ToolCallTrace) -> Trace:
    return Trace(calls=[CallTrace("model", 0, tool_calls=[call]) for call in attempts])


def test_order_independence_repeated_attempts_and_evidence() -> None:
    calls = [attempt("read_note"), attempt("list_notes", None), attempt("list_notes")]
    requirements = ToolChecks(
        required_tools=["list_notes", "read_note"],
        required_note_reads=["area/Note.md"],
    )
    for order in permutations(calls):
        checks = evaluate_tool_checks(requirements, traced(*order))
        assert all(check.passed for check in checks)
        assert len(checks[0].calls) == 2
        for check in checks:
            assert check.explanation
            for ref in check.calls:
                assert order[ref.invocation_index].call_id == ref.call_id
                assert ref.tool_index == 0


@pytest.mark.parametrize(
    argnames="output,passed",
    argvalues=[
        (None, False),
        ('{"ok":false,"error":{"code":"not_found","message":"Missing"}}', False),
        ('{"ok":true,"result":[]}', True),
        ('{"ok":true,"result":""}', True),
    ],
)
def test_required_tools_need_dispatcher_success(
    output: str | None, passed: bool
) -> None:
    checks = evaluate_tool_checks(
        tool_checks=ToolChecks(required_tools=["read_note"]),
        trace=traced(attempt(name="read_note", output=output)),
    )
    assert checks[0].passed is passed
    assert len(checks[0].calls) == 1


@pytest.mark.parametrize(
    "path,expected",
    [
        ("area/Note.md", True),
        ("area\\Note.md", True),
        ("./area/./Note.md", True),
        ("area//Note.md", True),
        ("Note.md", False),
        ("other/Note.md", False),
        ("area/note.md", False),
        ("/area/Note.md", False),
        ("area/../area/Note.md", False),
        ("C:\\area\\Note.md", False),
        ("C:area/Note.md", False),
        ("\\\\server\\area\\Note.md", False),
    ],
)
def test_note_paths_are_full_case_sensitive_and_lexical(
    path: str, expected: bool
) -> None:
    checks = evaluate_tool_checks(
        ToolChecks(required_note_reads=["area/Note.md"]),
        traced(attempt("read_note", path=path)),
    )
    assert checks[0].passed is expected


@pytest.mark.parametrize("arguments", ["{", "null", "[]", "{}", '{"notepath":2}'])
def test_invalid_read_arguments_do_not_match(arguments: str) -> None:
    call = attempt("read_note")
    call.arguments = arguments
    assert not evaluate_tool_checks(
        ToolChecks(required_note_reads=["area/Note.md"]), traced(call)
    )[0].passed


def test_forbidden_and_allowlist_count_unexecuted_attempts() -> None:
    checks = evaluate_tool_checks(
        ToolChecks(
            forbidden_tools=["read_note"],
            allowed_tools=["list_notes"],
        ),
        traced(attempt("read_note", None), attempt("unknown_tool", None)),
    )
    assert not any(check.passed for check in checks)
    assert len(checks[0].calls) == 1
    assert len(checks[1].calls) == 2
    assert evaluate_tool_checks(ToolChecks(), Trace()) == ()
    assert evaluate_tool_checks(ToolChecks(allowed_tools=[]), Trace())[0].passed


def test_required_note_read_needs_success_and_correct_tool() -> None:
    checks = ToolChecks(required_note_reads=["area/Note.md"])
    assert not evaluate_tool_checks(checks, traced(attempt("list_notes")))[0].passed
    assert not evaluate_tool_checks(checks, traced(attempt("read_note", None)))[
        0
    ].passed


@pytest.mark.parametrize(
    "value",
    [
        {"required_tools": ["read_note"], "forbidden_tools": ["read_note"]},
        {"required_tools": ["list_notes"], "allowed_tools": []},
        {"required_note_reads": ["Note.md"], "forbidden_tools": ["read_note"]},
        {"required_note_reads": ["Note.md"], "allowed_tools": ["list_notes"]},
        {"required_note_reads": ["/Note.md"]},
        {"required_note_reads": ["../Note.md"]},
        {"required_note_reads": ["a.txt"]},
        {"required_note_reads": ["a.md", "./a.md"]},
        {"required_tools": [" "]},
        {"required_tools": ["read_note", "read_note"]},
        {"required_tools": "read_note"},
        {"required_tools": [2]},
        {"allowed_tools": [True]},
        {"extra": []},
        {"required_tools": None},
    ],
)
def test_invalid_or_contradictory_requirements_rejected(
    value: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ToolChecks.model_validate(value)


def test_migrated_dataset_expectations() -> None:
    search, multi, synthesis = load_dataset(version=1)
    assert search.tool_checks.required_tools == ["list_notes"]
    assert search.tool_checks.allowed_tools == ["list_notes"]
    assert search.ground_truth.ideal_answer and not search.ground_truth.answer_criteria
    assert multi.tool_checks.required_tools == ["list_notes", "read_note"]
    assert multi.ground_truth.ideal_answer
    assert multi.ground_truth.answer_criteria
    assert synthesis.ground_truth.ideal_answer
    assert not synthesis.ground_truth.answer_criteria
    assert synthesis.tool_checks.required_note_reads == [
        "2 areas/ai engineering/evaluation/Rubric Based Evaluation.md",
        "2 areas/ai engineering/evaluation/LLM-as-a-Judge.md",
    ]
