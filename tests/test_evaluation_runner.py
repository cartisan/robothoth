"""Whole-dataset runs retain errors and repeats without aggregate verdicts."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from test_gold_dataset import entry, write_dataset
from test_harness import ScriptedChatModel

import src.evaluation.__main__ as cli
import src.evaluation.runner as runner
from src.evaluation import (
    DEFAULT_JUDGE_PROMPT,
    SemanticJudge,
    SystemPrompt,
    run_evaluation,
)
from src.evaluation.config import ModelConfig
from src.evaluation.judge import ExpectationScore, SemanticGrade
from src.evaluation.reporting import without_credentials
from src.harness import SYSTEM_PROMPT


def judge_with(*responses: Any) -> SemanticJudge:
    model = MagicMock()
    model.model_dump.return_value = {"model": "judge"}
    raw = AIMessage(content="raw judge output")
    model.with_structured_output.return_value.invoke.side_effect = [
        response
        if isinstance(response, Exception)
        else {
            "raw": raw,
            "parsed": {
                "scores": [
                    {
                        "expectation_id": "ideal_answer",
                        "score": response,
                        "explanation": "Meaning checked",
                    }
                ]
            },
            "parsing_error": None,
        }
        for response in responses
    ]
    return SemanticJudge(model)


@pytest.mark.parametrize(argnames="schema_version", argvalues=[0, 1, 2])
def test_all_cases_repeats_thresholds_and_version_metadata(
    tmp_path: Path,
    schema_version: int,
) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(
        path=path,
        entries=[entry(id="later", dataset_version=3), entry(id="earlier")],
        schema_version=schema_version,
    )
    candidate = ScriptedChatModel(
        responses=[AIMessage(content=str(i)) for i in range(4)]
    )
    judge = SemanticJudge(
        model=judge_with(4, 3, 5, 4).model,
        prompt=SystemPrompt(version=3, text="Judge instructions"),
    )
    run = run_evaluation(
        candidate_model=candidate,
        system_prompt=SystemPrompt(version=2, text="Candidate prompt"),
        vault_path=tmp_path,
        judge=judge,
        dataset_path=path,
        repeats=2,
        report_destination=tmp_path / "runs",
    )
    report = json.loads((run.run_directory / "report.json").read_text())
    assert report == run.report
    assert report["schema_version"] == schema_version
    assert report["gold_dataset_version"] == 3
    assert report["case_ids"] == ["later", "earlier"]
    assert [(item["case_id"], item["repeat"]) for item in report["results"]] == [
        ("later", 1),
        ("later", 2),
        ("earlier", 1),
        ("earlier", 2),
    ]
    assert [item["status"] for item in report["results"]] == [
        "passed",
        "failed",
        "passed",
        "passed",
    ]
    assert report["complete"] and report["summary"]["pass_rate"] == 0.75
    assert report["summary"]["scored"] == 4
    assert report["case_summaries"][0]["pass_rate"] == 0.5
    assert all(
        "status" not in case and "verdict" not in case
        for case in report["case_summaries"]
    )
    assert report["report_version"] == 2
    assert report["candidate_system_prompt"] == {
        "version": 2,
        "text": "Candidate prompt",
    }
    assert report["judge_prompt"] == judge.prompt.model_dump()
    assert report["judge_prompt"]["version"] == 3
    assert report["dataset_path"] == str(path.resolve())
    assert "entries" not in report and "dataset_hash" not in report
    assert report["finished_at"] and report["run_id"]
    assert len(candidate.requests) == 4
    assert all(len(request) == 2 for request in candidate.requests)
    assert all(
        request[0].content == "Candidate prompt" for request in candidate.requests
    )
    assert run.exit_code == 1
    summary = (run.run_directory / "summary.md").read_text()
    assert "3 / 4" in summary and "75.0%" in summary and "repeat 2: failed" in summary
    assert "| ideal_answer | 3 / 5 | failed | Meaning checked |" in summary
    assert "| ideal_answer | 4 / 5 | passed | Meaning checked |" in summary


def test_errors_continue_and_exclude_error_denominators(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(path, [entry(id=f"case-{i}") for i in range(3)])
    candidate = ScriptedChatModel(
        responses=[
            AIMessage(content="First"),
            ConnectionError("offline"),
            AIMessage(content="Last"),
        ]
    )
    judge = judge_with(5, "invalid integer")
    run = run_evaluation(
        candidate,
        vault_path=tmp_path,
        judge=judge,
        dataset_path=path,
        report_destination=tmp_path,
    )
    report = run.report
    assert [item["status"] for item in report["results"]] == [
        "passed",
        "harness_error",
        "judge_error",
    ]
    assert report["summary"]["completed"] == 3
    assert report["summary"]["errors"] == 2
    assert report["summary"]["scored"] == 1
    assert report["summary"]["pass_rate"] == 1.0
    assert not report["complete"] and run.exit_code == 2
    harness_failure, judge_failure = report["results"][1:]
    assert harness_failure["trace"]["calls"][0]["output"] is None
    assert harness_failure["error"]["cause"]["type"] == "ConnectionError"
    assert judge_failure["final_answer"] == "Last"
    assert judge_failure["raw_judge_response"]["content"] == "raw judge output"
    assert judge_failure["scores"] == []
    assert report["case_summaries"][1]["pass_rate"] is None
    assert report["summary"]["total_tokens"] is None
    assert all(item["elapsed_seconds"] >= 0 for item in report["results"])


def test_tool_failures_still_grade_and_partial_trace_retained(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(path, [entry(tool_checks={"required_tools": ["list_notes"]})])
    candidate = ScriptedChatModel(responses=[AIMessage(content="Answer")])
    judge = judge_with(5)
    run = run_evaluation(
        candidate,
        vault_path=tmp_path,
        judge=judge,
        dataset_path=path,
        report_destination=tmp_path,
    )
    result = run.report["results"][0]
    assert result["status"] == "failed"
    assert result["scores"][0]["score"] == 5
    assert not result["tool_checks"][0]["passed"]
    assert run.report["complete"]
    summary = (run.run_directory / "summary.md").read_text()
    assert (
        "| required_tools[0] | failed | Missing successful list_notes call."
    ) in summary
    assert "| ideal_answer | 5 / 5 | passed | Meaning checked |" in summary

    candidate = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "list_notes", "args": {"path": None}, "id": "a"}],
            )
        ]
    )
    run = run_evaluation(
        candidate,
        vault_path=tmp_path,
        judge=judge,
        dataset_path=path,
        max_api_calls=1,
        report_destination=tmp_path,
    )
    result = run.report["results"][0]
    assert result["status"] == "harness_error"
    assert result["tool_checks"][0]["passed"]
    assert result["trace"]["calls"][0]["tool_calls"][0]["output"]
    assert run.report["summary"]["pass_rate"] is None


def test_initial_report_and_atomic_updates_after_each_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(path, [entry()])
    snapshots = []
    original = runner.save_report

    def capture(directory: Path, report: dict[str, Any]) -> None:
        original(directory, report)
        stored = json.loads((directory / "report.json").read_text())
        snapshots.append(stored)
        assert (directory / "summary.md").is_file()
        assert sorted(item.name for item in directory.iterdir()) == [
            "report.json",
            "summary.md",
        ]

    monkeypatch.setattr(runner, "save_report", capture)
    run = run_evaluation(
        ScriptedChatModel(responses=[AIMessage(content="A"), AIMessage(content="B")]),
        vault_path=tmp_path,
        judge=judge_with(4, 5),
        dataset_path=path,
        repeats=2,
        report_destination=tmp_path,
    )
    assert [len(item["results"]) for item in snapshots] == [0, 1, 2]
    assert [item["complete"] for item in snapshots] == [False, False, True]
    assert snapshots[0]["summary"]["pass_rate"] is None
    assert snapshots[0]["finished_at"] is None
    assert run.exit_code == 0
    second = run_evaluation(
        ScriptedChatModel(responses=[AIMessage(content="C")]),
        vault_path=tmp_path,
        judge=judge_with(4),
        dataset_path=path,
        report_destination=tmp_path,
    )
    assert second.run_directory != run.run_directory


@pytest.mark.parametrize(
    "changes",
    [
        {"repeats": 0},
        {"repeats": True},
        {"max_api_calls": 0},
        {"vault_path": "/nonexistent-evaluation-vault"},
    ],
)
def test_invalid_setup_before_model_calls(
    changes: dict[str, Any],
    tmp_path: Path,
) -> None:
    candidate = MagicMock()
    judge = judge_with(5)
    options = {
        "vault_path": tmp_path,
        "judge": judge,
        "report_destination": tmp_path,
        **changes,
    }
    with pytest.raises(ValueError):
        run_evaluation(candidate, **options)
    candidate.bind_tools.assert_not_called()
    judge.model.with_structured_output.assert_not_called()  # type: ignore[attr-defined]


def test_empty_and_invalid_later_entries_rejected_before_calls(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    candidate = MagicMock()
    for entries in [
        [],
        [entry(), entry(id="invalid", dataset_version=10, request=" ")],
    ]:
        write_dataset(path, entries)
        with pytest.raises(ValueError):
            run_evaluation(
                candidate,
                vault_path=tmp_path,
                dataset_path=path,
                report_destination=tmp_path,
            )
    candidate.bind_tools.assert_not_called()


def test_credentials_excluded_from_nested_configuration(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(path, [entry()])
    settings = {
        "provider": "openai",
        "settings": {
            "api_key": "fixture-api-credential-9381",
            "max_tokens": 4096,
            "nested": {
                "Authorization": "fixture-api-credential-9381",
                "temperature": 0.1,
            },
        },
    }
    run = run_evaluation(
        ScriptedChatModel(responses=[AIMessage(content="A")]),
        vault_path=tmp_path,
        judge=judge_with(4),
        dataset_path=path,
        report_destination=tmp_path,
        candidate_config=settings,
        judge_config=settings,
    )
    stored = (run.run_directory / "report.json").read_text()
    assert "fixture-api-credential-9381" not in stored
    assert run.report["candidate_config"]["settings"]["max_tokens"] == 4096
    assert without_credentials({"API-Key": "sensitive", "token": "sensitive"}) == {}


@pytest.mark.parametrize(
    "response,grade,exit_code",
    [
        (AIMessage(content="A"), 4, 0),
        (AIMessage(content="A"), 3, 1),
        (ConnectionError("offline"), 4, 2),
        (AIMessage(content="A"), "bad", 2),
    ],
)
def test_cli_exit_codes(
    response: Any,
    grade: Any,
    exit_code: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(path, [entry()])
    candidate = ScriptedChatModel(responses=[response])
    judge = judge_with(grade)
    monkeypatch.setattr(
        target=ModelConfig,
        name="create_model",
        value=lambda self: candidate,
    )
    monkeypatch.setattr(target=cli, name="SemanticJudge", value=lambda **kwargs: judge)
    assert (
        cli.main(
            [
                "--dataset-path",
                str(path),
                "--vault-path",
                str(tmp_path),
                "--output-directory",
                str(tmp_path / "reports"),
            ]
        )
        == exit_code
    )


def test_cli_independent_provider_settings_and_bundled_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(path, [entry()])
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "candidate": {
                    "provider": "anthropic",
                    "model": "candidate",
                    "settings": {"max_tokens": 1000},
                    "tool_binding_kwargs": {"tool_choice": "auto"},
                },
                "judge": {
                    "provider": "openai",
                    "model": "judge",
                    "settings": {"reasoning_effort": "medium"},
                    "structured_output_kwargs": {"method": "function_calling"},
                },
            }
        )
    )
    candidate = ScriptedChatModel(responses=[AIMessage(content="A")])
    judge = judge_with(5)
    models = []

    def create(config: ModelConfig) -> Any:
        models.append(config)
        return candidate if config.model == "candidate" else judge.model

    monkeypatch.setattr(target=ModelConfig, name="create_model", value=create)
    assert (
        cli.main(
            [
                "--config",
                str(config),
                "--dataset-path",
                str(path),
                "--vault-path",
                str(tmp_path),
                "--max-api-calls",
                "7",
                "--output-directory",
                str(tmp_path / "reports"),
            ]
        )
        == 0
    )
    assert [model.provider for model in models] == ["anthropic", "openai"]
    assert models[0].settings == {"max_tokens": 1000}
    assert models[1].settings == {"reasoning_effort": "medium"}
    assert candidate.requests[0][0].content == SYSTEM_PROMPT.text
    messages = judge.model.with_structured_output.return_value.invoke.call_args.args[0]  # type: ignore[attr-defined]
    assert messages[0].content == DEFAULT_JUDGE_PROMPT.text
    assert candidate.binding_kwargs == {"tool_choice": "auto"}
    report_path = next((tmp_path / "reports").glob("*/report.json"))
    report = json.loads(report_path.read_text())
    assert report["candidate_system_prompt"] == SYSTEM_PROMPT.model_dump()
    assert report["judge_prompt"] == DEFAULT_JUDGE_PROMPT.model_dump()
    assert report["harness_settings"]["max_api_calls"] == 7


def test_cli_setup_errors_never_construct_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = MagicMock()
    monkeypatch.setattr(target=ModelConfig, name="create_model", value=create)
    path = tmp_path / "dataset.json"
    write_dataset(path, [])
    assert cli.main(["--dataset-path", str(path)]) == 2
    config = tmp_path / "config.json"
    config.write_text('{"unknown":true}')
    assert cli.main(["--config", str(config)]) == 2
    assert cli.main(["--case-id", "example"]) == 2
    assert cli.main(["--dataset-version", "1"]) == 2
    create.assert_not_called()


def test_cli_rejects_custom_judge_prompt_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        data=json.dumps(
            obj={
                "judge": {
                    "provider": "openai",
                    "model": "judge",
                    "prompt": {"version": 2, "text": "Custom instructions"},
                },
            }
        ),
        encoding="utf-8",
    )
    create = MagicMock()
    monkeypatch.setattr(target=ModelConfig, name="create_model", value=create)
    assert cli.main(argv=["--config", str(object=config)]) == 2
    create.assert_not_called()


def test_every_semantic_score_and_tool_requirement_must_pass(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(
        path,
        [
            entry(
                ground_truth={"ideal_answer": "A", "answer_criteria": ["Concise"]},
            )
        ],
    )
    judge = judge_with(4)
    grade = SemanticGrade(
        (
            ExpectationScore(
                expectation_id="ideal_answer", score=5, explanation="Agrees"
            ),
            ExpectationScore(
                expectation_id="answer_criteria[0]", score=3, explanation="Long"
            ),
        ),
        AIMessage(content="raw"),
    )
    judge.grade = MagicMock(return_value=grade)  # type: ignore[method-assign]
    run = run_evaluation(
        ScriptedChatModel(responses=[AIMessage(content="A")]),
        vault_path=tmp_path,
        judge=judge,
        dataset_path=path,
        report_destination=tmp_path,
    )
    assert run.report["results"][0]["status"] == "failed"

    write_dataset(
        path,
        [
            entry(
                tool_checks={
                    "required_tools": ["list_notes"],
                    "forbidden_tools": ["read_note"],
                }
            )
        ],
    )
    candidate = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "list_notes", "args": {"path": None}, "id": "a"},
                    {
                        "name": "read_note",
                        "args": {"notepath": "missing.md"},
                        "id": "b",
                    },
                ],
            ),
            AIMessage(content="A"),
        ]
    )
    run = run_evaluation(
        candidate,
        vault_path=tmp_path,
        judge=judge_with(5),
        dataset_path=path,
        report_destination=tmp_path,
    )
    result = run.report["results"][0]
    assert result["status"] == "failed"
    assert [check["passed"] for check in result["tool_checks"]] == [True, False]


def test_persistence_failure_leaves_previous_incomplete_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(path, [entry()])
    original = runner.save_report
    writes = 0

    def failing_save(directory: Path, report: dict[str, Any]) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("disk full")
        original(directory, report)

    monkeypatch.setattr(runner, "save_report", failing_save)
    with pytest.raises(OSError, match="disk full"):
        run_evaluation(
            ScriptedChatModel(responses=[AIMessage(content="A")]),
            vault_path=tmp_path,
            judge=judge_with(5),
            dataset_path=path,
            report_destination=tmp_path / "reports",
        )
    report = json.loads(next((tmp_path / "reports").glob("*/report.json")).read_text())
    assert not report["complete"] and report["summary"]["pending"] == 1


def test_cli_persistence_error_returns_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(path, [entry()])
    monkeypatch.setattr(
        target=ModelConfig,
        name="create_model",
        value=lambda self: MagicMock(),
    )
    monkeypatch.setattr(
        cli, "run_evaluation", MagicMock(side_effect=OSError("disk full"))
    )
    assert cli.main(["--dataset-path", str(path), "--vault-path", str(tmp_path)]) == 2


def test_suite_and_repeat_costs_include_judge_errors_and_wall_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(path=path, entries=[entry()])
    candidate = ScriptedChatModel(
        responses=[
            AIMessage(
                content="A",
                usage_metadata={
                    "input_tokens": 80,
                    "output_tokens": 20,
                    "total_tokens": 100,
                },
            ),
            AIMessage(
                content="B",
                usage_metadata={
                    "input_tokens": 150,
                    "output_tokens": 50,
                    "total_tokens": 200,
                },
            ),
        ]
    )
    model = MagicMock()
    model.model_dump.return_value = {"model": "judge"}
    model.with_structured_output.return_value.invoke.side_effect = [
        {
            "raw": AIMessage(
                content="grade",
                usage_metadata={
                    "input_tokens": tokens - 5,
                    "output_tokens": 5,
                    "total_tokens": tokens,
                },
            ),
            "parsed": {
                "scores": [
                    {
                        "expectation_id": "ideal_answer",
                        "score": score,
                        "explanation": "Meaning checked",
                    }
                ]
            },
            "parsing_error": None,
        }
        for tokens, score in [(25, 5), (35, "invalid")]
    ]
    monkeypatch.setattr(
        target=runner,
        name="perf_counter",
        value=MagicMock(
            side_effect=[0.0, 1.0, 5.0, 6.0, 7.0, 14.0, 15.0],
        ),
    )
    run = run_evaluation(
        candidate_model=candidate,
        vault_path=tmp_path,
        judge=SemanticJudge(model=model),
        dataset_path=path,
        repeats=2,
        report_destination=tmp_path / "reports",
    )
    report = run.report
    assert [item["status"] for item in report["results"]] == ["passed", "judge_error"]
    assert [item["elapsed_seconds"] for item in report["results"]] == [4.0, 7.0]
    assert [item["total_tokens"] for item in report["results"]] == [125, 235]
    assert report["summary"]["candidate_tokens"] == 300
    assert report["summary"]["judge_tokens"] == 60
    assert report["summary"]["total_tokens"] == 360
    assert report["elapsed_seconds"] == 15.0
    summary = (run.run_directory / "summary.md").read_text(encoding="utf-8")
    assert "Total cost: 360 tokens (candidate: 300 tokens; judge: 60 tokens)" in summary
    assert "Total time: 15.000 s" in summary
    assert "Cost: 125 tokens (candidate: 100 tokens; judge: 25 tokens)" in summary
    assert "Cost: 235 tokens (candidate: 200 tokens; judge: 35 tokens)" in summary
    assert "Time: 4.000 s" in summary and "Time: 7.000 s" in summary
    stored = json.loads(
        s=(run.run_directory / "report.json").read_text(encoding="utf-8")
    )
    assert stored["summary"]["total_tokens"] == 360
    assert stored["elapsed_seconds"] == 15.0
