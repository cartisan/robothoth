"""Run the entire validated gold dataset sequentially and retain every repeat."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel

from src.evaluation.judge import (
    CONCISION_PROMPT,
    CONTENT_PROMPT,
    TASK_USEFULNESS_PROMPT,
    JudgeError,
    SemanticJudge,
)
from src.evaluation.reporting import (
    json_value,
    model_configuration,
    repeat_token_usage,
    save_report,
    without_credentials,
)
from src.evaluation.tool_checks import evaluate_tool_checks
from src.gold_dataset.validate import (
    DEFAULT_DATASET_PATH,
    GoldDataset,
    load_dataset_document,
)
from src.harness import (
    SYSTEM_PROMPT,
    HarnessError,
    run_in_harness,
    validate_harness_options,
)
from src.utils import SystemPrompt

DEFAULT_VAULT_PATH = Path(__file__).resolve().parents[2] / "tests/extended_test_vault"


def _now() -> str:
    """Return a UTC ISO timestamp for persisted run and repeat boundaries."""
    return datetime.now(timezone.utc).isoformat()


def validate_evaluation_inputs(
    dataset_path: str | Path | None,
    vault_path: str | Path,
    repeats: int,
    max_api_calls: int,
    model_label: str | None,
) -> GoldDataset:
    """Return the full validated dataset when every case can be scheduled.

    The entire dataset is validated, and empty datasets are rejected. No model
    binding or invocation occurs. The CLI uses the same validation before model
    construction, so invalid inputs require no credentials.

    Raises:
        ValueError: If the dataset is empty, the vault is not a directory, or
            repeats and harness options are invalid.
        DatasetValidationError: If any dataset entry is invalid.
        OSError: If the dataset cannot be read.
    """
    if repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    validate_harness_options(
        user_prompt="evaluation request",
        vault_path=str(object=vault_path),
        max_api_calls=max_api_calls,
        model_label=model_label,
    )
    if not Path(vault_path).is_dir():
        raise ValueError("vault_path must be an existing directory")
    dataset = load_dataset_document(dataset_path)
    if not dataset.entries:
        raise ValueError("dataset must contain at least one entry")
    return dataset


@dataclass(frozen=True)
class EvaluationRun:
    """Expose the persisted run directory, report, and corresponding CLI status."""

    run_directory: Path
    report: dict[str, Any]

    @property
    def exit_code(self) -> int:
        """Return 0 for all passes, 1 for scored failures, or 2 for incomplete runs."""
        if not self.report["complete"]:
            return 2
        return 1 if self.report["summary"]["failed"] else 0


def _error_record(exc: Exception) -> dict[str, Any]:
    """Retain error type and explanation, including any original chained failure."""
    cause = exc.__cause__
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "cause": {"type": type(cause).__name__, "message": str(cause)}
        if cause is not None
        else None,
    }


def run_evaluation(
    candidate_model: BaseChatModel,
    system_prompt: SystemPrompt = SYSTEM_PROMPT,
    vault_path: str | Path = DEFAULT_VAULT_PATH,
    judge: SemanticJudge | None = None,
    dataset_path: str | Path | None = None,
    repeats: int = 1,
    max_api_calls: int = 20,
    model_label: str | None = None,
    tool_binding_kwargs: dict[str, Any] | None = None,
    report_destination: str | Path = "evaluation_runs",
    candidate_config: dict[str, Any] | None = None,
    judge_config: dict[str, Any] | None = None,
) -> EvaluationRun:
    """Evaluate every gold case in order, preserving independent repeat outcomes.

    Each repeat gets a fresh harness conversation and semantic grade. Valid
    answers are graded even when tool checks fail. A repeat passes only when all
    content and quality scores are at least 4 and every tool requirement passes.
    Harness and judge errors continue to later repeats and make the report incomplete;
    scored failures do not. Pass rates use only passed and failed repeats.
    Costs sum candidate and judge tokens, retaining unknown usage. Wall-clock
    durations include tool execution, grading, and suite overhead between repeats.

    A unique directory under report_destination receives an initial incomplete
    report and atomic JSON/Markdown updates after each repeat. Dataset contents
    and hashes are not copied; case IDs, entry versions, and the highest entry
    version identify the gold dataset. Model credentials are excluded from
    configurations. No aggregate case verdict is assigned.

    Raises:
        ValueError: If batch setup or harness arguments are invalid.
        DatasetValidationError: If the entire dataset fails validation.
        OSError: If the dataset or report files cannot be read or written.
        Exception: If default judge configuration fails, or an unexpected error
            escapes the harness or judge interface.
    """
    suite_started = perf_counter()
    started = _now()
    dataset = validate_evaluation_inputs(
        dataset_path=dataset_path,
        vault_path=vault_path,
        repeats=repeats,
        max_api_calls=max_api_calls,
        model_label=model_label,
    )
    evaluator = judge if judge is not None else SemanticJudge()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex
    directory = Path(report_destination).resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "report_version": 3,
        "run_id": run_id,
        "started_at": started,
        "updated_at": started,
        "finished_at": None,
        "elapsed_seconds": 0.0,
        "complete": False,
        "dataset_path": str(Path(dataset_path or DEFAULT_DATASET_PATH).resolve()),
        "schema_version": dataset.schema_version,
        "gold_dataset_version": max(item.dataset_version for item in dataset.entries),
        "case_ids": [item.id for item in dataset.entries],
        "repeats": repeats,
        "expected_repeats": len(dataset.entries) * repeats,
        "candidate_system_prompt": system_prompt.model_dump(),
        "judge_prompt": CONTENT_PROMPT.model_dump(),
        "quality_judge_prompts": {
            "concision": CONCISION_PROMPT.model_dump(),
            "task_usefulness": TASK_USEFULNESS_PROMPT.model_dump(),
        },
        "candidate_config": without_credentials(
            candidate_config
            if candidate_config is not None
            else model_configuration(candidate_model)
        ),
        "judge_config": without_credentials(
            judge_config
            if judge_config is not None
            else model_configuration(evaluator.model)
        ),
        "judge_structured_output_kwargs": without_credentials(
            evaluator.structured_output_kwargs
        ),
        "harness_settings": {
            "vault_path": str(Path(vault_path).resolve()),
            "max_api_calls": max_api_calls,
            "model_label": model_label,
            "tool_binding_kwargs": without_credentials(tool_binding_kwargs or {}),
        },
        "results": [],
    }
    save_report(directory, report)
    for case in dataset.entries:
        for repeat in range(1, repeats + 1):
            repeat_started = perf_counter()
            result: dict[str, Any] = {
                "case_id": case.id,
                "dataset_version": case.dataset_version,
                "repeat": repeat,
                "started_at": _now(),
                "finished_at": None,
                "status": "harness_error",
                "final_answer": None,
                "trace": None,
                "tool_checks": [],
                "scores": [],
                "judge_calls": [],
                "error": None,
            }
            try:
                harness_result = run_in_harness(
                    chat_model=candidate_model,
                    user_prompt=case.request,
                    vault_path=str(object=vault_path),
                    system_prompt=system_prompt,
                    max_api_calls=max_api_calls,
                    model_label=model_label,
                    tool_binding_kwargs=tool_binding_kwargs,
                )
            except HarnessError as exc:
                result["error"] = _error_record(exc=exc)
                result["trace"] = json_value(value=exc.trace)
                result["tool_checks"] = json_value(
                    value=evaluate_tool_checks(
                        tool_checks=case.tool_checks, trace=exc.trace
                    )
                )
            else:
                result["final_answer"] = harness_result.final_answer
                result["trace"] = json_value(harness_result.trace)
                checks = evaluate_tool_checks(case.tool_checks, harness_result.trace)
                result["tool_checks"] = json_value(checks)
                try:
                    grade = evaluator.grade(
                        request=case.request,
                        final_answer=harness_result.final_answer,
                        ground_truth=case.ground_truth,
                    )
                except JudgeError as exc:
                    result["status"] = "judge_error"
                    result["error"] = _error_record(exc=exc)
                    result["judge_calls"] = json_value(value=exc.judge_calls)
                else:
                    result["scores"] = json_value(grade.scores)
                    result["judge_calls"] = json_value(value=grade.judge_calls)
                    passed = all(item.passed for item in checks) and all(
                        item.score >= 4 for item in grade.scores
                    )
                    result["status"] = "passed" if passed else "failed"
            result["finished_at"] = _now()
            result["elapsed_seconds"] = perf_counter() - repeat_started
            result.update(**repeat_token_usage(result=result))
            report["results"].append(result)
            report["updated_at"] = result["finished_at"]
            report["elapsed_seconds"] = perf_counter() - suite_started
            if len(report["results"]) == report["expected_repeats"]:
                report["finished_at"] = result["finished_at"]
            save_report(directory, report)
    return EvaluationRun(directory, report)
