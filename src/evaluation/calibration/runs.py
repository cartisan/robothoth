"""Checkpoint isolated judge calls and correlate saved scores with current ratings."""

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator
from scipy.stats import pearsonr, spearmanr  # type: ignore[import-untyped]

from src.evaluation.calibration.dataset import (
    CalibrationDataset,
    ErrorEvidence,
    Nonblank,
    StrictRecord,
)
from src.evaluation.judge import (
    STAGE_PROMPTS,
    GradingPayload,
    JudgeError,
    JudgeResponseSchema,
    JudgeStage,
    RubricScore,
    SemanticJudge,
)
from src.evaluation.reporting import (
    _atomic_write,
    json_value,
    model_configuration,
    without_credentials,
)
from src.gold_dataset.validate import _unique_object

RUNS_DIRECTORY = Path(__file__).with_name("runs")
RATING_FIELDS: dict[JudgeStage, str] = {
    "content": "content",
    "concision": "concision",
    "task_usefulness": "usefulness",
}


class CandidateInput(StrictRecord):
    """Match candidate identity and exact answer independently of human ratings."""

    candidate_id: Nonblank
    answer: Nonblank | None


class QuestionInput(StrictRecord):
    """Snapshot the input fields that determine rubric grades."""

    case_id: Nonblank
    request: Nonblank
    ideal_answer: Nonblank
    candidates: list[CandidateInput]


class JudgeResult(StrictRecord):
    """Retain one rubric call's score or failure, with raw call evidence."""

    stage: JudgeStage
    case_id: Nonblank
    candidate_id: Nonblank
    score: RubricScore | None = None
    error: ErrorEvidence | None = None
    calls: list[dict[str, Any]]

    @model_validator(mode="after")
    def validate_outcome(self) -> "JudgeResult":
        """Require one outcome and the stage's production rubric."""
        if (self.score is None) == (self.error is None):
            raise ValueError("result requires exactly one score or error")
        if self.score is not None and self.score.rubric != self.stage:
            raise ValueError("score rubric does not match rubric")
        return self


class SplitContext(StrictRecord):
    """Identify a split role and its full source inputs, excluding human ratings."""

    role: Literal["test", "validation"]
    seed: int
    source_input_fingerprint: Nonblank  # Binds the subset to the full source.


class JudgeRun(StrictRecord):
    """Store one prompt/configuration snapshot and ordered checkpointed results."""

    schema_version: Annotated[int, Field(ge=3, le=3)] = 3
    split_context: SplitContext | None = None
    run_id: Nonblank
    dataset_id: Nonblank
    created_at: Nonblank
    prompts: dict[JudgeStage, Nonblank]
    configuration: dict[str, Any]
    output_schema: dict[str, Any]  # Production schema, once per prompt snapshot.
    inputs: list[QuestionInput]
    input_fingerprint: Nonblank
    results: list[JudgeResult] = Field(default_factory=list)
    complete: bool = False

    @model_validator(mode="after")
    def validate_snapshot(self) -> "JudgeRun":
        """Reject inconsistent prompts, fingerprints or result references/order."""
        if set(self.prompts) != set(STAGE_PROMPTS):
            raise ValueError("prompts must cover exactly the three rubrics")
        if _fingerprint(inputs=self.inputs) != self.input_fingerprint:
            raise ValueError("input fingerprint does not match snapshot")
        case_ids = [question.case_id for question in self.inputs]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("duplicate input case_id")
        for question in self.inputs:
            ids = [candidate.candidate_id for candidate in question.candidates]
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate input candidate_id")
        expected = [
            (stage, q.case_id, c.candidate_id)
            for stage in STAGE_PROMPTS
            for q in self.inputs
            for c in q.candidates
            if c.answer is not None
        ]
        actual = [
            (item.stage, item.case_id, item.candidate_id) for item in self.results
        ]
        if actual != expected[: len(actual)] or (self.complete and actual != expected):
            raise ValueError("results do not match scheduled inputs")
        return self


def snapshot_inputs(dataset: CalibrationDataset) -> list[QuestionInput]:
    """Return ordered grading inputs excluding human ratings and model settings."""
    return [
        QuestionInput(
            case_id=q.case_id,
            request=q.request,
            ideal_answer=q.ideal_answer,
            candidates=[
                CandidateInput(candidate_id=c.candidate_id, answer=c.answer)
                for c in q.candidates
            ],
        )
        for q in dataset.questions
    ]


def _fingerprint(inputs: list[QuestionInput]) -> str:
    """Hash canonical JSON while preserving question and candidate order."""
    content = json.dumps(
        obj=[item.model_dump() for item in inputs],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(string=content.encode(encoding="utf-8")).hexdigest()


def input_fingerprint(dataset: CalibrationDataset) -> str:
    """Identify exact requests, ideal answers and candidate answers, without ratings."""
    return _fingerprint(inputs=snapshot_inputs(dataset=dataset))


def save_run(run: JudgeRun, path: Path) -> None:
    """Atomically checkpoint a run; persistence failures propagate as OSError."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path=path, content=run.model_dump_json(indent=2) + "\n")


def load_run(
    path: str | Path,
    dataset: CalibrationDataset,
    split_context: SplitContext | None = None,
) -> JudgeRun:
    """Load a run only when its input snapshot matches the current dataset.

    Rating changes are allowed. Question/candidate order, IDs, requests, ideal
    answers and candidate answers must match exactly, including failed candidates.
    Partial checkpoint files are accepted and retain complete=False. Split runs
    additionally require matching role, seed and full source-input fingerprint.
    Only schema version 3 is accepted; older runs are rejected without migration.

    Raises:
        ValueError: If JSON, schema, snapshot or input matching is invalid.
        OSError: If the source cannot be read.
    """
    run = JudgeRun.model_validate(
        obj=json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    )
    if run.split_context != split_context:
        raise ValueError("saved split context does not match this dataset")
    if run.input_fingerprint != input_fingerprint(dataset=dataset):
        raise ValueError("saved judge inputs do not match this dataset")
    return run


def run_judges(
    dataset: CalibrationDataset,
    prompts: dict[JudgeStage, str],
    judge: SemanticJudge,
    runs_directory: Path = RUNS_DIRECTORY,
    progress: Callable[[int, int], None] | None = None,
    split_context: SplitContext | None = None,
) -> JudgeRun:
    """Judge in rubric → question → candidate order, checkpointing every call.

    Prompt text and sanitized model/binding settings are snapshotted once. Each
    call delegates to the shared production judge. Failed generations are skipped;
    individual judging errors retain evidence and allow later calls to continue.
    Optional split context binds this subset to its role and full source inputs.
    The checkpoint is written before progress is reported, including the last call.

    Raises:
        ValueError: If the prompt snapshot or dataset inputs are invalid.
        OSError: If initial persistence or any checkpoint fails.
        Exception: If the progress callback fails; earlier calls remain on disk.
    """
    inputs = snapshot_inputs(dataset=dataset)
    run = JudgeRun(
        split_context=split_context,
        run_id=uuid4().hex,
        dataset_id=dataset.dataset_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        prompts=dict(prompts),
        configuration={
            "model": model_configuration(model=judge.model),
            "structured_output": without_credentials(
                value=judge.structured_output_kwargs
            ),
        },
        output_schema=JudgeResponseSchema.model_json_schema(),
        inputs=inputs,
        input_fingerprint=_fingerprint(inputs=inputs),
    )
    path = runs_directory / f"{run.run_id}.json"
    save_run(run=run, path=path)
    total = 3 * sum(c.answer is not None for q in inputs for c in q.candidates)

    # for each stage, run over the questions and the candidates to make use of caching
    for stage in STAGE_PROMPTS:
        for question in inputs:
            for candidate in question.candidates:
                if candidate.answer is None:
                    continue
                values: dict[str, Any] = dict(
                    stage=stage,
                    case_id=question.case_id,
                    candidate_id=candidate.candidate_id,
                )
                try:
                    grade = judge.grade_stage(
                        payload=GradingPayload(
                            rubric=stage,
                            request=question.request,
                            final_answer=candidate.answer,
                            ideal_answer=question.ideal_answer
                            if stage == "content"
                            else None,
                        ),
                        prompt=run.prompts[stage],
                    )
                    result = JudgeResult(
                        **values,
                        score=grade.scores[0],
                        calls=json_value(value=grade.judge_calls),
                    )
                except Exception as exc:
                    result = JudgeResult(
                        **values,
                        error=ErrorEvidence(type=type(exc).__name__, message=str(exc)),
                        calls=json_value(value=exc.judge_calls)
                        if isinstance(exc, JudgeError)
                        else [],
                    )
                run.results.append(result)
                run.complete = len(run.results) == total
                save_run(run=run, path=path)
                if progress is not None:
                    progress(len(run.results), total)
    if total == 0:
        run.complete = True
        save_run(run=run, path=path)
    return run


class RatingStatistics(StrictRecord):
    """Compare paired human/judge ratings; undefined metrics are None."""

    pair_count: int
    spearman_rho: float | None = None
    pearson_r: float | None = None
    linear_weighted_kappa: float | None = None
    quadratic_weighted_kappa: float | None = None
    mean_absolute_error: float | None = None
    mean_signed_error: float | None = None  # Judge minus human, in rating points.


def _weighted_kappa(pairs: Sequence[tuple[int, int]], power: int) -> float | None:
    """Return chance-corrected agreement using actual 1–5 rating distances.

    Power 1 gives linear weights and 2 gives quadratic weights. Expected
    disagreement uses independent empirical human and judge marginals. With
    zero expected disagreement (identical constant ratings), kappa is undefined.
    Callers supply nonempty pairs of validated ratings.
    """
    human_counts = Counter(h for h, _ in pairs)
    judge_counts = Counter(j for _, j in pairs)
    observed = sum(abs(h - j) ** power for h, j in pairs)
    # Normalizing distances by 4 (or 16) cancels in the disagreement ratio.
    expected = sum(
        abs(h - j) ** power * hc * jc
        for h, hc in human_counts.items()
        for j, jc in judge_counts.items()
    )
    return None if expected == 0 else 1.0 - observed * len(pairs) / expected


def rating_statistics(
    human: Sequence[int | None],
    judge: Sequence[int | None],
) -> RatingStatistics:
    """Compare validated 1–5 ratings, omitting pairs with either rating missing.

    Spearman rho handles tied ranks; Pearson r measures linear correlation.
    Both are unavailable with fewer than two pairs or either side constant.
    Weighted kappas use linear/quadratic distances on the full 1–5 scale and
    are unavailable with no pairs or zero expected disagreement. Absolute and
    signed errors are in rating points, with signed error defined as judge minus
    human; both are available with one pair. Missing metrics return None.

    Raises:
        ValueError: If the input sequences have different lengths.
    """
    pairs = [
        (h, j)
        for h, j in zip(human, judge, strict=True)
        if h is not None and j is not None
    ]
    count = len(pairs)
    statistics = RatingStatistics(pair_count=count)
    if not pairs:
        return statistics
    statistics.mean_absolute_error = sum(abs(j - h) for h, j in pairs) / count
    statistics.mean_signed_error = sum(j - h for h, j in pairs) / count
    statistics.linear_weighted_kappa = _weighted_kappa(pairs=pairs, power=1)
    statistics.quadratic_weighted_kappa = _weighted_kappa(pairs=pairs, power=2)
    hs, js = zip(*pairs, strict=True)
    if count >= 2 and len(set(hs)) > 1 and len(set(js)) > 1:
        statistics.spearman_rho = float(spearmanr(a=hs, b=js).statistic)
        statistics.pearson_r = float(pearsonr(x=hs, y=js).statistic)
    return statistics


def compute_statistics(
    dataset: CalibrationDataset,
    run: JudgeRun,
) -> dict[JudgeStage, RatingStatistics]:
    """Compare current human ratings with successful saved scores for each rubric.

    Raises:
        ValueError: If current grading inputs differ from the run snapshot.
    """
    if input_fingerprint(dataset=dataset) != run.input_fingerprint:
        raise ValueError("judge inputs do not match this dataset")
    candidates = {
        (q.case_id, c.candidate_id): c for q in dataset.questions for c in q.candidates
    }
    statistics = {}
    for stage, field in RATING_FIELDS.items():
        results = [item for item in run.results if item.stage == stage]
        statistics[stage] = rating_statistics(
            human=[
                getattr(
                    candidates[item.case_id, item.candidate_id].human_ratings, field
                )
                for item in results
            ],
            judge=[
                item.score.score if item.score is not None else None for item in results
            ],
        )
    return statistics
