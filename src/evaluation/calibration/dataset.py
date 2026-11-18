"""Validate and atomically persist calibration answers and human ratings."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.evaluation.config import CandidateConfig
from src.evaluation.reporting import _atomic_write
from src.gold_dataset.validate import _unique_object
from src.utils import SystemPrompt

SPLIT_SEED = 42

Nonblank = Annotated[str, StringConstraints(pattern=r"\S")]
Rating = Annotated[int, Field(ge=1, le=5)]


class StrictRecord(BaseModel):
    """Reject unknown fields and coercion at persisted data boundaries."""

    model_config = ConfigDict(extra="forbid", strict=True)


class HumanRatings(StrictRecord):
    """Retain independently nullable rubric ratings; None means not rated."""

    content: Rating | None = None
    concision: Rating | None = None
    usefulness: Rating | None = None


class ErrorEvidence(StrictRecord):
    """Retain operational failure details and optional trace or call evidence."""

    type: Nonblank
    message: str
    evidence: Any = None


class Candidate(StrictRecord):
    """Identify an answer or generation failure and its current human ratings."""

    candidate_id: Nonblank
    model_id: Nonblank  # Resolves to generation.models, never a repeated config.
    answer: Nonblank | None = None
    generation_error: ErrorEvidence | None = None
    human_ratings: HumanRatings = Field(default_factory=HumanRatings)

    @model_validator(mode="after")
    def require_outcome(self) -> "Candidate":
        """Require exactly one of a usable answer or a generation error."""
        if (self.answer is None) == (self.generation_error is None):
            raise ValueError(
                "candidate requires exactly one answer or generation_error"
            )
        return self


class Question(StrictRecord):
    """Retain gold identity, request and reference with ordered candidates."""

    case_id: Nonblank
    request: Nonblank
    ideal_answer: Nonblank
    candidates: list[Candidate] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_candidates(self) -> "Question":
        """Reject ambiguous candidate identities within this question."""
        ids = [item.candidate_id for item in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate candidate_id")
        return self


class GenerationRun(StrictRecord):
    """Store source paths and candidate configuration once for the whole run."""

    gold_schema_version: int
    gold_dataset_path: Nonblank
    vault_path: Nonblank
    system_prompt: SystemPrompt
    max_api_calls: int = Field(gt=0)
    models: dict[Nonblank, CandidateConfig] = Field(min_length=1)


class CalibrationDataset(StrictRecord):
    """Versioned ordered questions belonging to a fresh generation run."""

    schema_version: Annotated[int, Field(ge=1, le=1)] = 1
    dataset_id: Nonblank
    created_at: Nonblank
    generation: GenerationRun
    questions: list[Question] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> "CalibrationDataset":
        """Require unique cases and resolve every candidate's model reference."""
        ids = [question.case_id for question in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate case_id")
        for question in self.questions:
            for candidate in question.candidates:
                if candidate.model_id not in self.generation.models:
                    raise ValueError(f"unknown model_id: {candidate.model_id}")
        return self

    def create_test_val_split(
        self,
    ) -> tuple["CalibrationDataset", "CalibrationDataset"]:
        """Return test and validation views with equal whole-question model counts.

        Sorted case IDs are shuffled with local seed 42; the first half belongs to
        test. Views retain source question/candidate order and reference current
        records, including failed generations and human ratings. Never save a view
        over the full source dataset.

        Raises:
            ValueError: If there are fewer than two questions, the count is odd, or
                per-question model composition (including multiplicities) differs.
        """
        count = len(self.questions)
        if count < 2 or count % 2:
            raise ValueError(
                "calibration requires an even question count of at least two"
            )
        composition = Counter(c.model_id for c in self.questions[0].candidates)
        if any(
            Counter(c.model_id for c in q.candidates) != composition
            for q in self.questions
        ):
            raise ValueError(
                "calibration requires identical per-question model composition"
            )

        ids = sorted(q.case_id for q in self.questions)
        random.Random(x=SPLIT_SEED).shuffle(x=ids)
        test_ids = set(ids[: count // 2])
        return (
            self.model_copy(
                update={
                    "questions": [q for q in self.questions if q.case_id in test_ids]
                }
            ),
            self.model_copy(
                update={
                    "questions": [
                        q for q in self.questions if q.case_id not in test_ids
                    ]
                }
            ),
        )


def load_dataset(path: str | Path) -> CalibrationDataset:
    """Load strict versioned JSON, rejecting duplicate object keys.

    Raises:
        ValueError: If JSON, schema version, fields or references are invalid.
        OSError: If the source cannot be read.
    """
    return CalibrationDataset.model_validate(
        obj=json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    )


def save_dataset(dataset: CalibrationDataset, path: str | Path) -> None:
    """Atomically save the complete current dataset, including unset ratings.

    Raises:
        OSError: If parent directory creation or atomic replacement fails.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path=destination, content=dataset.model_dump_json(indent=2) + "\n")
