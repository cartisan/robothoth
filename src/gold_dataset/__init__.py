"""Validated, manually maintained gold cases for future evaluation work."""

import json
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from src.utils import normalize_note_path

DEFAULT_DATASET_PATH = Path(__file__).with_name("dataset.json")


class DatasetValidationError(ValueError):
    """Report invalid dataset content with its source path and field locations."""


def _nonblank(value: str) -> str:
    """Return text containing a non-whitespace character, or reject it."""
    if not value.strip():
        raise ValueError("must not be blank")
    return value


class GroundTruth(BaseModel):
    """Reference answer and/or required criteria for evaluating a response."""

    model_config = ConfigDict(extra="forbid", strict=True)

    ideal_answer: StrictStr | None = None
    answer_criteria: list[StrictStr] | None = None

    @field_validator("ideal_answer")
    @classmethod
    def validate_ideal_answer(cls, value: str | None) -> str | None:
        """Accept an absent answer or a nonblank reference answer."""
        return None if value is None else _nonblank(value)

    @field_validator("answer_criteria")
    @classmethod
    def validate_criteria(cls, value: list[str] | None) -> list[str] | None:
        """Accept absent criteria or nonblank criteria, including an empty list."""
        return None if value is None else [_nonblank(item) for item in value]

    @model_validator(mode="after")
    def require_expectation(self) -> "GroundTruth":
        """Require a usable answer or at least one criterion."""
        if self.ideal_answer is None and not self.answer_criteria:
            raise ValueError("ideal_answer or nonempty answer_criteria is required")
        return self


class ToolChecks(BaseModel):
    """Require successful calls or prohibit attempted tools, regardless of order."""

    model_config = ConfigDict(extra="forbid", strict=True)

    required_tools: list[StrictStr] = Field(default_factory=list)
    required_note_reads: list[StrictStr] = Field(default_factory=list)
    forbidden_tools: list[StrictStr] = Field(default_factory=list)
    allowed_tools: list[StrictStr] | None = None

    @field_validator("required_tools", "forbidden_tools", "allowed_tools")
    @classmethod
    def validate_tools(cls, value: list[str] | None) -> list[str] | None:
        """Accept absent allowlists or unique nonblank tool names."""
        if value is not None:
            for item in value:
                _nonblank(item)
            if len(set(value)) != len(value):
                raise ValueError("tool names must be unique")
        return value

    @field_validator("required_note_reads")
    @classmethod
    def validate_paths(cls, value: list[str]) -> list[str]:
        """Accept unique full relative Markdown paths after lexical normalization."""
        normalized = [normalize_note_path(item) for item in value]
        if any(not item.endswith(".md") for item in normalized):
            raise ValueError("required note reads must be Markdown file paths")
        if len(set(normalized)) != len(normalized):
            raise ValueError("required note paths must be unique")
        return normalized

    @model_validator(mode="after")
    def reject_contradictions(self) -> "ToolChecks":
        """Require demanded tools, including note reads, to be permitted."""
        required = set(self.required_tools)
        if self.required_note_reads:
            required.add("read_note")
        if required.intersection(self.forbidden_tools):
            raise ValueError("required tools must not be forbidden")
        if self.allowed_tools is not None and not required.issubset(self.allowed_tools):
            raise ValueError("required tools must be in allowed_tools")
        return self


class GoldEntry(BaseModel):
    """One versioned evaluation request with its reason and ground truth."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: StrictStr
    request: StrictStr
    added_reason: StrictStr
    dataset_version: StrictInt = Field(gt=0)
    ground_truth: GroundTruth
    tool_checks: ToolChecks

    @field_validator("id", "request", "added_reason")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Accept only text containing a non-whitespace character."""
        return _nonblank(value)


class GoldDataset(BaseModel):
    """Collection with unique entry IDs and an unrestricted integer schema version.

    Every version uses the same entry validation rules. The supplied version is
    retained as metadata without selecting or filtering entries.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: StrictInt
    entries: list[GoldEntry]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "GoldDataset":
        """Reject a repeated ID and identify its later entry position."""
        seen: set[str] = set()
        for index, item in enumerate(self.entries):
            if item.id in seen:
                raise ValueError(f"duplicate ID at entries[{index}].id: {item.id!r}")
            seen.add(item.id)
        return self


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Return a JSON object only when each key occurs once."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _format_validation_error(error: ValidationError) -> str:
    """Return readable model errors with each available field location."""
    lines = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"]) or "dataset"
        lines.append(f"{location}: {detail['msg']}")
    return "; ".join(lines)


def load_dataset_document(path: str | Path | None = None) -> GoldDataset:
    """Return the entire validated dataset with its schema metadata.

    The bundled path is independent of the current directory. Duplicate JSON
    keys, unknown fields, and coercion are rejected. Any integer schema version
    is accepted and preserved; entries use the current model's validation rules.

    Raises:
        DatasetValidationError: If JSON content or the dataset schema is invalid.
        OSError: If the file cannot be read.
    """
    source = Path(path) if path is not None else DEFAULT_DATASET_PATH
    try:
        content = source.read_text(encoding="utf-8")
        raw = json.loads(content, object_pairs_hook=_unique_object)
        dataset = GoldDataset.model_validate(raw)
    except (UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        detail = (
            _format_validation_error(exc)
            if isinstance(exc, ValidationError)
            else str(exc)
        )
        raise DatasetValidationError(f"{source}: {detail}") from exc
    return dataset


def load_dataset(
    path: str | Path | None = None, version: int | None = None
) -> tuple[GoldEntry, ...]:
    """Return validated entries in file order, optionally through a version.

    The entire file is validated before filtering. Version 0 selects no entries;
    later versions include all earlier cases, using their current expectations.

    Raises:
        ValueError: If version is not a nonnegative integer.
        DatasetValidationError: If JSON content or the dataset schema is invalid.
        OSError: If the file cannot be read.
    """
    if version is not None and (type(version) is not int or version < 0):
        raise ValueError("version must be a nonnegative integer")
    dataset = load_dataset_document(path)
    return tuple(
        item
        for item in dataset.entries
        if version is None or item.dataset_version <= version
    )
