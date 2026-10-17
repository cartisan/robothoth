"""Behavioral tests for the manually maintained gold dataset."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.gold_dataset.validate import (
    DatasetValidationError,
    GoldDataset,
    GoldEntry,
    GroundTruth,
    load_dataset,
    load_dataset_document,
)


def entry(**changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "id": "case-one",
        "request": "Find the Golden Evaluation Dataset note",
        "added_reason": "Check note discovery",
        "dataset_version": 1,
        "ground_truth": {"ideal_answer": "The note is in evaluation."},
        "tool_checks": {},
    }
    result.update(changes)
    return result


def write_dataset(
    path: Path,
    entries: list[dict[str, object]],
    schema_version: int = 1,
) -> None:
    path.write_text(
        data=json.dumps(obj={"schema_version": schema_version, "entries": entries}),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "ground_truth",
    [
        {"ideal_answer": "An answer"},
        {"answer_criteria": ["Names the note"]},
        {"ideal_answer": "An answer", "answer_criteria": ["Names the note"]},
        {"ideal_answer": "An answer", "answer_criteria": []},
        {"ideal_answer": None, "answer_criteria": ["Names the note"]},
        {"ideal_answer": "An answer", "answer_criteria": None},
    ],
)
def test_ground_truth_accepts_usable_combinations(
    ground_truth: dict[str, object],
) -> None:
    assert GoldEntry.model_validate(entry(ground_truth=ground_truth)).ground_truth


@pytest.mark.parametrize(
    "ground_truth",
    [
        {},
        {"ideal_answer": None},
        {"answer_criteria": []},
        {"ideal_answer": None, "answer_criteria": None},
    ],
)
def test_ground_truth_requires_an_expectation(ground_truth: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        GroundTruth.model_validate(ground_truth)


@pytest.mark.parametrize(
    "ground_truth",
    [
        {"ideal_answer": "  "},
        {"answer_criteria": ["  "]},
        {"ideal_answer": "Valid", "answer_criteria": ["  "]},
        {"ideal_answer": "Valid", "answer_criteria": [], "extra": "no"},
        {"ideal_answer": 2},
        {"answer_criteria": "criterion"},
        {"answer_criteria": ("criterion",)},
        {"answer_criteria": [1]},
    ],
)
def test_ground_truth_rejects_invalid_supplied_fields(
    ground_truth: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        GroundTruth.model_validate(ground_truth)


@pytest.mark.parametrize(
    "changes",
    [
        {"id": " "},
        {"request": " "},
        {"added_reason": " "},
        {"id": 123},
        {"request": 123},
        {"added_reason": 123},
        {"dataset_version": 0},
        {"dataset_version": -1},
        {"dataset_version": True},
        {"dataset_version": 1.0},
        {"extra": "no"},
        {"ground_truth": "answer"},
    ],
)
def test_entry_rejects_blank_wrong_type_and_extra_fields(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        GoldEntry.model_validate(entry(**changes))


@pytest.mark.parametrize(
    "missing",
    ["id", "request", "added_reason", "dataset_version", "ground_truth", "tool_checks"],
)
def test_entry_requires_all_fields(missing: str) -> None:
    value = entry()
    del value[missing]
    with pytest.raises(ValidationError):
        GoldEntry.model_validate(value)


def test_dataset_rejects_duplicate_ids_and_extra_fields() -> None:
    with pytest.raises(ValidationError, match="entries\\[1\\].id"):
        GoldDataset.model_validate({"schema_version": 1, "entries": [entry(), entry()]})
    with pytest.raises(ValidationError):
        GoldDataset.model_validate({"schema_version": 1, "entries": [], "extra": 1})


@pytest.mark.parametrize(argnames="version", argvalues=[-1, 0, 1, 2, 99])
def test_dataset_accepts_and_preserves_any_integer_schema_version(
    version: int,
    tmp_path: Path,
) -> None:
    entries = [entry(id="first"), entry(id="later", dataset_version=3)]
    dataset = GoldDataset.model_validate(
        obj={"schema_version": version, "entries": entries}
    )
    assert dataset.schema_version == version
    path = tmp_path / "dataset.json"
    write_dataset(path=path, entries=entries, schema_version=version)
    loaded = load_dataset_document(path=path)
    assert loaded.schema_version == version
    assert [item.id for item in loaded.entries] == ["first", "later"]
    assert [item.id for item in load_dataset(path=path)] == ["first", "later"]


@pytest.mark.parametrize(argnames="version", argvalues=[True, "1", 1.0])
def test_dataset_rejects_noninteger_schema_versions(version: object) -> None:
    with pytest.raises(expected_exception=ValidationError):
        GoldDataset.model_validate(
            obj={"schema_version": version, "entries": [entry()]}
        )


def test_loader_preserves_order_and_filters_cumulatively(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(
        path,
        [
            entry(id="third", dataset_version=3),
            entry(id="first"),
            entry(id="second", dataset_version=2),
        ],
    )
    assert [item.id for item in load_dataset(path)] == ["third", "first", "second"]
    assert [item.id for item in load_dataset(path, 2)] == ["first", "second"]
    assert [item.id for item in load_dataset(path=path, version=1)] == ["first"]
    assert load_dataset(path, version=0) == ()


@pytest.mark.parametrize("version", [-1, True, 1.0, "1"])
def test_loader_rejects_invalid_filter_values(tmp_path: Path, version: object) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(path, [entry()])
    with pytest.raises(ValueError, match="version"):
        load_dataset(path, version)  # type: ignore[arg-type]


def test_loader_validates_entire_file_before_filtering(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(path, [entry(), entry(id="bad", dataset_version=2, request=" ")])
    with pytest.raises(DatasetValidationError, match=r"entries\.1\.request"):
        load_dataset(path, 1)


def test_loader_reports_duplicate_ids_with_path(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(path, [entry(), entry()])
    with pytest.raises(DatasetValidationError) as caught:
        load_dataset(path)
    assert str(path) in str(caught.value)
    assert "entries[1].id" in str(caught.value)


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text('{"schema_version":1,"entries":[],"entries":[]}', encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="duplicate JSON key.*entries"):
        load_dataset(path)
    path.write_text(
        '{"schema_version":1,"entries":[{"id":"x","id":"y"}]}',
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="duplicate JSON key.*id"):
        load_dataset(path)


def test_loader_wraps_malformed_json_and_model_errors(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(DatasetValidationError) as malformed:
        load_dataset(path)
    assert str(path) in str(malformed.value)
    path.write_text('{"schema_version":1,"entries":[{}]}', encoding="utf-8")
    with pytest.raises(DatasetValidationError) as invalid:
        load_dataset(path)
    assert str(path) in str(invalid.value)
    assert "entries.0.id" in str(invalid.value)


def test_loader_wraps_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_bytes(b"\xff")
    with pytest.raises(DatasetValidationError) as caught:
        load_dataset(path)
    assert str(path) in str(caught.value)


def test_loader_preserves_filesystem_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.json")
    with pytest.raises(IsADirectoryError):
        load_dataset(tmp_path)


def test_default_dataset_is_independent_of_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    items = load_dataset()
    assert len(items) == 3
    assert all(item.dataset_version == 1 for item in items)


def test_cli_success_and_failure(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    write_dataset(path, [entry()])
    ok = subprocess.run(
        [sys.executable, "-m", "src.gold_dataset.validate", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0
    assert "1" in ok.stdout
    path.write_text("{", encoding="utf-8")
    bad = subprocess.run(
        [sys.executable, "-m", "src.gold_dataset.validate", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode != 0
    assert str(path) in bad.stderr


def test_cli_default_dataset() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.gold_dataset.validate"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "3" in result.stdout
