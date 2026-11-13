"""Calibration files validate references, nullable ratings, and generation failures."""

import json
import os
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from test_harness import ScriptedChatModel

from src.evaluation.calibration import generate_dataset as generator
from src.evaluation.calibration.dataset import (
    CalibrationDataset,
    Candidate,
    HumanRatings,
    load_dataset,
    save_dataset,
)
from src.gold_dataset.validate import GoldDataset, load_dataset_document


def make_dataset() -> CalibrationDataset:
    """Build a two-answer fixture with one failed candidate."""
    return CalibrationDataset.model_validate(
        obj={
            "schema_version": 1,
            "dataset_id": "fixture",
            "created_at": "now",
            "generation": {
                "gold_schema_version": 3,
                "gold_dataset_path": "gold.json",
                "vault_path": "vault",
                "system_prompt": {"version": 1, "text": "System"},
                "max_api_calls": 20,
                "models": {
                    "model-a": {
                        "provider": "openai",
                        "model": "model-a",
                        "settings": {},
                    },
                    "model-b": {
                        "provider": "openai",
                        "model": "model-b",
                        "settings": {},
                    },
                    "model-c": {
                        "provider": "openai",
                        "model": "model-c",
                        "settings": {},
                    },
                },
            },
            "questions": [
                {
                    "case_id": "one",
                    "request": "Question",
                    "ideal_answer": "Ideal",
                    "candidates": [
                        {
                            "candidate_id": "a",
                            "model_id": "model-a",
                            "answer": "Answer A",
                        },
                        {
                            "candidate_id": "b",
                            "model_id": "model-b",
                            "answer": "Answer B",
                        },
                        {
                            "candidate_id": "c",
                            "model_id": "model-c",
                            "generation_error": {
                                "type": "RuntimeError",
                                "message": "offline",
                                "evidence": None,
                            },
                        },
                    ],
                }
            ],
        }
    )


@pytest.mark.parametrize("value", [None, 1, 5])
def test_nullable_ratings_round_trip(value: int | None, tmp_path: Path) -> None:
    dataset = make_dataset()
    dataset.questions[0].candidates[0].human_ratings = HumanRatings(content=value)
    path = tmp_path / "dataset.json"
    save_dataset(dataset=dataset, path=path)
    assert load_dataset(path=path) == dataset
    assert json.loads(path.read_text())["questions"][0]["candidates"][0][
        "human_ratings"
    ] == {"content": value, "concision": None, "usefulness": None}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("value", [0, 6, True, "3", 3.0])
def test_ratings_reject_coercion_and_out_of_range(value: object) -> None:
    with pytest.raises(ValidationError):
        HumanRatings.model_validate(obj={"content": value})


@pytest.mark.parametrize(
    "changes",
    [
        {"answer": None},
        {"answer": " "},
        {"generation_error": {"type": "Error", "message": "Failed", "evidence": None}},
        {"extra": 1},
        {"model_id": " "},
    ],
)
def test_candidates_require_exactly_answer_or_error(changes: dict) -> None:
    with pytest.raises(ValidationError):
        Candidate.model_validate(
            obj={"candidate_id": "a", "model_id": "m", "answer": "Answer", **changes}
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_question",
        "duplicate_candidate",
        "unknown_model",
        "blank_reference",
        "version",
        "extra",
    ],
)
def test_dataset_rejects_invalid_contracts(mutation: str) -> None:
    raw = make_dataset().model_dump()
    if mutation == "duplicate_question":
        raw["questions"] *= 2
    elif mutation == "duplicate_candidate":
        raw["questions"][0]["candidates"] *= 2
    elif mutation == "unknown_model":
        raw["questions"][0]["candidates"][0]["model_id"] = "missing"
    elif mutation == "blank_reference":
        raw["questions"][0]["ideal_answer"] = " "
    elif mutation == "version":
        raw["schema_version"] = 99
    else:
        raw["extra"] = True
    with pytest.raises(ValidationError):
        CalibrationDataset.model_validate(obj=raw)


def test_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_dataset(path=path)


def test_generation_is_independent_records_failure_and_never_judges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gold = load_dataset_document()
    gold = GoldDataset(schema_version=3, entries=gold.entries[:2])
    monkeypatch.setattr(generator, "load_dataset_document", lambda path: gold)
    models = [
        ScriptedChatModel(
            responses=[AIMessage(content="first"), AIMessage(content="second")]
        ),
        ScriptedChatModel(
            responses=[RuntimeError("offline"), AIMessage(content="recovered")]
        ),
    ]
    create = MagicMock(side_effect=models)
    monkeypatch.setattr(generator.CandidateConfig, "create_model", create)
    monkeypatch.setattr(generator, "MODELS", ("first", "second"))
    dataset = generator.generate_dataset(output_path=tmp_path / "answers.json")
    assert len(dataset.questions) == 2
    assert dataset.questions[0].candidates[1].generation_error is not None
    assert dataset.questions[1].candidates[1].answer == "recovered"
    assert all(
        candidate.human_ratings == HumanRatings()
        for question in dataset.questions
        for candidate in question.candidates
    )
    for model in models:
        for messages in model.requests:
            assert [m.type for m in messages] == ["system", "human"]
    assert load_dataset(path=tmp_path / "answers.json") == dataset
    create.side_effect = models
    again = generator.generate_dataset(output_path=tmp_path / "again.json")
    assert again.dataset_id != dataset.dataset_id


@pytest.mark.parametrize("version", [True, 1.0, "1", 0, 2])
def test_schema_version_is_strict(version: object) -> None:
    raw = make_dataset().model_dump()
    raw["schema_version"] = version
    with pytest.raises(ValidationError):
        CalibrationDataset.model_validate(obj=raw)


def test_generation_disables_langfuse_before_loading_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gold = load_dataset_document()
    monkeypatch.setenv(name="LANGFUSE_TRACING_ENABLED", value="true")

    def load_gold(path: Path) -> GoldDataset:
        assert os.environ["LANGFUSE_TRACING_ENABLED"] == "false"
        return GoldDataset(schema_version=3, entries=gold.entries[:1])

    monkeypatch.setattr(generator, "load_dataset_document", load_gold)
    monkeypatch.setattr(generator, "MODELS", ("fixture",))
    monkeypatch.setattr(
        generator.CandidateConfig,
        "create_model",
        lambda self: ScriptedChatModel(responses=[AIMessage(content="Answer")]),
    )
    dataset = generator.generate_dataset(output_path=tmp_path / "answers.json")
    assert dataset.questions[0].candidates[0].answer == "Answer"


@pytest.fixture
def mocked_generation(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Generate one answer without credentials or external calls."""
    gold = load_dataset_document()
    monkeypatch.setattr(
        generator,
        "load_dataset_document",
        lambda path: GoldDataset(schema_version=3, entries=gold.entries[:1]),
    )
    monkeypatch.setattr(generator, "MODELS", ("fixture",))
    monkeypatch.setattr(generator.CandidateConfig, "create_model", MagicMock())
    harness = MagicMock(return_value=SimpleNamespace(final_answer="Answer"))
    monkeypatch.setattr(generator, "run_in_harness", harness)
    return harness


@pytest.mark.parametrize("filename", ["dataset.json", "answers.json"])
def test_regeneration_archives_each_previous_dataset(
    tmp_path: Path, mocked_generation: MagicMock, filename: str
) -> None:
    path = tmp_path / "nested" / filename
    first = generator.generate_dataset(output_path=path)
    assert list(path.parent.iterdir()) == [path]
    first.questions[0].candidates[0].human_ratings = HumanRatings(content=5)
    # Preserve human edits and noncanonical whitespace byte for byte.
    original = first.model_dump_json(indent=4).encode() + b"\n\n"
    path.write_bytes(original)

    def answer(**kwargs: object) -> SimpleNamespace:
        assert path.read_bytes() == original
        assert list(path.parent.iterdir()) == [path]
        return SimpleNamespace(final_answer="New answer")

    mocked_generation.side_effect = answer
    second = generator.generate_dataset(output_path=path)
    assert second.questions[0].candidates[0].answer == "New answer"
    archive = path.with_name(f"{path.stem}_{first.dataset_id}{path.suffix}")
    assert archive.read_bytes() == original
    assert load_dataset(path=archive) == first
    assert load_dataset(path=path) == second

    mocked_generation.side_effect = None
    second_bytes = path.read_bytes()
    third = generator.generate_dataset(output_path=path)
    second_archive = path.with_name(f"{path.stem}_{second.dataset_id}{path.suffix}")
    assert second_archive.read_bytes() == second_bytes
    assert archive.read_bytes() == original
    assert load_dataset(path=path) == third
    assert len({first.dataset_id, second.dataset_id, third.dataset_id}) == 3
    assert set(path.parent.iterdir()) == {path, archive, second_archive}


@pytest.mark.parametrize(
    "problem",
    ["invalid", "traversal", "backslash", "absolute", "nul", "collision"],
)
def test_regeneration_rejects_invalid_archive_without_changing_files(
    tmp_path: Path, mocked_generation: MagicMock, problem: str
) -> None:
    path = tmp_path / "dataset.json"
    dataset = make_dataset()
    unsafe_ids = {
        "traversal": "../outside",
        "backslash": "..\\outside",
        "absolute": "/outside",
        "nul": "bad\x00id",
    }
    dataset.dataset_id = unsafe_ids.get(problem, "fixture")
    save_dataset(dataset=dataset, path=path)
    if problem == "invalid":
        path.write_text("not JSON")
    if problem == "collision":
        (tmp_path / "dataset_fixture.json").write_bytes(b"existing archive")
    before = {item.name: item.read_bytes() for item in tmp_path.iterdir()}
    with pytest.raises((ValueError, FileExistsError)):
        generator.generate_dataset(output_path=path)
    assert {item.name: item.read_bytes() for item in tmp_path.iterdir()} == before


@pytest.mark.parametrize(
    ("operation", "existing"),
    [
        ("staging", False),
        ("staging", True),
        ("rename", True),
        ("publish", False),
        ("publish", True),
    ],
)
def test_generation_persistence_failure_preserves_output_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocked_generation: MagicMock,
    existing: bool,
    operation: str,
) -> None:
    path = tmp_path / "dataset.json"
    if existing:
        save_dataset(dataset=make_dataset(), path=path)
    before = {item.name: item.read_bytes() for item in tmp_path.iterdir()}

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError(f"{operation} failed")

    if operation == "staging":
        monkeypatch.setattr(generator.os, "fsync", fail)
    elif operation == "rename":
        monkeypatch.setattr(Path, "rename", fail)
    else:
        replace = Path.replace

        def fail_publish(self: Path, target: str | Path) -> Path:
            if self.name != "dataset_fixture.json":
                raise OSError("publish failed")
            return replace(self, target=target)

        monkeypatch.setattr(Path, "replace", fail_publish)
    with pytest.raises(OSError, match=f"{operation} failed"):
        generator.generate_dataset(output_path=path)
    assert {item.name: item.read_bytes() for item in tmp_path.iterdir()} == before


def test_rating_saves_update_in_place_without_archiving(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    dataset = make_dataset()
    save_dataset(dataset=dataset, path=path)
    dataset.questions[0].candidates[0].human_ratings.content = 4
    save_dataset(dataset=dataset, path=path)
    assert load_dataset(path=path) == dataset
    assert list(tmp_path.iterdir()) == [path]


def test_generation_does_not_overwrite_dangling_output_symlink(
    tmp_path: Path, mocked_generation: MagicMock
) -> None:
    path = tmp_path / "dataset.json"
    target = tmp_path / "missing.json"
    path.symlink_to(target=target)
    with pytest.raises(FileNotFoundError):
        generator.generate_dataset(output_path=path)
    assert path.is_symlink()
    assert path.readlink() == target
    assert list(tmp_path.iterdir()) == [path]


def balanced_dataset() -> CalibrationDataset:
    """Include distinct requests and a failed generation on each side."""
    dataset = make_dataset()
    other = dataset.questions[0].model_copy(deep=True)
    other.case_id = "two"
    other.request = "Held out request"
    other.ideal_answer = "Held out ideal"
    other.candidates[0].answer = "Held out A"
    other.candidates[1].answer = "Held out B"
    dataset.questions.append(other)
    return dataset


def test_create_test_val_split_is_grouped_reproducible_ordered_and_rating_independent(
) -> None:
    dataset = balanced_dataset()
    before = dataset.model_dump_json()
    random_state = random.getstate()
    test, validation = dataset.create_test_val_split()
    assert [q.case_id for q in test.questions] == ["two"]
    assert [q.case_id for q in validation.questions] == ["one"]
    assert test.questions[0].candidates[-1].generation_error is not None
    assert dataset.model_dump_json() == before
    assert random.getstate() == random_state
    dataset.questions[0].candidates[0].human_ratings.content = 5
    again = dataset.create_test_val_split()
    assert again[1].questions[0].candidates[0].human_ratings.content == 5
    assert [q.case_id for q in again[0].questions] == ["two"]


def test_create_test_val_split_bundled_has_seven_candidates_per_model_per_side(
) -> None:
    dataset = load_dataset(path=Path("src/evaluation/calibration/dataset.json"))
    test, validation = dataset.create_test_val_split()
    left = {q.case_id for q in test.questions}
    right = {q.case_id for q in validation.questions}
    assert len(left) == len(right) == 7
    assert not left & right
    assert left | right == {q.case_id for q in dataset.questions}
    for view in (test, validation):
        assert Counter(c.model_id for q in view.questions for c in q.candidates) == {
            model: 7 for model in dataset.generation.models
        }
        assert view.questions == [
            q
            for q in dataset.questions
            if q.case_id in {v.case_id for v in view.questions}
        ]

    reversed_source = dataset.model_copy(
        update={"questions": list(reversed(dataset.questions))}
    )
    reversed_test, reversed_validation = reversed_source.create_test_val_split()
    assert [q.case_id for q in reversed_test.questions] == list(
        reversed([q.case_id for q in test.questions])
    )
    assert [q.case_id for q in reversed_validation.questions] == list(
        reversed([q.case_id for q in validation.questions])
    )


@pytest.mark.parametrize("shape", ["one", "odd", "composition", "multiplicity"])
def test_create_test_val_split_unsupported_shapes_fail_clearly(shape: str) -> None:
    dataset = balanced_dataset()
    if shape == "one":
        dataset.questions.pop()
    elif shape == "odd":
        extra = dataset.questions[0].model_copy(deep=True)
        extra.case_id = "three"
        dataset.questions.append(extra)
    elif shape == "composition":
        dataset.questions[1].candidates.pop()
    else:
        duplicate_model = dataset.questions[1].candidates[0].model_copy(deep=True)
        duplicate_model.candidate_id = "another-answer"
        dataset.questions[1].candidates.append(duplicate_model)
    with pytest.raises(ValueError, match="even|composition"):
        dataset.create_test_val_split()
