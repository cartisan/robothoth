from pathlib import Path

import pytest

from src.tools.list_notes import list_notes

VAULT = Path(__file__).parent / "test_vault"
EVALUATION = "ai engineering/2 areas/ai engineering/evaluation"


def test_lists_all_vault_notes_as_sorted_relative_markdown_paths() -> None:
    notes = list_notes(VAULT)

    assert len(notes) == 14
    assert notes == sorted(notes)
    assert "ai engineering/Index.md" in notes
    assert f"{EVALUATION}/LLM-as-a-Judge.md" in notes
    assert "ai engineering/2 areas/machine learning/z> MOC ML.md" in notes
    assert all(note.endswith(".md") and not note.startswith("/") for note in notes)


def test_filters_to_nested_directory_recursively() -> None:
    notes = list_notes(VAULT, "ai engineering/2 areas/ai engineering")

    assert len(notes) == 12
    assert notes == sorted(notes)
    assert f"{EVALUATION}/RAG.md" in notes
    assert (
        "ai engineering/2 areas/ai engineering/"
        "Agentic Software Engineering Factory.md"
    ) in notes
    assert all(
        note.startswith("ai engineering/2 areas/ai engineering/") for note in notes
    )


def test_empty_directory_and_non_markdown_files(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    (tmp_path / "draft.txt").write_text("not a note")
    (tmp_path / "note.md").write_text("a note")

    assert list_notes(tmp_path, "empty") == []
    assert list_notes(tmp_path) == ["note.md"]


def test_missing_vault_or_directory_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_notes(tmp_path / "absent")
    with pytest.raises(FileNotFoundError):
        list_notes(tmp_path, "absent")


@pytest.mark.parametrize("path", ["../outside", "/tmp", "nested/../../outside"])
def test_rejects_paths_outside_vault(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError):
        list_notes(tmp_path, path)
