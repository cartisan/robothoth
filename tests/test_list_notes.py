from pathlib import Path

import pytest

from src.tools.list_notes import list_notes

VAULT = str(Path(__file__).parent / "test_vault")
EVALUATION = "2 areas/ai engineering/evaluation"


def test_lists_all_vault_notes_as_sorted_relative_markdown_paths() -> None:
    """List every Markdown note as a sorted path relative to the vault."""
    notes = list_notes(VAULT)

    assert len(notes) == 14
    assert notes == sorted(notes)
    assert "Index.md" in notes
    assert f"{EVALUATION}/LLM-as-a-Judge.md" in notes
    assert "2 areas/machine learning/z> MOC ML.md" in notes
    assert all(note.endswith(".md") and not note.startswith("/") for note in notes)


def test_filters_to_nested_directory_recursively() -> None:
    """Limit recursive listing to a requested nested directory."""
    notes = list_notes(VAULT, "2 areas/ai engineering")

    assert len(notes) == 12
    assert notes == sorted(notes)
    assert f"{EVALUATION}/RAG.md" in notes
    assert (
        "2 areas/ai engineering/"
        "Agentic Software Engineering Factory.md"
    ) in notes
    assert all(
        note.startswith("2 areas/ai engineering/") for note in notes
    )


def test_empty_directory_and_non_markdown_files(tmp_path: Path) -> None:
    """Return no entries for empty directories and ignore non-Markdown files."""
    (tmp_path / "empty").mkdir()
    (tmp_path / "draft.txt").write_text("not a note")
    (tmp_path / "note.md").write_text("a note")

    assert list_notes(str(tmp_path), "empty") == []
    assert list_notes(str(tmp_path)) == ["note.md"]


def test_missing_vault_or_directory_raises_file_not_found(tmp_path: Path) -> None:
    """Raise FileNotFoundError when the vault or requested directory is absent."""
    with pytest.raises(FileNotFoundError):
        list_notes(str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError):
        list_notes(str(tmp_path), "absent")


@pytest.mark.parametrize("path", ["../outside", "/tmp", "nested/../../outside"])
def test_rejects_paths_outside_vault(tmp_path: Path, path: str) -> None:
    """Reject absolute and parent paths that could escape the vault."""
    with pytest.raises(ValueError):
        list_notes(str(tmp_path), path)
