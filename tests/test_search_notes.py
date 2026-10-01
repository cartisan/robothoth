from pathlib import Path

import pytest

from src.tools.search_notes import search_notes

TEST_VAULT = str(Path(__file__).parent / "test_vault")
EVALUATION_DIR = "ai engineering/2 areas/ai engineering/evaluation"


def test_searches_real_vault_content_case_insensitively() -> None:
    """Match note content without requiring the query's original casing."""
    assert search_notes(TEST_VAULT, "MEDIOCRE METRICS ON GOOD DATA") == [
        f"{EVALUATION_DIR}/Golden Evaluation Dataset.md"
    ]


def test_searches_real_vault_with_python_regex() -> None:
    """Apply Python regular expression syntax when searching note content."""
    assert search_notes(TEST_VAULT, r"mediocre\s+metrics\s+on\s+good\s+data") == [
        f"{EVALUATION_DIR}/Golden Evaluation Dataset.md"
    ]


def test_returns_sorted_relative_paths_and_each_note_once(tmp_path: Path) -> None:
    """Return each matching note once in sorted vault-relative order."""
    (tmp_path / "z.md").write_text("needle needle", encoding="utf-8")
    nested = tmp_path / "folder"
    nested.mkdir()
    (nested / "a.md").write_text("needle", encoding="utf-8")
    (nested / "c.md").write_text("NEEDLE", encoding="utf-8")

    assert search_notes(str(tmp_path), "needle") == [
        "folder/a.md",
        "folder/c.md",
        "z.md",
    ]


def test_searches_content_only_and_ignores_non_markdown_files(tmp_path: Path) -> None:
    """Search file contents while ignoring matching text in non-Markdown files."""
    (tmp_path / "needle.md").write_text("unrelated text", encoding="utf-8")
    (tmp_path / "other.md").write_text("needle is here", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("needle", encoding="utf-8")

    assert search_notes(str(tmp_path), "needle") == ["other.md"]


def test_no_matches_or_notes_returns_empty_list(tmp_path: Path) -> None:
    """Return an empty list when the vault has no matching note."""
    assert search_notes(str(tmp_path), "anything") == []
    (tmp_path / "one.md").write_text("some content", encoding="utf-8")
    assert search_notes(str(tmp_path), "absent") == []


def test_missing_vault_raises_file_not_found(tmp_path: Path) -> None:
    """Raise FileNotFoundError when the search root is missing."""
    with pytest.raises(FileNotFoundError):
        search_notes(str(tmp_path / "missing"), "anything")


@pytest.mark.parametrize("query", ["", "[", "("])
def test_empty_or_invalid_regex_raises_value_error(tmp_path: Path, query: str) -> None:
    """Reject empty and syntactically invalid regular expressions."""
    with pytest.raises(ValueError):
        search_notes(str(tmp_path), query)
