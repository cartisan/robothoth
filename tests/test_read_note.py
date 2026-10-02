from pathlib import Path

import pytest

from src.tools.read_note import read_note

VAULT = str(Path(__file__).parent / "test_vault")
NESTED_NOTE = (
    "2 areas/ai engineering/evaluation/AI Evaluation Metrics.md"
)


@pytest.mark.parametrize(
    "notepath",
    ["Index.md", NESTED_NOTE],
)
def test_read_note_returns_full_utf8_content(notepath: str) -> None:
    """Read complete UTF-8 content from root and nested Markdown notes."""
    assert read_note(VAULT, notepath) == (Path(VAULT) / notepath).read_text(
        encoding="utf-8"
    )


def test_read_note_raises_for_missing_note() -> None:
    """Raise FileNotFoundError when the requested note is missing."""
    with pytest.raises(FileNotFoundError):
        read_note(VAULT, "Does Not Exist.md")


def test_read_note_rejects_non_markdown_file(tmp_path: Path) -> None:
    """Reject an existing file whose path does not use the Markdown extension."""
    (tmp_path / "private.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError):
        read_note(str(tmp_path), "private.txt")


@pytest.mark.parametrize("notepath", ["", "../outside.md", "/tmp/outside.md"])
def test_read_note_rejects_invalid_or_escaping_path(
    tmp_path: Path, notepath: str
) -> None:
    """Reject empty, absolute, and parent paths outside the vault."""
    with pytest.raises(ValueError):
        read_note(str(tmp_path), notepath)


def test_read_note_rejects_symlink_escape(tmp_path: Path) -> None:
    """Reject a Markdown symlink that resolves outside the vault."""
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside vault", encoding="utf-8")
    (vault / "linked.md").symlink_to(outside)

    with pytest.raises(ValueError):
        read_note(str(vault), "linked.md")
