from pathlib import Path

import pytest

from src.tools.read_note import read_note

VAULT = str(Path(__file__).parent / "test_vault")
NESTED_NOTE = (
    "ai engineering/2 areas/ai engineering/evaluation/AI Evaluation Metrics.md"
)


@pytest.mark.parametrize(
    "notepath",
    ["ai engineering/Index.md", NESTED_NOTE],
)
def test_read_note_returns_full_utf8_content(notepath: str) -> None:
    assert read_note(VAULT, notepath) == (Path(VAULT) / notepath).read_text(
        encoding="utf-8"
    )


def test_read_note_raises_for_missing_note() -> None:
    with pytest.raises(FileNotFoundError):
        read_note(VAULT, "ai engineering/Does Not Exist.md")


def test_read_note_rejects_non_markdown_file(tmp_path: Path) -> None:
    (tmp_path / "private.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError):
        read_note(str(tmp_path), "private.txt")


@pytest.mark.parametrize("notepath", ["", "../outside.md", "/tmp/outside.md"])
def test_read_note_rejects_invalid_or_escaping_path(
    tmp_path: Path, notepath: str
) -> None:
    with pytest.raises(ValueError):
        read_note(str(tmp_path), notepath)


def test_read_note_rejects_symlink_escape(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside vault", encoding="utf-8")
    (vault / "linked.md").symlink_to(outside)

    with pytest.raises(ValueError):
        read_note(str(vault), "linked.md")
