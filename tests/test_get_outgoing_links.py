from pathlib import Path

import pytest

from src.tools.get_outgoing_links import get_outgoing_links

VAULT = str(Path(__file__).parent / "test_vault")
EVALUATION = "ai engineering/2 areas/ai engineering/evaluation"


def test_real_vault_resolves_aliases_and_deduplicates_links() -> None:
    """Resolve real-vault aliases and return each outgoing note once."""
    assert get_outgoing_links(VAULT, f"{EVALUATION}/RAG Evaluation.md") == [
        f"{EVALUATION}/AI Evaluation Metrics.md",
        f"{EVALUATION}/Golden Evaluation Dataset.md",
        f"{EVALUATION}/LLM-as-a-Judge.md",
        f"{EVALUATION}/RAG.md",
        f"{EVALUATION}/Rubric Based Evaluation.md",
        f"{EVALUATION}/z> MOC AI Evaluation.md",
    ]


def test_real_vault_resolves_embedded_note_with_heading() -> None:
    """Resolve embedded notes that include a heading fragment."""
    links = get_outgoing_links(VAULT, f"{EVALUATION}/Golden Evaluation Dataset.md")

    assert links == [
        f"{EVALUATION}/AI Evaluation Fundamentals.md",
        f"{EVALUATION}/AI Evaluation System Roadmap.md",
        f"{EVALUATION}/Rubric Based Evaluation.md",
        f"{EVALUATION}/z> MOC AI Evaluation.md",
    ]


def test_resolves_path_qualified_and_unique_bare_targets(tmp_path: Path) -> None:
    """Resolve qualified paths and unique bare note names with link metadata."""
    (tmp_path / "area").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "area" / "First.md").write_text("", encoding="utf-8")
    (tmp_path / "other" / "Second.md").write_text("", encoding="utf-8")
    (tmp_path / "source.md").write_text(
        "[[area/First|alias]] [[other/Second.md#section]] [[First#heading]] "
        "![[Second|embed]]",
        encoding="utf-8",
    )

    assert get_outgoing_links(str(tmp_path), "source.md") == [
        "area/First.md",
        "other/Second.md",
    ]


def test_omits_missing_ambiguous_and_attachment_targets(tmp_path: Path) -> None:
    """Ignore missing, ambiguous, and non-Markdown link targets."""
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    (tmp_path / "one" / "Duplicate.md").write_text("", encoding="utf-8")
    (tmp_path / "two" / "Duplicate.md").write_text("", encoding="utf-8")
    (tmp_path / "one" / "Unique.md").write_text("", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"png")
    (tmp_path / "source.md").write_text(
        "[[Duplicate]] [[Missing]] ![[image.png]] [[one/Duplicate]] [[Unique]]",
        encoding="utf-8",
    )

    assert get_outgoing_links(str(tmp_path), "source.md") == [
        "one/Duplicate.md",
        "one/Unique.md",
    ]


def test_note_with_no_links_returns_empty_list(tmp_path: Path) -> None:
    """Return no outgoing links for a note without wikilinks."""
    (tmp_path / "plain.md").write_text("No wikilinks here.", encoding="utf-8")

    assert get_outgoing_links(str(tmp_path), "plain.md") == []


def test_missing_source_raises_file_not_found(tmp_path: Path) -> None:
    """Raise FileNotFoundError when the source note is missing."""
    with pytest.raises(FileNotFoundError):
        get_outgoing_links(str(tmp_path), "missing.md")


@pytest.mark.parametrize(
    "notepath", ["", "../outside.md", "/tmp/outside.md", "file.txt"]
)
def test_rejects_invalid_source_path(tmp_path: Path, notepath: str) -> None:
    """Reject empty, absolute, parent, and non-Markdown source paths."""
    with pytest.raises(ValueError):
        get_outgoing_links(str(tmp_path), notepath)


def test_rejects_source_symlink_escape(tmp_path: Path) -> None:
    """Reject a source symlink that resolves outside the vault."""
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (vault / "link.md").symlink_to(outside)

    with pytest.raises(ValueError):
        get_outgoing_links(str(vault), "link.md")
