from pathlib import Path

import pytest

from src.tools.get_backlinks import get_backlinks

VAULT = str(Path(__file__).parent / "test_vault")
EVALUATION = "ai engineering/2 areas/ai engineering/evaluation"


def test_finds_real_vault_backlinks_from_wikilinks_and_heading_links() -> None:
    """Find every real-vault note linking to the requested note."""
    assert get_backlinks(VAULT, f"{EVALUATION}/LLM-as-a-Judge.md") == [
        f"{EVALUATION}/AI Evaluation Metrics.md",
        f"{EVALUATION}/AI Evaluation System Roadmap.md",
        f"{EVALUATION}/Evaluation Methods.md",
        f"{EVALUATION}/RAG Evaluation.md",
        f"{EVALUATION}/Rubric Based Evaluation.md",
        f"{EVALUATION}/z> MOC AI Evaluation.md",
    ]


def test_real_vault_alias_and_embed_links_count_as_backlinks() -> None:
    """Count aliases and embedded notes as backlinks to their targets."""
    roadmap = f"{EVALUATION}/AI Evaluation System Roadmap.md"
    golden = f"{EVALUATION}/Golden Evaluation Dataset.md"

    # The roadmap uses [[Golden Evaluation Dataset|dataset]].
    assert roadmap in get_backlinks(VAULT, golden)
    # The golden dataset embeds a heading from the roadmap.
    assert golden in get_backlinks(VAULT, roadmap)


def test_resolves_qualified_links_and_omits_ambiguous_or_missing_targets(
    tmp_path: Path,
) -> None:
    """Resolve qualified targets while ignoring ambiguous and missing ones."""
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    (tmp_path / "alpha" / "Shared.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "beta" / "Shared.md").write_text("beta", encoding="utf-8")
    (tmp_path / "unique.md").write_text("unique", encoding="utf-8")
    (tmp_path / "z.md").write_text(
        "[[Shared]] [[alpha/Shared|alias]] [[unique#Heading]] "
        "![[unique|embed]] [[Missing]] [[alpha/Shared]]",
        encoding="utf-8",
    )
    (tmp_path / "a.md").write_text("[[alpha/Shared.md]]", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("[[alpha/Shared]]", encoding="utf-8")

    assert get_backlinks(str(tmp_path), "alpha/Shared.md") == ["a.md", "z.md"]
    assert get_backlinks(str(tmp_path), "beta/Shared.md") == []
    assert get_backlinks(str(tmp_path), "unique.md") == ["z.md"]


def test_unique_unqualified_stem_and_duplicate_links_return_one_source(
    tmp_path: Path,
) -> None:
    """Accept a unique bare target and report a source only once."""
    (tmp_path / "target.md").write_text("target", encoding="utf-8")
    (tmp_path / "z.md").write_text(
        "[[target]] [[target|alias]] ![[target#Heading]]", encoding="utf-8"
    )
    (tmp_path / "a.md").write_text("[[target]]", encoding="utf-8")

    assert get_backlinks(str(tmp_path), "target.md") == ["a.md", "z.md"]


def test_missing_requested_note_raises_file_not_found(tmp_path: Path) -> None:
    """Raise FileNotFoundError when the backlink target is missing."""
    with pytest.raises(FileNotFoundError):
        get_backlinks(str(tmp_path), "missing.md")


@pytest.mark.parametrize("notepath", ["", "../outside.md", "/tmp/outside.md", "a.txt"])
def test_invalid_requested_note_path_raises_value_error(
    tmp_path: Path, notepath: str
) -> None:
    """Reject empty, escaping, absolute, and non-Markdown target paths."""
    with pytest.raises(ValueError):
        get_backlinks(str(tmp_path), notepath)


def test_requested_note_symlink_cannot_escape_vault(tmp_path: Path) -> None:
    """Reject a backlink target symlink that resolves outside the vault."""
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (vault / "linked.md").symlink_to(outside)

    with pytest.raises(ValueError):
        get_backlinks(str(vault), "linked.md")
