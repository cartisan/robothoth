"""Verify the knowledge snapshot and the evidence behind its gold cases."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from src.evaluation.runner import DEFAULT_VAULT_PATH
from src.gold_dataset.validate import load_dataset
from src.tools.get_backlinks import get_backlinks
from src.tools.get_outgoing_links import get_outgoing_links
from src.tools.list_notes import list_notes
from src.tools.read_note import read_note
from src.tools.registry import registry
from src.tools.search_notes import search_notes

TESTS = Path(__file__).resolve().parent
VAULT = str(object=TESTS / "extended_test_vault")
ML = "2 areas/machine learning/"
EVALUATION = "2 areas/ai engineering/evaluation/"
SERVICE = "4 archive/example company/service-a/"
SOFT = "2 areas/soft skills/"
COVARIANCE = (
    ML + "Covariance - a measure of the joint variability of two variables.md"
)
VARIANCE = (
    ML + "Variance - a measure of data spread that gives more weight to outliers.md"
)
MAHALANOBIS = (
    ML + "Mahalanobis Distance - the distance between a point and a distribution.md"
)
HARD_CASE_IDS = {
    "contextual-guidance-synthesis",
    "outdated-guidance-detection",
    "reference-based-calculation",
    "multipart-concept-application",
    "source-caveat-preservation",
}


def test_snapshot_matches_manifest_and_excludes_nonknowledge_records() -> None:
    """Keep the snapshot complete, byte-stable, and free of excluded records.

    Verification uses only repository files, without accessing the source vault.
    The manifest accounts for every note and attachment but stays outside the
    navigable corpus so metadata does not become an answer source.
    """
    manifest = json.loads(
        s=(TESTS / "extended_test_vault_manifest.json").read_text(encoding="utf-8")
    )
    assert "source_vault" not in manifest
    root = Path(VAULT)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob(pattern="*")
        if path.is_file()
    }
    assert actual == {item["path"] for item in manifest["files"]}
    notes = list_notes(vault_path=VAULT)
    assert len(notes) == manifest["note_count"] == 152
    assert len(actual) - len(notes) == manifest["attachment_count"] == 0
    assert DEFAULT_VAULT_PATH == root
    assert not any(
        path.startswith(("0 inbox/", "1 projects/", "5 automated/"))
        or "/1 weekly/" in path
        or "/2 quarterly/" in path
        or "/4 feedback/" in path
        or "/5 experiments/" in path
        or "/incidents/" in path
        or Path(path).name in {"PDP 2024.md", "LLM Prompt Archive.md"}
        or Path(path).name.startswith(".")
        for path in actual
    )
    for item in manifest["files"]:
        path = root / item["path"]
        assert not path.is_symlink()
        assert hashlib.sha256(string=path.read_bytes()).hexdigest() == item["sha256"]
        if item["kind"] == "attachment":
            assert item["referenced_by"]
            assert set(item["referenced_by"]) <= set(notes)
    assert manifest["unavailable_attachments"]
    assert manifest["unresolved_note_links"]


def test_small_fixture_is_preserved_in_the_redacted_snapshot() -> None:
    """Retain every original fixture note, including its credential redaction."""
    small = TESTS / "test_vault"
    for relative in list_notes(vault_path=str(object=small)):
        assert read_note(vault_path=VAULT, notepath=relative) == read_note(
            vault_path=str(object=small), notepath=relative
        )
    assert "[redacted API key]" in read_note(
        vault_path=VAULT, notepath=EVALUATION + "RAG.md"
    )
    patterns = (
        r"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}"
        r"|(?:AKIA|ASIA)[A-Z0-9]{16}"
        r"|gh[pousr]_[A-Za-z0-9]{20,}"
        r"|github_pat_[A-Za-z0-9_]{20,}"
        r"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        r'|(?i:"SecretString"\s*:\s*"(?!<secret-value>")[^"]+")'
    )
    for relative in list_notes(vault_path=VAULT):
        assert re.search(
            pattern=patterns, string=read_note(vault_path=VAULT, notepath=relative)
        ) is None, relative


def test_new_cases_have_ground_truth_readable_evidence_and_tool_coverage() -> None:
    """Require usable expectations, real evidence, and coverage of every tool."""
    entries = [entry for entry in load_dataset() if entry.dataset_version == 2]
    registered = {tool["name"] for tool in registry.tool_definitions()}
    covered: set[str] = set()
    hard_cases = []
    for entry in entries:
        assert entry.ground_truth.ideal_answer
        checks = entry.tool_checks
        demanded = set(checks.required_tools)
        if checks.required_note_reads:
            demanded.add("read_note")
        assert demanded <= registered
        assert set(checks.forbidden_tools) <= registered
        if checks.allowed_tools is not None:
            assert set(checks.allowed_tools) <= registered
        covered |= demanded
        for relative in checks.required_note_reads:
            assert read_note(vault_path=VAULT, notepath=relative).strip()
        if entry.id in HARD_CASE_IDS:
            assert len(checks.required_note_reads) >= 3
            assert len(demanded) >= 2
            assert entry.added_reason.startswith("Hard:")
            hard_cases.append(entry.id)
    assert len(entries) == 11
    assert covered == registered
    assert set(hard_cases) == HARD_CASE_IDS
    testing = next(
        entry for entry in entries
        if entry.id == "multi-source-integration"
    )
    assert set(testing.tool_checks.required_note_reads) == {
        "2 areas/testing/Parametrize tests with error context or nullcontext.md",
        "2 areas/testing/Test that raised error contains the right message.md",
    }


@pytest.mark.parametrize(
    argnames="query, expected",
    argvalues=[
        (
            "nullcontext",
            "2 areas/testing/Parametrize tests with error context or nullcontext.md",
        ),
        (
            "right type of error",
            "2 areas/testing/Test that raised error contains the right message.md",
        ),
        ("max_features", ML + "Feature order matters for tree based transformers.md"),
        ("graph is not the goal", SOFT + "z MOC Personal Knowledge Management.md"),
    ],
)
def test_evidence_is_discoverable_by_content(query: str, expected: str) -> None:
    """Find source evidence by content, including an irregular Unicode title."""
    assert expected in search_notes(vault_path=VAULT, query=query)


def test_graph_only_reference_answers_match_the_snapshot() -> None:
    """Ground the two complete graph answers in actual resolved links."""
    assert get_outgoing_links(
        vault_path=VAULT, notepath=ML + "Monitoring and Observability.md"
    ) == [
        "4 archive/Observability Course - Designing Effective Monitoring.md",
        "4 archive/Observability Course - Intro.md",
        "4 archive/Observability Course - Monitoring Metrics.md",
    ]
    assert get_backlinks(
        vault_path=VAULT,
        notepath=COVARIANCE,
    ) == [
        MAHALANOBIS,
        VARIANCE,
        ML + "c\u0327.md",
    ]


def test_hard_cases_have_the_graph_routes_their_requests_describe() -> None:
    """Keep required graph exploration feasible without repairing stale links."""
    version_note = SERVICE + "Model Compatibility and Release Versions.md"
    assert version_note in get_outgoing_links(
        vault_path=VAULT,
        notepath=SERVICE + "Running an Older Service Version During Retraining.md",
    )
    assert set(
        get_outgoing_links(
            vault_path=VAULT,
            notepath=MAHALANOBIS,
        )
    ) == {
        COVARIANCE,
        VARIANCE,
    }
    assert {
        EVALUATION + "AI Evaluation Metrics.md",
        EVALUATION + "Golden Evaluation Dataset.md",
    } <= set(
        get_outgoing_links(vault_path=VAULT, notepath=EVALUATION + "RAG Evaluation.md")
    )
    assert SOFT + "Knowledge work should accrete.md" in get_backlinks(
        vault_path=VAULT, notepath=SOFT + "Note taking is thinking.md"
    )
