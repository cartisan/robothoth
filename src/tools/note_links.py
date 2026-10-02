"""Shared vault indexing and wikilink resolution for note graph tools."""

import re
from pathlib import Path

WIKILINK = re.compile(r"\[\[([^\[\]]+)\]\]")


def index_notes(vault: Path) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """Return Markdown notes indexed by vault-relative path and filename stem.

    ``vault`` must be the resolved vault root. Only regular ``.md`` files
    resolving inside it are included. The path index maps POSIX paths to files;
    the stem index retains every matching path so ambiguous names can be ignored
    during link resolution.

    Raises:
        OSError: If a candidate note's filesystem metadata cannot be accessed.
    """
    notes: dict[str, Path] = {}
    by_stem: dict[str, list[str]] = {}
    for note in vault.rglob("*.md"):
        if not note.is_file() or not note.resolve().is_relative_to(vault):
            continue
        relative = note.relative_to(vault).as_posix()
        notes[relative] = note
        by_stem.setdefault(note.stem, []).append(relative)
    return notes, by_stem


def resolve_links(
    content: str, notes: dict[str, Path], by_stem: dict[str, list[str]]
) -> set[str]:
    """Return unique vault-relative Markdown paths linked from ``content``.

    ``notes`` and ``by_stem`` are the indexes returned by ``index_notes``.
    Wikilinks, aliases, heading links, and embedded-note links are resolved
    against them. Qualified targets use vault-relative paths; bare note names
    must identify a unique note. Explicit ``.md`` targets preserve dotted stems.
    Empty, absolute, escaping, missing, ambiguous, and non-Markdown targets are
    ignored. The returned set is unordered.
    """
    links: set[str] = set()
    for match in WIKILINK.finditer(content):
        target = match.group(1).split("|", 1)[0].split("#", 1)[0].strip()
        if not target:
            continue

        target_path = Path(target)
        if target_path.is_absolute() or ".." in target_path.parts:
            continue
        # Check the actual extension before removing it: a Markdown note's
        # stem can itself contain dots, as in "Release 1.2.md".
        if target_path.suffix and target_path.suffix != ".md":
            continue

        if "/" in target:
            candidate = target if target_path.suffix else f"{target}.md"
            if candidate in notes:
                links.add(candidate)
        else:
            stem = target_path.stem if target_path.suffix else target
            matches = by_stem.get(stem, [])
            if len(matches) == 1:
                links.add(matches[0])

    return links
