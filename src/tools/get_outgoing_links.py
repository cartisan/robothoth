import re
from pathlib import Path

from src.tools.read_note import read_note

WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")


def get_outgoing_links(vault_path: str, notepath: str) -> list[str]:
    """Return existing Markdown notes linked from a vault-relative note."""
    content = read_note(vault_path, notepath)
    vault = Path(vault_path).resolve()

    notes: set[str] = set()
    by_stem: dict[str, list[str]] = {}
    for note in vault.rglob("*.md"):
        if not note.is_file() or not note.resolve().is_relative_to(vault):
            continue
        relative = note.relative_to(vault).as_posix()
        notes.add(relative)
        by_stem.setdefault(note.stem, []).append(relative)

    links: set[str] = set()
    for match in WIKILINK.finditer(content):
        target = match.group(1).split("|", 1)[0].split("#", 1)[0].strip()
        if not target:
            continue

        target_path = Path(target)
        if target_path.is_absolute() or ".." in target_path.parts:
            continue
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

    return sorted(links)
