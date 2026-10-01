import re
from pathlib import Path


def get_backlinks(vault_path: str, notepath: str) -> list[str]:
    """Return sorted Markdown paths that link to the requested note.

    Wikilinks, aliases, heading links, and embedded-note links are resolved
    against the vault. Bare note names must identify a unique note; invalid,
    missing, ambiguous, and non-Markdown targets are ignored.

    Raises:
        FileNotFoundError: If the requested note does not exist.
        ValueError: If ``notepath`` is not a vault-relative Markdown path or
            resolves outside the vault.
    """
    relative_path = Path(notepath)
    if (
        not notepath
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.suffix != ".md"
    ):
        raise ValueError("notepath must be a vault-relative Markdown file path")

    vault = Path(vault_path).resolve()
    requested = (vault / relative_path).resolve()
    if not requested.is_relative_to(vault):
        raise ValueError("notepath must remain inside the vault")
    if not requested.is_file():
        raise FileNotFoundError(requested)

    notes: dict[str, Path] = {}
    stems: dict[str, list[str]] = {}
    for candidate in vault.rglob("*.md"):
        resolved = candidate.resolve()
        if not resolved.is_relative_to(vault) or not resolved.is_file():
            continue
        path = candidate.relative_to(vault).as_posix()
        notes[path] = candidate
        stems.setdefault(candidate.stem, []).append(path)

    backlinks: set[str] = set()
    for source_path, source in notes.items():
        content = source.read_text(encoding="utf-8")
        for match in re.finditer(r"\[\[([^\[\]]+)\]\]", content):
            target = match.group(1).split("|", 1)[0].split("#", 1)[0].strip()
            if not target:
                continue
            if "/" in target:
                qualified = target if target.endswith(".md") else f"{target}.md"
                if Path(qualified).is_absolute() or ".." in Path(qualified).parts:
                    continue
                destination = qualified if qualified in notes else None
            else:
                stem = target[:-3] if target.endswith(".md") else target
                if Path(stem).suffix:
                    continue
                matches = stems.get(stem, [])
                destination = matches[0] if len(matches) == 1 else None
            if destination is not None and notes[destination].resolve() == requested:
                backlinks.add(source_path)
                break

    return sorted(backlinks)
