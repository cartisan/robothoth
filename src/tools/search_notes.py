import re
from pathlib import Path


def search_notes(vault_path: Path, query: str) -> list[str]:
    """Find vault-relative Markdown note paths whose content matches a regex."""
    if not query:
        raise ValueError("Search query must not be empty")

    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error as error:
        raise ValueError(f"Invalid search regex: {error}") from error

    vault_root = vault_path.resolve()
    if not vault_root.is_dir():
        raise FileNotFoundError(vault_path)

    matches: list[str] = []
    for note in vault_root.rglob("*.md"):
        if not note.is_file() or not note.resolve().is_relative_to(vault_root):
            continue
        if pattern.search(note.read_text(encoding="utf-8")):
            matches.append(note.relative_to(vault_root).as_posix())

    return sorted(set(matches))
