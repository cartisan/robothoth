from pathlib import Path

from src.tools.note_links import index_notes, resolve_links


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

    notes, by_stem = index_notes(vault)

    backlinks: set[str] = set()
    for source_path, source in notes.items():
        content = source.read_text(encoding="utf-8")
        for destination in resolve_links(content, notes, by_stem):
            if notes[destination].resolve() == requested:
                backlinks.add(source_path)
                break

    return sorted(backlinks)
