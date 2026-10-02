from pathlib import Path

from src.tools.note_links import index_notes, resolve_links
from src.tools.read_note import read_note


def get_outgoing_links(vault_path: str, notepath: str) -> list[str]:
    """Return sorted existing Markdown notes linked from ``notepath``.

    Wikilinks, aliases, heading links, and embedded-note links are resolved
    against the vault. Bare note names are followed only when they identify a
    unique note; duplicate links are returned once.

    Raises:
        FileNotFoundError: If the source note does not exist.
        ValueError: If ``notepath`` is not a vault-relative Markdown path or
            resolves outside the vault.
    """
    content = read_note(vault_path, notepath)
    vault = Path(vault_path).resolve()

    notes, by_stem = index_notes(vault)
    return sorted(resolve_links(content, notes, by_stem))
