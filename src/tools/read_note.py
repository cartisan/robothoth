from pathlib import Path


def read_note(vault_path: str, notepath: str) -> str:
    """Read a UTF-8 Markdown note addressed by its vault-relative path.

    Symlinks and path components that resolve outside the vault are rejected,
    so callers can use the result without granting access to other files.

    Args:
        notepath: Vault-relative path of the Markdown note to read.

    Raises:
        FileNotFoundError: If the requested note does not exist.
        ValueError: If ``notepath`` is not a relative Markdown path inside the
            vault.
        UnicodeDecodeError: If the note is not valid UTF-8.
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
    note = (vault / relative_path).resolve()
    if not note.is_relative_to(vault):
        raise ValueError("notepath must remain inside the vault")

    return note.read_text(encoding="utf-8")
