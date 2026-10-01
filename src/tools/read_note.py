from pathlib import Path


def read_note(vault_path: Path, notepath: str) -> str:
    relative_path = Path(notepath)
    if (
        not notepath
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.suffix != ".md"
    ):
        raise ValueError("notepath must be a vault-relative Markdown file path")

    vault = vault_path.resolve()
    note = (vault / relative_path).resolve()
    if not note.is_relative_to(vault):
        raise ValueError("notepath must remain inside the vault")

    return note.read_text(encoding="utf-8")
