from pathlib import Path


def list_notes(vault_path: Path, path: str | None = None) -> list[str]:
    """List Markdown note paths relative to a vault, optionally within a directory."""
    vault = vault_path.resolve()
    if not vault.is_dir():
        raise FileNotFoundError(vault_path)

    relative_dir = Path(path) if path is not None else Path(".")
    if relative_dir.is_absolute() or ".." in relative_dir.parts:
        raise ValueError(f"Directory must be within the vault: {path}")

    directory = (vault / relative_dir).resolve()
    if not directory.is_relative_to(vault):
        raise ValueError(f"Directory must be within the vault: {path}")
    if not directory.is_dir():
        raise FileNotFoundError(directory)

    return sorted(
        note.relative_to(vault).as_posix()
        for note in directory.rglob("*.md")
        if note.is_file() and note.resolve().is_relative_to(vault)
    )
