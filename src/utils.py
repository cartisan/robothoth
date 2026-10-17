"""Utility methods for all modules"""

from pathlib import PurePosixPath, PureWindowsPath

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator


class SystemPrompt(BaseModel):
    """Identify immutable system instructions by text and a positive integer version.

    Text is preserved exactly and must contain a non-whitespace character.
    Unknown fields and type coercion are rejected. Versions are independent for
    each prompt and maintained manually; no comparison with prior revisions occurs.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    version: StrictInt = Field(gt=0)
    text: StrictStr

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        """Return the original text when it contains a non-whitespace character.

        Raises:
            ValueError: If the supplied prompt text is blank.
        """
        if not value.strip():
            raise ValueError("prompt text must not be blank")
        return value


def normalize_note_path(path: str) -> str:
    """Return a case-preserving path with separators and dot components normalized.

    Both slash styles are accepted. Absolute paths, drive-qualified paths,
    parent traversal, and empty paths are rejected without reading the vault.

    Raises:
        ValueError: If path is not a nonempty vault-relative file path.
    """
    if not isinstance(path, str) or not path.strip() or "\x00" in path:
        raise ValueError("note path must be nonempty vault-relative text")
    portable = path.replace("\\", "/")
    parts = PurePosixPath(portable)
    if (
        parts.is_absolute()
        or PureWindowsPath(path).drive
        or ".." in parts.parts
        or str(parts) == "."
    ):
        raise ValueError("note path must be relative without parent traversal")
    return str(parts)
