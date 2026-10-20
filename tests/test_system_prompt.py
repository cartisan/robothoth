"""Validate shared prompt revisions independently of provider execution."""

import pytest
from pydantic import ValidationError

from src.evaluation.judge import CONTENT_PROMPT
from src.harness import SYSTEM_PROMPT
from src.utils import SystemPrompt


@pytest.mark.parametrize(argnames="version", argvalues=[0, -1, True, "1", 1.0, None])
def test_version_must_be_a_strict_positive_integer(version: object) -> None:
    with pytest.raises(expected_exception=ValidationError):
        SystemPrompt.model_validate(obj={"version": version, "text": "Instructions"})


@pytest.mark.parametrize(argnames="text", argvalues=["", " \n\t", None, 123])
def test_text_must_be_nonblank_text(text: object) -> None:
    with pytest.raises(expected_exception=ValidationError):
        SystemPrompt.model_validate(obj={"version": 1, "text": text})


def test_prompt_preserves_text_and_is_immutable() -> None:
    text = "  First line.\nSecond line.\n"
    prompt = SystemPrompt(version=3, text=text)
    assert prompt.model_dump() == {"version": 3, "text": text}
    with pytest.raises(expected_exception=ValidationError, match="frozen"):
        prompt.text = "Changed"  # type: ignore[misc]
    with pytest.raises(expected_exception=ValidationError, match="frozen"):
        prompt.version = 4  # type: ignore[misc]
    with pytest.raises(expected_exception=ValidationError):
        SystemPrompt.model_validate(obj={"version": 1, "text": text, "extra": "value"})


def test_bundled_prompts_share_the_model() -> None:
    assert isinstance(SYSTEM_PROMPT, SystemPrompt)
    assert isinstance(CONTENT_PROMPT, SystemPrompt)
    assert SYSTEM_PROMPT is not CONTENT_PROMPT
