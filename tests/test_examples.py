"""Verify each example model can bind the registered vault tools."""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel

import src.main as examples
from src.tools.registry import registry


@pytest.mark.parametrize(
    "method,key",
    [
        ("execute_openai", "OPENAI_API_KEY"),
        ("execute_claude", "ANTHROPIC_API_KEY"),
        ("execute_kimi", "KIMI_API_KEY"),
    ],
)
def test_example_binds_tools(
    method: str, key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bind each example's actual model configuration without invoking an API."""
    monkeypatch.setenv(key, "test-key")
    bound_models: list[BaseChatModel] = []

    def check_binding(model: BaseChatModel, **kwargs: Any) -> str:
        """Capture the model after its registry schemas bind successfully."""
        model.bind_tools(
            registry.tool_definitions(),
            **(kwargs.get("tool_binding_kwargs") or {}),
        )
        bound_models.append(model)
        return "Done"

    monkeypatch.setattr(examples, "run_in_harness", check_binding)
    getattr(examples, method)()
    assert len(bound_models) == 1
