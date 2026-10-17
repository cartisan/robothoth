"""Independent candidate and judge model settings for batch evaluations."""

from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class ModelConfig(BaseModel):
    """Describe a provider model and its provider-specific constructor settings."""

    model_config = ConfigDict(extra="forbid", strict=True)
    provider: Literal["openai", "anthropic"]
    model: StrictStr
    settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model")
    @classmethod
    def nonblank_model(cls, value: str) -> str:
        """Accept a nonblank provider model identifier."""
        if not value.strip():
            raise ValueError("model must not be blank")
        return value

    @field_validator("settings")
    @classmethod
    def reject_model_overrides(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Keep the configured model identifier unambiguous."""
        if {"model", "model_name"}.intersection(value):
            raise ValueError("set the model identifier in model, not settings")
        return value

    def create_model(self) -> BaseChatModel:
        """Return the configured LangChain model without invoking an API.

        Credentials normally come from provider environment variables. Settings
        are forwarded to the provider constructor; OpenAI defaults to Responses.

        Raises:
            Exception: If provider settings or credentials cannot configure the model.
        """
        settings = self.settings.copy()
        if self.provider == "openai":
            settings = {"use_responses_api": True, **settings}
            return ChatOpenAI(model=self.model, **settings)
        return ChatAnthropic(model_name=self.model, **settings)


class CandidateConfig(ModelConfig):
    """Add tool-binding options independently of the candidate's API settings."""

    tool_binding_kwargs: dict[str, Any] = Field(default_factory=dict)


class JudgeConfig(ModelConfig):
    """Add independent structured output options for the bundled judge prompt."""

    structured_output_kwargs: dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """Configure candidate and judge separately, with usable provider defaults."""

    model_config = ConfigDict(extra="forbid", strict=True)
    candidate: CandidateConfig = Field(
        default_factory=lambda: CandidateConfig(
            provider="openai",
            model="gpt-6-luna",
            settings={"reasoning_effort": "medium", "use_responses_api": True},
            tool_binding_kwargs={"strict": True},
        )
    )
    judge: JudgeConfig = Field(
        default_factory=lambda: JudgeConfig(
            provider="openai",
            model="gpt-6.1-sol",
            settings={"reasoning_effort": "medium", "use_responses_api": True},
        )
    )
