"""Provider-neutral requests, responses, and conversation contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ToolDefinition:
    """Describe a callable tool using a provider-independent JSON schema."""

    name: str
    description: str
    input_schema: dict[str, object]


@dataclass(frozen=True)
class ToolCall:
    """Retain a tool request and its arguments as JSON, including malformed JSON.

    Providers returning argument objects serialize them; providers returning
    strings retain those strings unchanged.
    """

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolResult:
    """Pair a requested call ID with the exact dispatcher output string."""

    call_id: str
    output: str


@dataclass(frozen=True)
class ModelResponse:
    """Expose text, ordered tool calls, and metrics without SDK response objects.

    Missing usage remains None. Input counts include cache reads and writes;
    breakdowns are not additional tokens. A validation error describes a response
    that must be traced but must not trigger tool execution or final completion.
    """

    model: str
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    response_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    validation_error: str | None = None


class ProviderSession(Protocol):
    """Own one run's native history without executing tools or recording traces.

    Each invocation makes one logical model call; SDK retries may happen inside
    it. Initial request data must contain plain snapshot-compatible values.
    """

    @property
    def model(self) -> str:
        """Return the requested model label for failed-call traces."""
        ...

    @property
    def initial_request(self) -> dict[str, object]:
        """Return the initial invocation arguments as plain data."""
        ...

    @property
    def prompt_messages(self) -> tuple[tuple[str, str], ...]:
        """Return initial prompt roles and text for provider-independent display."""
        ...

    def invoke(self) -> ModelResponse:
        """Return normalized output while retaining native continuation history.

        Invalid API responses carry validation_error instead of raising, so
        their usage and tool requests can be traced before rejection.

        Raises:
            Exception: If the model invocation fails; original provider exceptions
                propagate unchanged.
        """
        ...

    def add_tool_results(self, results: Sequence[ToolResult]) -> None:
        """Append the complete ordered result batch for the preceding response.

        Raises:
            ValueError: If result IDs do not match the pending tool calls in order.
        """
        ...


class LLMProvider(Protocol):
    """Create independent conversation sessions using registered tool schemas."""

    def start_session(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: Sequence[ToolDefinition],
    ) -> ProviderSession:
        """Return a fresh conversation with no history shared with earlier runs.

        Raises:
            Exception: If provider configuration or tool binding fails.
        """
        ...


def validate_tool_results(
    calls: Sequence[ToolCall], results: Sequence[ToolResult]
) -> None:
    """Accept exactly one result per pending call, in response order.

    Raises:
        ValueError: If result IDs differ from pending IDs or the batch is empty.
    """
    if not calls or [call.call_id for call in calls] != [r.call_id for r in results]:
        raise ValueError("Tool results must match pending calls in order")
