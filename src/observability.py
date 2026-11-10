"""Export optional Langfuse telemetry without changing application outcomes.

Credentials and the base URL come from the environment, loaded by CLI callers.
Programmatic callers own environment loading and should call flush before exit.
"""

import json
import logging
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any, Literal

from langfuse import Langfuse, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)
_client: Langfuse | None = None
_warned = False


def client() -> Langfuse | None:
    """Return the lazy client, or None when disabled or unavailable.

    Missing credentials disable export. Incomplete configuration and SDK setup
    failures emit warnings without exposing credentials or interrupting work.
    """
    global _client, _warned
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "false":
        return None
    keys = [os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")]
    if not any(keys):
        return None
    if not all(keys) or not os.getenv("LANGFUSE_BASE_URL"):
        if not _warned:
            logger.warning("Langfuse requires both keys and LANGFUSE_BASE_URL")
            _warned = True
        return None
    if _client is None:
        try:
            _client = get_client()
        except Exception:
            logger.warning("Langfuse initialization failed")
    return _client


class Observation:
    """Provide best-effort updates to an optional SDK observation."""

    def __init__(self, span: Any = None) -> None:
        self.span = span

    def update(self, **fields: Any) -> None:
        """Record explicitly selected output or metadata; tolerate SDK failures."""
        if self.span is not None:
            try:
                self.span.update(**fields)
            except Exception:
                logger.warning("Langfuse observation update failed")


    def record_tool_output(self, output: str) -> None:
        """Retain exact tool text and mark recognized JSON error envelopes.

        Plain text and other JSON values remain valid tool outputs. Telemetry
        inspection never imposes a new output contract on the dispatcher.
        """
        self.update(output=output)
        try:
            envelope = json.loads(s=output)
        except json.JSONDecodeError:
            return
        if isinstance(envelope, dict) and envelope.get("ok") is False:
            self.update(level="ERROR", status_message="Tool returned an error")


@contextmanager
def observation(
    name: str,
    kind: Literal["span", "agent", "tool", "evaluator"] = "span",
    input: Any = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> Iterator[Observation]:
    """Scope an observation and inherited metadata, preserving application errors.

    Capture only supplied fields. Exceptions are marked by type to avoid leaking
    filesystem paths or credentials through exception messages and tracebacks.
    """
    stack = ExitStack()
    record = Observation()
    sdk = client()
    if sdk is not None:
        try:
            stack.enter_context(
                propagate_attributes(
                    session_id=session_id,
                    metadata=metadata,
                )
            )
            record = Observation(
                span=stack.enter_context(
                    sdk.start_as_current_observation(
                        name=name,
                        as_type=kind,
                        input=input,
                        metadata=metadata,
                    )
                )
            )
        except Exception:
            logger.warning("Langfuse observation setup failed")
    try:
        yield record
    except BaseException as exc:
        record.update(level="ERROR", status_message=type(exc).__name__)
        raise
    finally:
        try:
            stack.close()
        except Exception:
            logger.warning("Langfuse observation cleanup failed")


def observability_options(name: str) -> dict[str, Any]:
    """Return invocation options with a fresh callback, or no options if disabled."""
    if client() is not None:
        try:
            return {"config": {"callbacks": [CallbackHandler()], "run_name": name}}
        except Exception:
            logger.warning("Langfuse callback setup failed")
    return {}


def flush() -> None:
    """Flush an initialized client without masking application errors."""
    if _client is not None:
        try:
            _client.flush()
        except Exception:
            logger.warning("Langfuse flush failed")
