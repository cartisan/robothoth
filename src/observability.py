"""Export optional Langfuse telemetry without changing application outcomes.

Credentials and the base URL come from the environment, loaded by CLI callers.
Programmatic callers own environment loading and should call flush before exit.
"""

import json
import logging
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from threading import Lock
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from langfuse import Langfuse, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from opentelemetry import context, trace
from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider

logger = logging.getLogger(__name__)
_client: Langfuse | None = None
_warned = False
_client_lock = Lock()
_EXPERIMENT_CONTEXT = context.create_key(keyname="robothoth-experiment")


class ExperimentSpanProcessor(SpanProcessor):
    """Copy context-scoped experiment identity to root and descendant spans.

    The first span supplies the item root ID. Each item owns its context mapping;
    no experiment attributes are added outside that scope. Propagation failures
    warn without interrupting span creation or evaluation.
    """

    def on_start(
        self, span: Span, parent_context: context.Context | None = None
    ) -> None:
        try:
            attributes = context.get_value(
                key=_EXPERIMENT_CONTEXT, context=parent_context
            )
            if isinstance(attributes, dict):
                attributes.setdefault(
                    "langfuse.experiment.item.root_observation_id",
                    trace.format_span_id(span_id=span.get_span_context().span_id),
                )
                span.set_attributes(attributes=attributes)
        except Exception:
            logger.warning("Langfuse experiment propagation failed")


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
    with _client_lock:
        if _client is None:
            try:
                _client = get_client()
            except Exception:
                logger.warning("Langfuse initialization failed")
            else:
                try:
                    provider = trace.get_tracer_provider()
                    if not isinstance(provider, TracerProvider):
                        raise RuntimeError("No global SDK tracer provider")
                    provider.add_span_processor(span_processor=ExperimentSpanProcessor())
                except Exception:
                    logger.warning("Langfuse experiment propagation setup failed")
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

    def score(
        self,
        name: str,
        value: float | str,
        data_type: Literal["NUMERIC", "BOOLEAN", "CATEGORICAL"],
        comment: str | None = None,
    ) -> None:
        """Attach a validated score to this observation; warn on SDK failures.

        Boolean scores use numeric 0/1 values with BOOLEAN data type. A disabled
        observation does nothing, and failed exports never trigger retries here.
        """
        if self.span is not None:
            try:
                self.span.score(
                    name=name, value=value, data_type=data_type, comment=comment
                )
            except Exception:
                logger.warning("Langfuse observation score export failed")

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
    tags: list[str] | None = None,
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
                    tags=tags,
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


class Experiment:
    """Report an existing local evaluation run.

    Run and case/repeat identifiers are deterministic. Metadata must already be
    credential-filtered. No hosted dataset is created.
    Caller-owned flushing and local result persistence remain independent.
    """

    def __init__(
        self,
        run_id: str,
        candidate_model: str,
        reasoning_level: str,
        metadata: dict[str, Any],
    ) -> None:
        self.name = f"evaluation-{candidate_model}-{reasoning_level}-{run_id}"
        # Display names can change without changing the identity of an existing run.
        self.id = str(
            uuid5(namespace=NAMESPACE_URL, name=f"robothoth-evaluation-{run_id}")
        )
        self.metadata = {
            **metadata,
            "candidate_model": candidate_model,
            "candidate_reasoning_effort": reasoning_level,
        }
        self.tags = [
            f"candidate-model:{candidate_model}",
            f"candidate-reasoning:{reasoning_level}",
        ]

    @contextmanager
    def item(
        self,
        case_id: str,
        repeat: int,
        input: str,
        metadata: dict[str, Any],
    ) -> Iterator[Observation]:
        """Yield an independent item root, restoring caller context on every exit.

        Case/repeat IDs are stable across runs. Local dataset version belongs in
        metadata, never the hosted-dataset item version attribute. Setup and
        cleanup failures warn; exceptions from the evaluated work still escape.
        """
        if client() is None:
            yield Observation()
            return
        token = None
        try:
            attributes = {
                "langfuse.experiment.id": self.id,
                "langfuse.experiment.name": self.name,
                "langfuse.experiment.dataset.id": "robothoth-gold",
                "langfuse.experiment.item.id": str(
                    uuid5(
                        namespace=NAMESPACE_URL,
                        name=json.dumps(obj=[case_id, repeat]),
                    )
                ),
                **{
                    f"langfuse.experiment.metadata.{key}": (
                        value if isinstance(value, str) else json.dumps(value)
                    )
                    for key, value in self.metadata.items()
                },
            }
            token = context.attach(
                context=context.set_value(
                    key=_EXPERIMENT_CONTEXT,
                    value=attributes,
                    context=context.Context(),
                )
            )
        except Exception:
            logger.warning("Langfuse experiment item setup failed")
        try:
            with observation(
                name="runner-run-evaluation",
                kind="evaluator",
                input=input,
                session_id=f"evaluation-run-{self.name}",
                tags=self.tags,
                metadata={**self.metadata, **metadata},
            ) as record:
                current = trace.get_current_span()
                if isinstance(current, Span) and (
                    current.attributes is None
                    or current.attributes.get("langfuse.experiment.id") != self.id
                ):
                    logger.warning(
                        "Langfuse experiment item propagation unavailable"
                    )
                yield record
        finally:
            if token is not None:
                try:
                    context.detach(token=token)
                except Exception:
                    logger.warning("Langfuse experiment item cleanup failed")

    def score(self, name: str, value: float) -> None:
        """Attach a report-derived numeric run score; warn on export failure."""
        sdk = client()
        if sdk is not None:
            try:
                sdk.create_score(
                    dataset_run_id=self.id, name=name, value=value, data_type="NUMERIC"
                )
            except Exception:
                logger.warning("Langfuse experiment score export failed")
