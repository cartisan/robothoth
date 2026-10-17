"""Evaluate tool requirements using recorded evidence alone."""

import json
from dataclasses import dataclass

from src.gold_dataset.validate import ToolChecks, _unique_object
from src.tracing import ToolCallTrace, Trace
from src.utils import normalize_note_path


@dataclass(frozen=True)
class CallReference:
    """Identify a tool attempt by zero-based invocation and tool indexes."""

    invocation_index: int
    tool_index: int
    call_id: str


@dataclass(frozen=True)
class ToolCheckResult:
    """Retain one requirement's result, explanation, and relevant attempts."""

    expectation_id: str
    passed: bool
    explanation: str
    calls: tuple[CallReference, ...]


def _successful(call: ToolCallTrace) -> bool:
    """Return whether a recorded dispatcher envelope reports success.

    Unexecuted calls have no output and do not count as success.
    """
    return call.output is not None and json.loads(s=call.output)["ok"]


def _read_path(call: ToolCallTrace) -> str | None:
    """Return a lexically valid read_note target, or None for invalid arguments."""
    if call.name != "read_note":
        return None
    try:
        arguments = json.loads(s=call.arguments, object_pairs_hook=_unique_object)
        if not isinstance(arguments, dict):
            return None
        path = arguments.get("notepath")
        return normalize_note_path(path=path) if isinstance(path, str) else None
    except ValueError:
        return None


def evaluate_tool_checks(
    tool_checks: ToolChecks, trace: Trace
) -> tuple[ToolCheckResult, ...]:
    """Return each deterministic requirement's result and trace references.

    All attempts are examined without sequence or frequency constraints. A
    required tool or full case-sensitive note path needs at least one successful
    dispatcher envelope; prior failures do not negate later success. Forbidden
    and allowlist checks count attempts even if no output was produced. Only
    path separators and dot components are normalized; no files or models are
    accessed. Empty requirements return an empty result tuple.
    """
    attempts = [
        (CallReference(i, j, call.call_id), call)
        for i, invocation in enumerate(trace.calls)
        for j, call in enumerate(invocation.tool_calls)
    ]
    results: list[ToolCheckResult] = []
    for index, name in enumerate(tool_checks.required_tools):
        matching = [(ref, call) for ref, call in attempts if call.name == name]
        passed = any(_successful(call) for _, call in matching)
        results.append(
            ToolCheckResult(
                f"required_tools[{index}]",
                passed,
                f"{'Found' if passed else 'Missing'} successful {name} call.",
                tuple(ref for ref, _ in matching),
            )
        )
    for index, expected in enumerate(tool_checks.required_note_reads):
        matching = [
            (ref, call) for ref, call in attempts if _read_path(call) == expected
        ]
        passed = any(_successful(call) for _, call in matching)
        results.append(
            ToolCheckResult(
                f"required_note_reads[{index}]",
                passed,
                f"{'Found' if passed else 'Missing'} successful read of {expected}.",
                tuple(ref for ref, _ in matching),
            )
        )
    for index, name in enumerate(tool_checks.forbidden_tools):
        matching_refs = tuple(ref for ref, call in attempts if call.name == name)
        results.append(
            ToolCheckResult(
                f"forbidden_tools[{index}]",
                not matching_refs,
                f"{name} {'was attempted' if matching_refs else 'was not attempted'}.",
                matching_refs,
            )
        )
    if tool_checks.allowed_tools is not None:
        violations = tuple(
            ref for ref, call in attempts if call.name not in tool_checks.allowed_tools
        )
        results.append(
            ToolCheckResult(
                "allowed_tools",
                not violations,
                "Attempts outside allowlist."
                if violations
                else "All attempts allowed.",
                violations,
            )
        )
    return tuple(results)
