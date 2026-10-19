"""Persist independent repeat results and pass-rate summaries atomically."""

import json
import os
import re
from dataclasses import asdict, is_dataclass
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, SecretBytes, SecretStr


def json_value(value: Any) -> Any:
    """Return JSON-compatible evidence while preserving message and trace fields.

    Pydantic objects and dataclasses retain their named fields. Unusual response
    objects fall back to text; secret wrappers are always redacted.
    """
    if isinstance(value, (SecretStr, SecretBytes)):
        return "[REDACTED]"
    if isinstance(value, BaseModel):
        return json_value(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def without_credentials(value: Any) -> Any:
    """Return configuration with credential and authentication fields excluded.

    Nested provider settings and secret wrappers are filtered too. Token budgets
    and usage settings remain because they are not credentials.
    """
    if isinstance(value, (SecretStr, SecretBytes)):
        return None
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            name = str(key).lower().replace("-", "_")
            if (
                any(
                    part in name
                    for part in (
                        "api_key",
                        "apikey",
                        "authorization",
                        "password",
                        "credential",
                        "secret",
                        "access_token",
                        "refresh_token",
                        "headers",
                        "cookie",
                    )
                )
                or name in {"key", "token", "auth"}
                or name.endswith("_token")
            ):
                continue
            if isinstance(item, (SecretStr, SecretBytes)):
                continue
            cleaned[str(key)] = without_credentials(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [without_credentials(item) for item in value]
    return json_value(value)


def model_configuration(model: BaseChatModel) -> dict[str, Any]:
    """Return noncredential model settings for a Python-configured candidate or judge.

    Client objects are omitted. LangChain's serializable model fields retain
    provider options such as reasoning, temperature, endpoint, and token limits.
    """
    fields = model.model_dump(
        exclude={
            "client",
            "async_client",
            "root_client",
            "root_async_client",
            "http_client",
            "http_async_client",
            "callbacks",
            "rate_limiter",
        }
    )
    return {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "settings": without_credentials(fields),
    }


def _sum_tokens(records: list[dict[str, Any]], field: str) -> int | None:
    """Return summed token counts, or None if any count is unknown.

    Empty records cost zero. Token breakdowns are never added to their totals.
    """
    total = 0
    for record in records:
        value = record[field]
        if value is None:
            return None
        total += value
    return total


def _judge_token_usage(calls: list[dict[str, Any]]) -> int | None:
    """Return attempted judge calls' total tokens, or None if any usage is unknown.

    Raw provider metadata must contain a nonnegative integer total; missing
    responses, absent usage, and malformed totals make the sum unknown. An empty
    sequence of attempted calls costs zero.
    """
    total = 0
    for call in calls:
        raw = call["raw_response"]
        usage = raw.get("usage_metadata") if isinstance(raw, dict) else None
        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        if type(tokens) is not int or tokens < 0:
            return None
        total += tokens
    return total


def repeat_token_usage(result: dict[str, Any]) -> dict[str, int | None]:
    """Return candidate, judge, and combined token costs for a completed repeat.

    Candidate cost includes every traced invocation, including failed attempts.
    Judge cost sums usage across all attempted calls, even when grading
    validation fails. A skipped judge costs zero. Missing or invalid provider
    usage makes that component and the combined total unknown, while known
    components remain available. Cost follows Trace's token unit, not currency.
    """
    candidate_tokens = _sum_tokens(
        records=result["trace"]["calls"], field="total_tokens"
    )
    judge_tokens = _judge_token_usage(calls=result["judge_calls"])
    total_tokens = (
        candidate_tokens + judge_tokens
        if candidate_tokens is not None and judge_tokens is not None
        else None
    )
    return {
        "candidate_tokens": candidate_tokens,
        "judge_tokens": judge_tokens,
        "total_tokens": total_tokens,
    }


def counts(results: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    """Return completion, error, and scored counts with passed/scored pass rate.

    Harness and judge errors are excluded from scored repeats. An empty scored
    denominator yields None, and unfinished repeats count as pending.
    Token costs include all completed repeats, including errors; any unknown
    cost makes its corresponding total unknown. An empty run costs zero.
    """
    passed = sum(item["status"] == "passed" for item in results)
    failed = sum(item["status"] == "failed" for item in results)
    harness_errors = sum(item["status"] == "harness_error" for item in results)
    judge_errors = sum(item["status"] == "judge_error" for item in results)
    scored = passed + failed
    return {
        "expected": expected,
        "completed": len(results),
        "pending": expected - len(results),
        "scored": scored,
        "passed": passed,
        "failed": failed,
        "harness_errors": harness_errors,
        "judge_errors": judge_errors,
        "errors": harness_errors + judge_errors,
        "pass_rate": passed / scored if scored else None,
        "candidate_tokens": _sum_tokens(records=results, field="candidate_tokens"),
        "judge_tokens": _sum_tokens(records=results, field="judge_tokens"),
        "total_tokens": _sum_tokens(records=results, field="total_tokens"),
    }


def refresh_summary(report: dict[str, Any]) -> None:
    """Update run and case pass rates without assigning aggregate case verdicts."""
    report["summary"] = counts(report["results"], report["expected_repeats"])
    report["case_summaries"] = [
        {
            "case_id": case_id,
            **counts(
                [item for item in report["results"] if item["case_id"] == case_id],
                report["repeats"],
            ),
        }
        for case_id in report["case_ids"]
    ]
    report["complete"] = (
        report["summary"]["pending"] == 0 and report["summary"]["errors"] == 0
    )


def _table_text(text: str) -> str:
    """Return table-safe text, preserving full explanations and line breaks.

    Pipes and backslashes are escaped, HTML is shown literally, and line breaks
    become HTML breaks so multiline evidence stays within its table cell.
    """
    escaped = re.sub(
        pattern=r"([\\|])",
        repl=r"\\\1",
        string=escape(s=text, quote=False),
    )
    return re.sub(pattern=r"\r\n?|\n", repl="<br>", string=escaped)


def _token_text(tokens: int | None) -> str:
    """Display token cost explicitly, retaining unknown provider usage."""
    return "unknown" if tokens is None else f"{tokens:,} tokens"


def _cost_text(metrics: dict[str, Any]) -> str:
    """Display combined cost with separate candidate and judge token costs."""
    return (
        f"{_token_text(tokens=metrics['total_tokens'])} "
        f"(candidate: {_token_text(tokens=metrics['candidate_tokens'])}; "
        f"judge: {_token_text(tokens=metrics['judge_tokens'])})"
    )


def _time_text(seconds: float) -> str:
    """Display elapsed wall time in seconds."""
    return f"{seconds:.3f} s"


def _repeat_markdown(result: dict[str, Any]) -> list[str]:
    """Return one repeat's status, trace checks, judge ratings, and available errors.

    Every tool check and semantic expectation includes its full explanation.
    Judge scores of at least 4 pass. Empty checks and unavailable grades are
    explicit, and errors include any chained original failure.
    Cost includes candidate and judge tokens; elapsed time includes tools and
    grading, rather than only summed model invocation latency.
    """
    case_id = _table_text(text=result["case_id"])
    lines = [
        f"### {case_id}, repeat {result['repeat']}: {result['status']}",
        "",
        f"Cost: {_cost_text(metrics=result)}",
        f"Time: {_time_text(seconds=result['elapsed_seconds'])}",
        "",
        "Trace validation:",
        "",
    ]
    if result["tool_checks"]:
        lines += [
            "| Check | Result | Explanation | Trace calls |",
            "| --- | --- | --- | --- |",
        ]
        for check in result["tool_checks"]:
            status = "passed" if check["passed"] else "failed"
            evidence = ""
            for reference in check["calls"]:
                if evidence:
                    evidence += "; "
                evidence += (
                    f"invocation {reference['invocation_index'] + 1}, "
                    f"tool {reference['tool_index'] + 1} ({reference['call_id']})"
                )
            lines += [
                f"| {_table_text(text=check['expectation_id'])} | {status} | "
                f"{_table_text(text=check['explanation'])} | "
                f"{_table_text(text=evidence) if evidence else 'None'} |"
            ]
    else:
        lines += ["No tool checks configured."]
    lines += ["", "Judge ratings (passing score: 4 or 5):", ""]
    if result["scores"]:
        lines += [
            "| Expectation | Score | Result | Explanation |",
            "| --- | --- | --- | --- |",
        ]
        for score in result["scores"]:
            status = "passed" if score["score"] >= 4 else "failed"
            lines += [
                f"| {_table_text(text=score['expectation_id'])} | "
                f"{score['score']} / 5 | {status} | "
                f"{_table_text(text=score['explanation'])} |"
            ]
    elif result["status"] == "harness_error":
        lines += ["Not graded because harness execution failed."]
    else:
        lines += ["No valid judge ratings available."]
    if error := result["error"]:
        lines += [
            "",
            f"Error: {_table_text(text=error['type'] + ': ' + error['message'])}",
        ]
        if cause := error["cause"]:
            lines += [
                "",
                "Caused by: "
                f"{_table_text(text=cause['type'] + ': ' + cause['message'])}",
            ]
    return lines + [""]


def summary_markdown(report: dict[str, Any]) -> str:
    """Render pass rates and detailed evidence for each independent repeat.

    Repeat sections include trace-check outcomes and call references, every judge
    score and full explanation, and available errors. Trace call numbers are
    one-based for display; recorded JSON indexes remain zero-based.
    Suite and repeat costs include candidate and judge token usage. Suite time
    includes overhead between repeats, so it can exceed summed repeat durations.
    """
    total = report["summary"]
    lines = [
        "# Evaluation report",
        "",
        f"Run: {report['run_id']}",
        f"Complete: {str(report['complete']).lower()}",
        f"Completed: {total['completed']} / {total['expected']}",
        f"Errors: {total['errors']} (harness: {total['harness_errors']}, "
        f"judge: {total['judge_errors']})",
        f"Pass rate: {_rate(summary=total)}",
        f"Total cost: {_cost_text(metrics=total)}",
        f"Total time: {_time_text(seconds=report['elapsed_seconds'])}",
        "",
        "| Case | Passed / scored | Pass rate | Completed | Errors |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in report["case_summaries"]:
        case_id = case["case_id"].replace("|", "\\|").replace("\n", " ")
        rate = "null" if case["pass_rate"] is None else f"{case['pass_rate']:.1%}"
        lines.append(
            f"| {case_id} | {case['passed']} / {case['scored']} | {rate} | "
            f"{case['completed']} / {case['expected']} | {case['errors']} |"
        )
    lines.extend(["", "## Repeats", ""])
    for result in report["results"]:
        lines += _repeat_markdown(result=result)
    return "\n".join(lines) + "\n"


def _rate(summary: dict[str, Any]) -> str:
    """Display a scored denominator and null when no repeat was scored."""
    rate = "null" if summary["pass_rate"] is None else f"{summary['pass_rate']:.1%}"
    return f"{summary['passed']} / {summary['scored']} ({rate})"


def _atomic_write(path: Path, content: str) -> None:
    """Replace one report file atomically with complete UTF-8 content.

    Raises:
        OSError: If the destination cannot be written or replaced.
    """
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_report(directory: Path, report: dict[str, Any]) -> None:
    """Atomically update report.json and summary.md from the same run snapshot.

    The report contains JSON-compatible fields produced by the runner.
    Each file replacement is atomic; readers may briefly see different revisions
    of the two files. JSON is the authoritative artifact.

    Raises:
        OSError: If report persistence fails.
    """
    refresh_summary(report)
    _atomic_write(
        directory / "report.json",
        json.dumps(obj=report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(directory / "summary.md", summary_markdown(report))
