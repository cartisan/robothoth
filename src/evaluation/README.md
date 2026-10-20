# Evaluation

The runner evaluates every [gold case](../gold_dataset/README.md) in file order,
with a fresh candidate conversation for each repeat. Defaults: the bundled
dataset, `tests/extended_test_vault`, one repeat, and 20 model invocations per
conversation. Invalid or empty datasets fail before model calls.

## Running evaluations

```sh
uv run --locked python -m src.evaluation.run --config evaluation.json \
  --dataset-path src/gold_dataset/dataset.json --vault-path tests/extended_test_vault \
  --repeats 3 --max-api-calls 20 --output-directory evaluation_runs
```

All flags are optional. The CLI loads credentials from `.env`.

```python
from langchain_openai import ChatOpenAI
from src.evaluation.runner import run_evaluation

run = run_evaluation(
    candidate_model=ChatOpenAI(
        model="gpt-6-luna", reasoning_effort="medium", use_responses_api=True,
    ),
    repeats=2,
    report_destination="evaluation_runs",
)
print(run.run_directory)
print(run.report["summary"])
```

Pass `system_prompt=SystemPrompt(version=2, text="...")` from `src.utils` to
customize candidate instructions. The judge uses the bundled `CONTENT_PROMPT`,
`CONCISION_PROMPT`, and `TASK_USEFULNESS_PROMPT`; their text and versions are saved
in reports.

`SemanticJudge(model=model).grade(request=request, final_answer=final_answer,
answer_criteria=answer_criteria)` also grades saved answers. Import `SemanticJudge` from
`src.evaluation.judge` and `AnswerCriteria` from `src.gold_dataset.validate`.
The result exposes `scores` and `judge_calls`. Each score uses `criterion_id`.
Stages are mutable and reused across answers; calls on one judge must be sequential.

## Model configuration

The default candidate is OpenAI `gpt-6-luna`; the judge is OpenAI `gpt-6.1-sol`.
Both use medium reasoning and the Responses API. The judge defaults to strict
native JSON-schema output.

An optional config file selects each model independently:

```json
{
  "candidate": {
    "provider": "openai",
    "model": "gpt-6-luna",
    "settings": {"reasoning_effort": "medium", "use_responses_api": true},
    "tool_binding_kwargs": {"strict": true}
  },
  "judge": {
    "provider": "anthropic",
    "model": "claude-haiku-5-5",
    "settings": {"max_tokens": 4096},
    "structured_output_kwargs": {"method": "json_schema"}
  }
}
```

Providers are `openai` and `anthropic`; `settings` forwards provider constructor
arguments. Keep the model identifier in `model`. Judge structured-output options
must retain `include_raw=True`; external tools, schema overrides, and explicit
conversation-continuation IDs are rejected.

## Scoring

Each grade makes three sequential calls:

| Stage | Supplied evidence | Criterion IDs |
| --- | --- | --- |
| Content | Request, final answer, ideal answer, case-specific criteria | `ideal_answer`, applicable `additional_criteria[i]` |
| Concision | Request and final answer only | `quality.concision` |
| Task usefulness | Request and final answer only | `quality.task_usefulness` |

Each call starts with a fresh system/user message pair. Quality calls see only
their own rubric and the request and answer, without reference material or other
stages' results. Judges have no trace or vault access; tool requirements are
checked separately and deterministically.

Content scoring requires essential meaning and all case-specific criteria while
accepting paraphrases. Concision rewards proportional detail without padding;
necessary explanations can earn full marks. Task usefulness measures an
appropriately scoped, formatted, and usable deliverable.

Scores are integers: 1 fails, 2 has major gaps, 3 partially meets the requirement,
4 meets it with minor issues, and 5 fully meets it. Missing or contradicted
essential content scores at most 3. Each criterion needs one score and a short
explanation; missing, duplicate, unknown, or malformed scores invalidate grading.

| Status | Meaning |
| --- | --- |
| `passed` | Every content and quality score is at least 4 and every tool check passes |
| `failed` | A valid grade or tool check fails a requirement |
| `harness_error` | Execution fails to produce a valid final answer |
| `judge_error` | Grading invocation or validation fails |

Low scores and failed tool checks still allow all grading calls. Technical grading
errors stop later stages and raise `JudgeError`, retaining the failed `stage` and
all attempted `judge_calls`. Errors return no partial grade; evaluation continues
to later repeats.

Pass rates are **passed / scored repeats**, excluding errors, or `null` if nothing
was scored. Errors make a run incomplete. CLI exit codes are 0 for all passes,
1 for complete runs with scored failures, and 2 for incomplete runs or
setup/persistence errors.

## Reports

Each run writes `report.json` and `summary.md` before execution and after every
repeat. Files are replaced atomically; JSON is authoritative.

Report format `4` includes answers, scores, explanations, tool evidence, traces,
credential-free configuration, and versioned prompt snapshots. `judge_prompt`
stores the content prompt; `quality_judge_prompts` stores the two quality prompts.
Scores and tool-check results identify each requirement with `criterion_id`.
Each repeat's `judge_calls` records `{stage, raw_response}` in call order, using
`null` when an attempted call produced no response.

JSON and Markdown summaries include repeat and case pass rates, errors, token
costs, and elapsed wall time. Cost means tokens, summed across candidate calls
and all attempted judge calls, including failed grading. If no judge calls are
attempted, judge cost is zero; unknown usage is `null` in JSON and `unknown` in
Markdown.
`elapsed_seconds` includes tool execution, grading, and suite overhead.

Dataset contents are not embedded in reports. Use the corresponding Git revision
when regrading historical answers against their original criteria.
