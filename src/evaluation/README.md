# Evaluation

The runner evaluates the entire validated gold dataset sequentially in file
order, with independent conversations and grades for each repeat. Empty or
invalid datasets are rejected before any model calls. Defaults are the bundled
dataset and vault, the existing harness prompt, one repeat, and 20 model
invocations per conversation.

In Python:

```python
from langchain_openai import ChatOpenAI
from src.evaluation import SemanticJudge, run_evaluation
from src.harness import SYSTEM_PROMPT

run = run_evaluation(
    candidate_model=ChatOpenAI(
        model="gpt-6-luna", reasoning_effort="medium", use_responses_api=True,
    ),
    system_prompt=SYSTEM_PROMPT,
    vault_path="tests/test_vault",
    judge=SemanticJudge(),
    repeats=2,
    tool_binding_kwargs={"strict": True},
    report_destination="evaluation_runs",
)
print(run.run_directory)
print(run.report["summary"])
```

`SemanticJudge(model, prompt, structured_output_kwargs)` can also grade saved
answers independently via `grade(request, final_answer, ground_truth)`, with
`GroundTruth` from `src.gold_dataset`. Both the harness and judge accept the shared
immutable `SystemPrompt` model from `src.utils`, for example
`SystemPrompt(version=2, text="Custom system instructions")`. It requires a
strict positive integer version and nonblank text, rejects extra fields, and
preserves text exactly. Bundled prompts start at version `1`; increment the
respective version whenever its text changes. Versions are maintained manually,
without CI enforcement.

Custom prompts are available through the Python APIs: pass `system_prompt` to
the harness or runner, and `prompt` to `SemanticJudge`. The judge receives only the request,
answer, ideal answer, and semantic criteria, with stable expectation IDs. It
receives no trace, candidate system prompt, model identity, or vault access.

The ideal answer is one expectation allowing paraphrases while requiring
agreement on essential meaning; each criterion is another expectation. The
version 1 prompt anchors scores at 1 (fails), 2 (major gaps), 3 (partial),
4 (meets with minor issues), and 5 (fully meets). Missing or materially
contradicted essential content requires a score of at most 3. Candidate text
is evidence, never grading instructions. Scores must be strict integers, with
one nonblank explanation per expected ID. Missing, duplicate, unknown, or
malformed grades raise `JudgeError` retaining `raw_response`.

`evaluate_tool_checks(tool_checks, trace)` independently returns each tool
requirement's boolean result, explanation, and references using zero-based
`invocation_index` and `tool_index` plus the original `call_id`. It performs no
model calls or filesystem reads. A successful dispatcher envelope has exactly
`ok: true` and `result` containing the registered tool's string or string-list
result. Earlier failures do not negate a later successful call.

CLI configuration is strict JSON with independent `candidate` and `judge`
objects. Provider choices are `openai` and `anthropic`; `settings` forwards
provider-specific LangChain constructor arguments. OpenAI defaults to Responses,
which can be overridden for compatible providers. Model identifiers belong in
`model`, rather than in `settings`.
Each `ModelConfig` creates its configured model with `create_model()`, for example
`EvaluationConfig().candidate.create_model()`.
The CLI always uses the bundled system prompts for both candidate and judge.

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

The default candidate uses OpenAI `gpt-6-luna` with `reasoning_effort: "medium"`
and `use_responses_api: true`.
The default judge uses OpenAI `gpt-6.1-sol`, `reasoning_effort: "medium"`,
`use_responses_api: true`, and `method: "json_schema"` with `strict: true`.
Structured output options can select a different provider-supported method;
`include_raw` must stay true. External tools and schema overrides are rejected.
Custom Python judge prompts should preserve the scoring anchors and semantic
input boundary; their full text and integer version are recorded for later review.

```sh
uv run --locked python -m src.evaluation --config evaluation.json \
  --dataset-path src/gold_dataset/dataset.json --vault-path tests/test_vault \
  --repeats 3 --max-api-calls 20 \
  --output-directory evaluation_runs
```

Every repeat has exactly one status:

| Status | Meaning |
| --- | --- |
| `passed` | All semantic scores are at least 4 and every tool check passes |
| `failed` | A valid grade or tool check fails a requirement |
| `harness_error` | Execution fails to produce a valid final answer |
| `judge_error` | Grading invocation or validation fails |

Valid answers are graded even when tool checks fail. Errors preserve available
artifacts and continue to later repeats. Errors make a run incomplete; scored
failures do not. Pass rates are **passed / scored repeats**, excluding errors,
and `null` when no repeat was scored. Case summaries report counts and pass
rates, without an overall case verdict.

Each run gets a unique directory containing `report.json` and `summary.md`.
An initial incomplete report is written before execution, and both files are
updated atomically after every repeat. Individual file replacement is atomic;
JSON is authoritative if a reader observes different revisions between files.
Under Repeats, the Markdown summary includes each tool check's pass/fail result,
explanation, and trace-call references, plus every judge score and its full
explanation. It also shows execution or grading errors and indicates when judge
ratings are unavailable. Trace invocation and tool numbers are one-based in the
summary; JSON indexes remain zero-based.

The summary also records total suite cost and time, and each repeat's cost and
time. Cost follows the harness convention of **tokens**, combining all candidate
invocations and the judge response, with a separate breakdown for each model.
Usage from failed grading validation still counts. A skipped judge costs zero;
missing usage is `null` in JSON and `unknown` in Markdown, including affected
totals. Errors remain part of the cost accounting even though pass rates exclude
them. Cached and reasoning tokens are already included in provider totals and
are not added again.

`elapsed_seconds` uses a monotonic wall clock and includes tool execution and
grading. Suite time starts at entry to `run_evaluation`, including dataset
validation, run setup, and report updates between repeats, so it can exceed
the sum of repeat times. The
final snapshot measures time through the end of evaluation, before its own file
write. JSON retains candidate, judge, and total token counts per repeat and in
the count summaries, plus elapsed seconds for the suite and each repeat.

Report format version `2` stores `candidate_system_prompt` and `judge_prompt` as
`{version, text}` objects, with integer versions. This replaces the candidate
prompt string and judge string version in report format version `1`.
The report records identity, timestamps, case IDs and entry versions, dataset
path, schema version, highest entry version as `gold_dataset_version`, both
prompt configurations, credential-free model and binding configurations, harness
settings, answers, traces, scores, tool evidence, raw judge responses, and errors.
It does not copy the dataset or compute hashes. CLI exit codes are 0 for complete
all-pass runs, 1 for complete runs with scored failures, and 2 for incomplete
runs or setup/persistence errors.

Saved answers and versioned grading prompts are independent of harness execution.
Future calibration can reload the recorded dataset revision and regrade saved
answers, then match expectation IDs against SME grades. Git must supply the
corresponding historical dataset when expectations have changed. Calibration is
not implemented.
