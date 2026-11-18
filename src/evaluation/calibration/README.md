# LLM judge calibration

Generate one independent answer from `gpt-6-luna`, `gpt-6.1-terra`, and
`gpt-6.1-sol` for each of the gold requests:

```sh
uv run --locked python -m src.evaluation.calibration.generate_dataset
```

The generator loads `.env`, disables Langfuse tracing, and uses medium reasoning,
the existing harness system prompt, strict tool binding, the extended test vault, and at most 20 model calls
per answer. Defaults are constants in `generate_dataset.py`. Every invocation
creates a fresh dataset identity. After all candidates finish and the new dataset
is staged in the output directory, the existing `dataset.json` is renamed to
`dataset_<old_dataset_id>.json`, preserving its exact bytes and human ratings.
The staged dataset is then published atomically as `dataset.json`. The first run
creates only the current file. Custom output names use the same pattern:
`answers.json` is archived as `answers_<old_dataset_id>.json`.

The current file remains available throughout generation. Invalid existing
datasets, unsafe archive IDs (path separators, control or reserved filename
characters, or the identities `.` and `..`), archive collisions, and staging failures
leave existing files untouched. Rename and publish failures propagate; a failed
publish restores the previous output from its archive. Temporary files are
cleaned up. Concurrent generation or editing of the same output is unsupported.
Ordinary rating saves update the loaded file without archiving it. Generation
errors are retained per candidate and do not prevent saving; they are reported
on stdout with exit code 1. No judges run during generation.

Launch the local rating app from the repository root:

```sh
uv run --locked python -m streamlit run src/evaluation/calibration/app.py \
  --browser.gatherUsageStats=false
```

The app disables Langfuse tracing on every execution.

Enter a local JSON path and click **Load dataset**. The **Human ratings** tab
shows each request, an expandable ideal answer, and candidates in model order.
Content, concision, and usefulness accept integers 1–5; clearing a control leaves
that rubric unrated (`null`). **Save dataset** atomically saves to the loaded
path. Loading a dataset resets rating widgets, prompts, both judge results and
statistics flags, and hides validation again.

In **Judge calibration**, questions are split into disjoint **Test set** and
**Validation set** views. Sorted question IDs are shuffled with a local random
generator using fixed seed `42`; the first half is test. Display order remains
the original question and candidate order. Each question's candidates stay
together, including generation failures. The dataset must contain an even number
of at least two questions and identical model composition per question (including
multiplicities); otherwise the tab shows an error and prevents judging. The
bundled dataset gives seven questions and 21 answers per side, seven answers per
model. Views are recomputed from the full dataset so rating edits remain visible;
only the full dataset is saved by the human-ratings tab.

Initially only the **Test set** is displayed. Edit the three bundled prompt texts
and click **Judge Candidates** to grade test answers. **Compute statistics** and
**Old results → Load results** also use only test inputs. Each full bundled split
run schedules 63 rubric calls. Click **Validate on held out data**, after the test
cases, to grade validation with the current prompt editors. This button requires
three nonblank prompts and a valid split; a prior test run is unnecessary. Before
the first click, the calibration tab renders no validation cases, references,
ratings, results or statistics. The human-ratings tab always shows all questions.

Validation reveals a separate section with its own results and automatically
computed statistics. Each click creates a fresh run, so validation is repeatable.
Prompt editing remains available afterward: edits mark differing result snapshots
stale and retain previously revealed validation without making judge calls. Both
runs have independent state. Setup or persistence failures clear the requested
split's active result; earlier saved checkpoints remain available for explicit
loading.

The shared production judge defaults to `gpt-6.1-terra`
with high reasoning, Responses and strict structured output. Calls run across
all content candidates first, then all concision candidates, then all usefulness
candidates. Each call has fresh system/user messages with only the request,
answer and rubric ID; content also includes the ideal answer. Human scores,
model identity, other scores and conversation history are excluded. Failed
generations are skipped. Individual judge failures retain their evidence and
allow later calls to continue.

Each click creates a separate JSON run in `src/evaluation/calibration/runs/`,
checkpointed before execution and after every call. Runs contain exact prompt
text, credential-filtered model and binding configuration, the production
`JudgeResponseSchema`, an input snapshot and SHA-256 fingerprint, scores,
explanations, and raw call/error evidence. Split runs also record their role,
seed and the full source-input fingerprint. Result loaders offer only compatible
runs and restore their prompts. Validation result loading is available only after
the validation button has been clicked. Loading requires matching split role,
seed, full source fingerprint and exact subset inputs, including order and IDs;
human-rating changes are allowed. Changing held-out inputs also invalidates test
runs. Partial checkpoints can be loaded. Legacy unsplit runs cannot serve as split
results. Editing a restored prompt marks differing results as an earlier snapshot.

**Compute statistics** compares current human ratings with successful judge
scores for each rubric. All metrics use the same pairs, omitting missing ratings
and unsuccessful judge calls. Pair counts are displayed with each rubric.

| Statistic | Interpretation | Unavailable when |
| --- | --- | --- |
| Spearman’s rho | Rank correlation, with average ranks for ties | Fewer than two pairs or either side constant |
| Pearson’s r | Linear correlation | Fewer than two pairs or either side constant |
| Linear weighted Cohen’s κ | Chance-corrected agreement, penalizing absolute rating differences | No pairs or zero expected disagreement |
| Quadratic weighted Cohen’s κ | Chance-corrected agreement, penalizing squared rating differences | No pairs or zero expected disagreement |
| Mean absolute error | Mean absolute difference in rating points | No pairs |
| Mean signed error | Mean **judge − human** difference in rating points; positive means the judge scores higher | No pairs |

Kappa uses the full ordinal 1–5 scale, including unobserved categories, and
independent empirical human/judge marginals for chance agreement. Identical
constant ratings have undefined kappa; different constant ratings have kappa 0.
Errors remain available with one pair. Statistics update from current human
ratings after computation without rerunning the judge.

Definitions: [Cohen’s kappa](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html)
and [Pearson’s r](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html).

Dataset JSON uses strict version `1`. Run JSON requires version `3` with
optional split context; `run_judges()` and `load_run()` callers may omit context
for unsplit usage. Versions `1` and `2` are rejected without migration or legacy
ID translation. Score `rubric` values match their rubric: `content`,
`concision`, and `task_usefulness`. Unknown
fields, coercion, blank inputs, duplicate identities and unresolved model
references are rejected. Candidate configurations occur once in generation metadata; each
candidate stores an identity/model reference, answer or error, and human ratings.
The initial dataset has no human scores. Initial live judging is triggered only
through the app.
