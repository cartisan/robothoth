# Gold dataset

`dataset.json` is the single manually maintained source of evaluation cases.
The bundled file's `schema_version` is `1`. `GoldDataset` accepts any integer
schema version and preserves it as metadata. Every version uses the same entry
validation rules defined by the Pydantic models in `validate.py`; the version
does not select or filter cases. Noninteger values and type coercion are rejected.

Each entry has a unique, nonblank `id`, a `request`, an `added_reason`, a
positive integer `dataset_version`, `ground_truth`, and required `tool_checks`
object. Ground truth may contain
an `ideal_answer`, an `answer_criteria` array, or both. A supplied answer and
every supplied criterion must be nonblank; at least one usable expectation is
required. When both are present, the ideal answer is a reference and **all**
criteria apply. Unknown fields and type coercion are rejected.

Tool requirements are deterministic and belong in `tool_checks`, rather than
semantic criteria:

| Field | Default | Requirement |
| --- | --- | --- |
| `required_tools` | `[]` | At least one successful call per tool name |
| `required_note_reads` | `[]` | Successful `read_note` for each full vault-relative Markdown path |
| `forbidden_tools` | `[]` | No attempted call to these names |
| `allowed_tools` | `null` | If supplied, every attempted tool must be in this list |

The object may be empty. Lists require unique nonblank strings. Required tools
must be allowed and cannot be forbidden; required note reads implicitly require
`read_note`. Paths reject absolute forms and parent traversal, normalize slash
styles and `.` components lexically, and remain case-sensitive. Checks ignore
call order and frequency. Unexecuted or failed calls do not satisfy requirements,
but still count for prohibitions and allowlists.

To add cases, append objects to the existing `entries` array in `dataset.json`.
Give each a new ID, set its introduction `dataset_version`, explain why the case
was added in `added_reason`, and supply its expectations. Name the ID after the
evaluation purpose or assistant behavior rather than the retrieved notes' subject
matter; prefer three lowercase hyphen-delimited parts, such as
`multi-source-integration`. Multiple cases may
share a version. The reason should identify the evaluation coverage and the
assistant capability, behavior, or failure mode being tested; it should not
summarize the source notes or repeat the expected answer. Run validation before
committing:

```sh
uv run --locked python -m src.gold_dataset.validate
uv run --locked python -m src.gold_dataset.validate path/to/dataset.json
```

In Python, import dataset models and loading functions directly from
`src.gold_dataset.validate`. `load_dataset(path=None, version=None)` returns all validated entries
in file order. Passing version `N` selects entries introduced at or before `N`;
version `0` selects none. The entire file is checked before filtering. 
Git preserves historical edits to cases; filtering does not restore earlier expectations.
The evaluation runner always uses every validated entry and offers no version
filter or individual case selection. Its gold dataset version is the highest
entry `dataset_version`.

## Bundled coverage

The bundled dataset has 14 entries: three introduced in version 1
and eleven introduced in version 2. The schema version remains 1. Every version 2
entry supplies both an ideal answer and semantic criteria. The five hard cases
require three or more specific note reads plus multiple tools. Tool checks do
not constrain call order; the two graph-only cases explicitly request their
single tool and use an allowlist.

| Version 2 case | Required tools | Focus |
| --- | --- | --- |
| `multi-source-integration` | search, read | Combine two testing notes; avoid unrelated errors passing a test |
| `content-based-discovery` | search, read | Fixed seeds versus feature ordering and sampling |
| `constraint-based-recommendation` | list, read | Apply robust-loss and Hessian requirements |
| `outgoing-link-discovery` | outgoing links only | Resolve existing cross-directory course links |
| `incoming-link-discovery` | backlinks only | Complete incoming links, including a Unicode filename |
| `insufficient-evidence-abstention` | list, read | Abstain when overview notes contain no comparative evidence |
| `contextual-guidance-synthesis` **(hard)** | search, outgoing links, read | Four-note monitoring design with delayed labels |
| `outdated-guidance-detection` **(hard)** | search, outgoing links, read | Serialization, migration, and deprecated instructions |
| `reference-based-calculation` **(hard)** | list, outgoing links, read | Three-note derivation with correlated dimensions |
| `multipart-concept-application` **(hard)** | search, outgoing links, read | Grounding versus truth, dataset design, and repeated trials |
| `source-caveat-preservation` **(hard)** | search, backlinks, read | Discover a MOC and synthesize a caveated knowledge workflow |

These cases use the [extended knowledge snapshot](../../tests/vault_fixtures.md),
which is also the runner's default vault. Required reads refer to full relative
paths in that snapshot. The small fixture alone cannot answer the whole suite.
All five registered tools are required at least once among the new cases;
there is no additional listing-only case.

The references are grounded in the sanitized notes, including their limitations.
Numeric criteria distinguish distances from squared distances and best-of-k
success from success on every trial.
Semantic criteria evaluate answers; successful evidence gathering is checked
separately using the execution trace. A scored candidate failure is useful benchmark
evidence and does not by itself justify weakening the reference answer.
