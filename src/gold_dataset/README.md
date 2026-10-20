# Gold dataset

`dataset.json` is the manually maintained source of evaluation cases. Its
`schema_version` is `2`; this is metadata and does not select validation rules.
Unknown fields and type coercion are rejected.

Each entry requires a unique nonblank `id`, `request`, `added_reason`, positive
integer `dataset_version`, `answer_criteria`, and `tool_checks`. The
`answer_criteria` object requires an `ideal_answer`, nonempty
`additional_criteria`, or both. Supplied answers and criteria must be nonblank;
when both are present, all criteria apply.

```json
{
  "answer_criteria": {
    "ideal_answer": "The reference answer.",
    "additional_criteria": ["Preserves the source caveat."]
  }
}
```

Use ideal answers for correctness and essential coverage. Keep case-specific
criteria focused on critical exclusions and source caveats. Shared concision and
task-usefulness rubrics belong in the [judge](../evaluation/README.md#scoring).
Base criteria on the vault snapshot and its caveats, independently of candidate
performance.

## Tool checks

Deterministic tool requirements belong in `tool_checks`:

| Field | Default | Requirement |
| --- | --- | --- |
| `required_tools` | `[]` | At least one successful call per tool name |
| `required_note_reads` | `[]` | Successful `read_note` for each full vault-relative Markdown path |
| `forbidden_tools` | `[]` | No attempted call to these names |
| `allowed_tools` | `null` | If supplied, every attempted tool must be in this list |

The object may be empty. Lists require unique nonblank values; required tools
must be allowed and cannot be forbidden. Required note reads imply `read_note`.
Paths are case-sensitive and vault-relative, with separators and `.` normalized;
absolute paths and parent traversal are rejected.

Checks ignore call order and frequency. Failed or unexecuted calls do not satisfy
requirements, but count toward prohibitions and allowlists.

## Adding and loading cases

Append entries to `dataset.json` and set their introduction `dataset_version`.
Name IDs after the tested behavior, preferably three lowercase hyphen-delimited
parts such as `multi-source-integration`. Use `added_reason` to explain evaluation
coverage or failure modes, keeping answer criteria in `answer_criteria`.

```sh
uv run --locked python -m src.gold_dataset.validate
uv run --locked python -m src.gold_dataset.validate path/to/dataset.json
```

Import models and loaders from `src.gold_dataset.validate`.
`load_dataset(path=None, version=None)` validates the entire file, then returns
entries in file order. Version `N` includes entries introduced at or before `N`;
`None` includes all and `0` includes none. Git retains historical edits to
criteria. The runner always evaluates all cases and reports the highest
entry version as `gold_dataset_version`.

## Bundled coverage

The 14 cases include three introduced in version 1 and eleven in version 2.
Five hard cases require at least three specific note reads and multiple tools.
Together, the version 2 cases exercise all five tools; graph-only cases use
allowlists. All cases use the [extended vault snapshot](../../tests/vault_fixtures.md),
the runner's default. The small tool-test fixture cannot answer the whole suite.

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
