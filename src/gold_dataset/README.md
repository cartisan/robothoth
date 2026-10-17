# Gold dataset

`dataset.json` is the single manually maintained source of evaluation cases.
The bundled file's `schema_version` is `1`. `GoldDataset` accepts any integer
schema version and preserves it as metadata. Every version uses the same entry
validation rules defined by the Pydantic models in `__init__.py`; the version
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
was added in `added_reason`, and supply its expectations. Multiple cases may
share a version. Run validation before committing:

```sh
uv run --locked python -m src.gold_dataset
uv run --locked python -m src.gold_dataset path/to/dataset.json
```

In Python, `load_dataset(path=None, version=None)` returns all validated entries
in file order. Passing version `N` selects entries introduced at or before `N`;
version `0` selects none. The entire file is checked before filtering. 
Git preserves historical edits to cases; filtering does not restore earlier expectations.
The evaluation runner always uses every validated entry and offers no version
filter or individual case selection. Its gold dataset version is the highest
entry `dataset_version`. Keep schema and existing entry versions at 1 for the
tool-check migration.
