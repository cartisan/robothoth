# Gold dataset

`dataset.json` is the single manually maintained source of evaluation cases.
Its `schema_version` is `1`. The Pydantic models in `__init__.py` define and
validate the format.

Each entry has a unique, nonblank `id`, a `request`, an `added_reason`, a
positive integer `dataset_version`, and `ground_truth`. Ground truth may contain
an `ideal_answer`, an `answer_criteria` array, or both. A supplied answer and
every supplied criterion must be nonblank; at least one usable expectation is
required. When both are present, the ideal answer is a reference and **all**
criteria apply. Unknown fields and type coercion are rejected.

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
Git preserves historical edits to cases;  filtering does not restore earlier expectations.