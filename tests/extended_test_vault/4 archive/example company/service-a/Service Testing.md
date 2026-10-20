## Test observable behavior
Prefer assertions that demonstrate the intended behavior over unexplained magic
numbers. Keep a test focused on the behavior it claims to cover instead of
depending on unrelated post-processing steps.

## Fixtures
Extract repeated base data into a fixture. Individual tests should make only the
variations they need, copying mutable data when the test takes ownership of a
modified version. This keeps shared setup changes from requiring many edits.

## Test data
Use small, understandable examples and document how their expected results are
derived. Simplify unrelated settings when testing one behavior, without assuming
that production inputs share those simplified properties.

---
Associations: [[pytest]], [[Parametrize tests with error context or nullcontext]]
