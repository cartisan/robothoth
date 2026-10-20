# Repository guidance

## Python conventions

- Prefer positional-or-keyword parameters; require keyword-only parameters only
  for a concrete interface need. Use explicit keywords in calls except for
  positional-only parameters and variadic positional arguments.
- Keep `__init__.py` docstring-only and import symbols directly from their
  defining modules. Put executable entrypoints in named modules with an
  `if __name__ == "__main__"` guard; invoke them with `python -m package.module`.
  Do not add package-level `__main__.py`.

## Data and validation

- Prefer named Pydantic models for records that need validation or a clear
  contract; keep trivial local data simple. Nest models when useful.
- Validate untrusted data at boundaries, then trust validated models and
  internal contracts. Retain checks for meaningful value constraints, external
  responses, filesystem boundaries, and operational failures. Avoid redundant
  validation or normalization and fallbacks that hide programming errors.
- Copy mutable data only for mutation isolation or snapshot ownership.
- Extract cohesive validation into documented, descriptively named helpers.
  Keep class-specific construction, validation and other helpers on their class.

## Documentation

- Follow `src/tools/get_backlinks.py`: describe the result, relevant inputs,
  resolution or validation rules, and edge cases. Document caller-relevant
  exceptions in `Raises:` or returned error codes for error envelopes.
  Describe observable behavior and keep docstrings consistent with the code.
- Comment on non-obvious model-field purposes, relationships, and transformations,
  in inline comments.
- State current facts directly in Markdown; omit change-history narration.

## Verification and evaluations

- For code changes, run affected tests with `uv run --locked python -m pytest -q`
  and suitable test paths; run `uv run --locked ruff check src tests` and
  `uv run --locked mypy src tests`.
- Use test-driven development when you imlement new functionality.
- Write unit tests to cover the happy path, edge cases and expected failure modes.
  Tests should cover the behavior of a method or class, not implementation details.
- When a test fails due to a code change, fix the code and not the test (unless
  the desired behavior of the system changed). Don't fix broken tests by deleting
  them.
- For behavior-preserving refactors, adapt existing tests rather than adding
  tests that mirror the implementation structure.
- When a bundled `SystemPrompt` text changes, increment its version independently,
  normally by one.
- Before editing gold cases, follow the authoring rules in
  [the dataset guide](src/gold_dataset/README.md#adding-and-loading-cases).
