# Docstrings

Follow the behavioral docstring pattern in `src/tools/get_backlinks.py`: start
with a concise description of the result, then explain relevant inputs,
resolution or validation rules, and edge cases. Add a `Raises:` section for
exceptions callers must handle. For functions that return error envelopes,
describe the returned error codes instead. Describe observable behavior rather
than narrating implementation steps, and keep docstrings consistent with the code.

# Function signatures

Prefer positional-or-keyword parameters over keyword-only `*` parameters unless
the interface has a concrete need to require keyword arguments.

# Validation and defensive coding

Validate untrusted data at its boundary, then trust validated models, annotated
internal interfaces, and invariants established by the caller. Do not dump and
revalidate model instances, repeat normalization, check types already guaranteed
by parsing or schemas, or supply fallbacks for required internal fields. Let
programming errors surface instead of silently treating them as missing data.
Keep checks for meaningful value constraints, external responses, filesystem
boundaries, and operational failures. Copy mutable data only when mutation or
snapshot ownership requires it.

# Function and method calls

Always use explicit keyword arguments for all function and method calls.

# Package layout and execution

Keep `__init__.py` files docstring-only. Define implementation in descriptive
modules and import symbols directly from their defining modules rather than
re-exporting them through package initializers. Put executable entrypoints in
named modules with an `if __name__ == "__main__"` guard, invoked as
`python -m package.module`. Do not add package-level `__main__.py` entrypoints.

# System prompt versions

When changing a bundled `SystemPrompt`'s text, increment its positive integer
version, normally by one. Versions are independent for each prompt .

# Markdown documentation

When editing Markdown documentation, replace outdated facts with the current
facts directly. Do not narrate that something was changed or updated.

# Gold dataset entries

Name each case `id` after its evaluation purpose or the assistant behavior being
tested, rather than the subject matter of the retrieved notes. Prefer three
lowercase parts delimited by hyphens, such as `multi-source-integration`.

Write each `added_reason` as an evaluation rationale: why the case belongs in
the dataset and which assistant capability, behavior, or failure mode it tests.
Describe the intended coverage rather than summarizing the source notes or
repeating facts from the expected answer. Put answer expectations in
`ground_truth` and deterministic tool requirements in `tool_checks`.
