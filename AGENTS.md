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
