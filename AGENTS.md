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

# System prompt versions

When changing a bundled `SystemPrompt`'s text, increment its positive integer
version, normally by one. Versions are independent for each prompt and maintained
manually; CI does not enforce version increases.
