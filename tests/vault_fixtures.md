# Vault fixtures

`test_vault` contains 14 notes for focused tool tests and the `src.main`
examples. `extended_test_vault` is the default evaluation corpus, with 152
sanitized Markdown notes and no images. Every small-fixture note has the same
content in the extended fixture.

## Content policy

The fixtures contain general published ML/math knowledge, cement knowledge,
reference material, and anonymized work practices derived from a private
Obsidian vault. They are independent of that source vault at runtime.

Company-specific algorithms, actual customer identities and data, experiment
reports, credential values, and private images are excluded. Private employer,
repository, service, personnel, account, host, and environment identities use
placeholders. Public tools, frameworks, and cited public authors retain their
names. Code examples use fictional data and self-contained demonstration classes.

| Source group | Notes | Content |
| --- | ---: | --- |
| Root | 1 | General index |
| Areas | 111 | Technical references, general ML/math, testing, engineering, and work practices |
| Resource root | 2 | Incident-review template and public product-demo notes |
| Archive root | 22 | Course/conference/seminar notes, MOCs, and anonymized work-practice notes |
| Cement references | 5 | General chemistry, measurement, grinding, and physical-property notes |
| Example Company operations | 6 | Anonymized environment and deployment practices |
| Service A work practices | 5 | Testing, debugging, artifact compatibility, and release coordination |

The [manifest](extended_test_vault_manifest.json) describes the current sanitized
file inventory, SHA-256 hashes, placeholder-redaction categories, and unresolved
reference targets. It contains no original private identities, removed-file
inventory, or absolute source-vault path. It lives outside the navigable corpus.

## Reference limitations

The corpus retains 47 unavailable attachment targets and 48 unresolved note
targets. These are references, not bundled images or invented content. Some
links use old generic paths or title characters different from actual filenames.
The manifest records each remaining target and its referring sanitized notes.
Every required gold read and graph route resolves to an existing note.

## Evaluation

The [gold dataset](../src/gold_dataset/README.md) has 14 cases, including eleven
introduced in version 2 and five hard multi-note cases. All five registered
tools are covered. The release-guidance case uses the anonymized Service A
references; numerical derivations use general mathematical definitions and
explicit fictional inputs.

Local validation does not require model API calls:

```sh
uv run --locked python -m src.gold_dataset.validate
uv run --locked python -m pytest -q
uv run --locked ruff check src tests
uv run --locked mypy src tests
```

Live evaluation is a separate, explicitly requested action. Generated API-run
reports remain private local artifacts under the ignored `evaluation_runs`
directory. Scores from a different corpus or dataset do not describe this suite.
