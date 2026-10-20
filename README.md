# Robothoth

A synchronous LangChain tool harness for navigating an Obsidian vault.

```sh
uv sync --locked --dev
uv run --locked python -m src.main openai
uv run --locked python -m src.main claude
uv run --locked python -m src.main kimi
```

Examples load `.env`, search the bundled test vault, and print the answer and
trace. Omitting the provider defaults to OpenAI.

| Provider | Model | Credential |
| --- | --- | --- |
| `openai` | `gpt-6-luna` | `OPENAI_API_KEY` |
| `claude` | `claude-haiku-5-5` | `ANTHROPIC_API_KEY` |
| `kimi` | `kimi-k2.6` via Moonshot | `KIMI_API_KEY` |

## Python API

Pass a configured, tool-capable `BaseChatModel`:

```python
from langchain_openai import ChatOpenAI
from src.harness import run_in_harness

result = run_in_harness(
    chat_model=ChatOpenAI(model="gpt-6-luna"),
    user_prompt="Find notes about agentic software engineering",
    vault_path="/path/to/vault",
    tool_binding_kwargs={"strict": True},
)
print(result.final_answer)
print(result.trace)
```

Each run has a fresh conversation and trace, with tool calls and token usage
recorded. Invalid inputs raise `ValueError`; operational failures raise
`HarnessError` with a partial `trace` and chained cause. Pass a versioned
`SystemPrompt` from `src.utils` as `system_prompt` to customize instructions.

## Evaluation

```sh
uv run --locked python -m src.evaluation.run --output-directory evaluation_runs
uv run --locked python -m src.evaluation.run --config evaluation.json
```

The [gold dataset](src/gold_dataset/README.md) contains 14 cases, including five
hard cases. Evaluations use `tests/extended_test_vault` with 152 sanitized notes;
examples and focused tool tests use the 14-note `tests/test_vault`.
See [vault provenance and boundaries](tests/vault_fixtures.md).

The default candidate is `gpt-6-luna` and the judge is `gpt-6.1-sol`, both at
medium reasoning. Content, concision, and task usefulness are graded in separate
calls with fresh histories. Passing requires every score to reach 4/5 and every
tool check to pass. See [evaluation configuration and reports](src/evaluation/README.md).
