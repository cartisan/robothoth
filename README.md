# Robothoth

A synchronous tool harness for navigating an Obsidian vault with a LangChain
chat model.

Install and run an example:

```sh
uv sync --locked --dev
uv run --locked python -m src.main openai
uv run --locked python -m src.main claude
uv run --locked python -m src.main kimi
```

The examples load `.env`, search the bundled test vault, and print the answer
and invocation trace. OpenAI uses `OPENAI_API_KEY` and `gpt-6-luna`; Claude uses
`ANTHROPIC_API_KEY` and `claude-haiku-5-5`. The Kimi example uses `KIMI_API_KEY`
and `kimi-k2.6` through Moonshot's OpenAI-compatible endpoint. Running
`src.main` without an argument defaults to OpenAI. The examples are also
available as `execute_openai()`, `execute_claude()`, and `execute_kimi()` in
`src.main`.

The manually maintained [gold dataset](src/gold_dataset/README.md) contains
validated evaluation cases. Run `uv run --locked python -m src.golden_dataset.validate` to
validate the bundled file and print its case count.

Pass a configured, tool-capable LangChain `BaseChatModel` directly:

```python
from langchain_openai import ChatOpenAI
from src.harness import run_in_harness

result = run_in_harness(
    ChatOpenAI(model="gpt-6-luna"),
    user_prompt="Find notes about agentic software engineering",
    vault_path="/path/to/vault",
    tool_binding_kwargs={"strict": True},
)
print(result.final_answer)
print(result.trace)
```

The harness binds registered tool schemas at the start of each run and keeps
that run's messages local. It executes tool calls, returns exact dispatcher
results as `ToolMessage` values, enforces the invocation limit, and traces
model responses and usage. Every invocation owns a fresh trace and conversation.
Pass a `SystemPrompt(version=2, text="...")` from `src.utils` as `system_prompt`
to evaluate different candidate instructions through the Python API. Operational
failures raise `HarnessError` with the partial trace in `error.trace` and the
original exception chained as `error.__cause__`. Invalid arguments raise
`ValueError` before execution. Missing token usage remains unknown. Pass
`model_label` when you want a particular fallback label for failed calls;
otherwise the chat model's class name is used.

Evaluate all gold cases with an independently configured judge:

```sh
uv run --locked python -m src.evaluation.run --output-directory evaluation_runs
uv run --locked python -m src.evaluation.run --config evaluation.json \
  --dataset-path src/golden_dataset/dataset.json --vault-path tests/test_vault \
  --repeats 3 --max-api-calls 20 \
  --output-directory evaluation_runs
```

The default candidate is `gpt-6-luna` at medium reasoning; the default judge is
`gpt-6.1-sol` at medium reasoning with the Responses API and native structured
output. These judge settings follow the [official model documentation](https://developers.openai.com/api/docs/models/gpt-6.1-sol).
Both use environment credentials, loaded from `.env` when run through the CLI.
The CLI always uses the bundled candidate and judge system prompts.
See [evaluation configuration and reports](src/evaluation/README.md) for the
Python API, provider settings, scoring rules, and report format.
