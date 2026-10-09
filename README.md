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

Pass a configured, tool-capable LangChain `BaseChatModel` directly:

```python
from langchain_openai import ChatOpenAI
from src.harness import run_in_harness
from src.tracing import Trace

trace = Trace()
answer = run_in_harness(
    ChatOpenAI(model="gpt-6-luna"),
    user_prompt="Find notes about agentic software engineering",
    vault_path="/path/to/vault",
    trace=trace,
    tool_binding_kwargs={"strict": True},
)
print(answer)
print(trace)
```

The harness binds registered tool schemas at the start of each run and keeps
that run's messages local. It executes tool calls, returns exact dispatcher
results as `ToolMessage` values, enforces the invocation limit, and traces
model responses and usage. Tool requests and completed results remain in the
trace if a later step fails. Missing token usage remains unknown. Pass
`model_label` when you want a particular fallback label for failed calls;
otherwise the chat model's class name is used.
