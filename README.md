# Robothoth

A synchronous tool harness for navigating an Obsidian vault through OpenAI,
Claude, or a tool-capable LangChain chat model.

Install the project and run any provider example:

```sh
uv sync --locked --dev
uv run --locked python -m src.providers.openai_provider
uv run --locked python -m src.providers.claude_provider
uv run --locked python -m src.providers.langchain_provider
```

Each example loads `.env`, searches the bundled test vault, and prints its answer
and invocation trace. OpenAI uses `OPENAI_API_KEY` and `gpt-6-luna`; Claude uses
`ANTHROPIC_API_KEY` and `claude-haiku-5-5`. The LangChain example uses `KIMI_API_KEY`
to call `kimi-k2.6` at `https://api.moonshot.ai/v1`, in instant mode through
`ChatOpenAI`. Sample vault paths are resolved relative to the repository.
`uv run --locked python -m src.main` remains an OpenAI example entrypoint.
Configure the vault path and prompt in your own caller for other queries.

The harness accepts a provider rather than an SDK client:

```python
from src.harness import run_in_harness
from src.providers.openai_provider import OpenAIProvider
from src.tracing import Trace

trace = Trace()
with OpenAIProvider(model="gpt-6-luna") as provider:
    try:
        answer = run_in_harness(
            provider,
            user_prompt="Find notes about agentic software engineering",
            vault_path="/path/to/vault",
            trace=trace,
            max_api_calls=20,
        )
        print(answer)
    finally:
        print(trace)
```

For Claude, use `ClaudeProvider(model=..., max_tokens=2048)` from
`src.providers.claude_provider` with your selected model identifier and `ANTHROPIC_API_KEY`.
The native providers accept an injected SDK client as their first argument.
They close clients they create when used as context managers; injected clients
remain caller-owned.

For LangChain, pass your configured tool-capable `BaseChatModel`:

```python
from src.providers.langchain_provider import LangChainProvider
from src.harness import run_in_harness
from src.tracing import Trace

# chat_model is a configured ChatOpenAI, ChatAnthropic, or other chat model.
provider = LangChainProvider(chat_model, model_label="your-configured-model")
answer = run_in_harness(
    provider,
    user_prompt="List the notes in this vault",
    vault_path="/path/to/vault",
    trace=Trace(),
)
```

`langchain-core` and `langchain-openai` are installed with the project. For other
integrations, install the corresponding package, such as `langchain-anthropic`,
and configure its model, credentials, and API options.
The adapter binds the vault tool schemas and retains native messages. The
harness executes the tools; supply a chat model rather than a LangChain agent.

The harness owns tracing, tool dispatch, call limits, and final completion.
Providers own API calls, wire formats, model settings, and native conversation
history. A fresh session is created for every run. Traces retain tool requests
and completed results on failure. Latency includes provider invocation and
response normalization but excludes tool execution; cost is measured in tokens.
Missing usage stays unknown.

Custom providers implement `LLMProvider` and `ProviderSession` from
`src.providers.llm_provider`. They return `ModelResponse` and accept ordered `ToolResult`
batches. Invalid responses should return `validation_error` so the harness can
record usage before raising; invocation exceptions propagate unchanged.

The previous `run(client, ...)` interface is replaced by
`src.harness.run_in_harness(provider, ...)`. Model selection now belongs to the
provider, and the example entrypoint is `src.main`.
