"""Run vault harness examples with OpenAI, Claude, or Kimi chat models."""

import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.harness import run_in_harness
from src.tracing import Trace

VAULT_PATH = str(Path(__file__).resolve().parents[1] / "tests/test_vault")
EXAMPLE_PROMPT = (
    "Help me locate the file called 'Agentic Software Engineering Factory'."
)


def _execute(model: BaseChatModel, **kwargs: Any) -> None:
    """Print the bundled vault answer and trace for a configured chat model.

    Keyword arguments are passed to the harness. The invocation trace is printed
    even if the harness fails.

    Raises:
        Exception: If model invocation or the harness fails.
    """
    trace = Trace()
    try:
        answer = run_in_harness(
            model,
            user_prompt=EXAMPLE_PROMPT,
            vault_path=VAULT_PATH,
            trace=trace,
            **kwargs,
        )
        print("Final output:")
        print(answer)
    finally:
        print()
        print(trace)


def execute_openai() -> None:
    """Run the bundled vault query with OpenAI and strict tool schemas.

    Environment configuration is loaded from dotenv before model creation.

    Raises:
        Exception: If model configuration, invocation, or the harness fails.
    """
    load_dotenv()
    _execute(
        ChatOpenAI(model="gpt-6-luna"),
        model_label="gpt-6-luna",
        tool_binding_kwargs={"strict": True},
    )


def execute_claude() -> None:
    """Run the bundled vault query with Claude Haiku and a 4096 token budget.

    Environment configuration is loaded from dotenv before model creation.

    Raises:
        Exception: If model configuration, invocation, or the harness fails.
    """
    load_dotenv()
    model = ChatAnthropic(
        model_name="claude-haiku-5-5",
        max_tokens_to_sample=4096,
        timeout=None,
        stop=None,
    )
    _execute(model, model_label="claude-haiku-5-5")


def execute_kimi() -> None:
    """Run the bundled vault query with Kimi through Moonshot's endpoint.

    KIMI_API_KEY is read from the environment or dotenv before model creation.

    Raises:
        KeyError: If KIMI_API_KEY is missing.
        Exception: If model configuration, invocation, or the harness fails.
    """
    load_dotenv()
    model = ChatOpenAI(
        model="kimi-k2.6",
        api_key=SecretStr(os.environ["KIMI_API_KEY"]),
        base_url="https://api.moonshot.ai/v1",
        use_responses_api=False,
        temperature=0.6,
        max_completion_tokens=4096,
        extra_body={"thinking": {"type": "disabled"}},
    )
    _execute(model, model_label="kimi-k2.6")


def main() -> None:
    """Run the selected example, defaulting to OpenAI.

    The optional command-line argument is openai, claude, or kimi.

    Raises:
        Exception: If the selected example fails.
    """
    parser = argparse.ArgumentParser(description="Run a vault harness example")
    parser.add_argument(
        "model", nargs="?", choices=("openai", "claude", "kimi"), default="openai"
    )
    selection = parser.parse_args().model
    {"openai": execute_openai, "claude": execute_claude, "kimi": execute_kimi}[
        selection
    ]()


if __name__ == "__main__":
    main()
