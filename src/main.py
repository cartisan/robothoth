"""Run the example vault query with the default OpenAI provider."""

from pathlib import Path

from dotenv import load_dotenv

from src.harness import run_in_harness
from src.providers.openai_provider import OpenAIProvider
from src.tracing import Trace

VAULT_HOME = str(Path(__file__).resolve().parents[1] / "tests/test_vault")


def main() -> None:
    """Print the example answer and trace, including metrics on run failure.

    Environment configuration is loaded from dotenv before provider creation.
    The provider closes its owned client even when the run fails.

    Raises:
        openai.OpenAIError: If client creation or an API invocation fails.
        RuntimeError: If the harness cannot produce a valid final answer.
    """
    load_dotenv()
    trace = Trace()
    with OpenAIProvider() as provider:
        try:
            output = run_in_harness(
                provider,
                user_prompt=(
                    "Help me locate the file called "
                    "'Agentic Software Engineering Factory'."
                ),
                vault_path=VAULT_HOME,
                trace=trace,
            )
            print("Final output:")
            print(output)
        finally:
            print()
            print(trace)


if __name__ == "__main__":
    main()
