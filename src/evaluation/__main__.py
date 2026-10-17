"""Configure and execute a whole-dataset evaluation from the command line."""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.evaluation.config import EvaluationConfig
from src.evaluation.judge import SemanticJudge
from src.evaluation.runner import (
    DEFAULT_VAULT_PATH,
    run_evaluation,
    validate_evaluation_inputs,
)
from src.gold_dataset import _unique_object
from src.harness import SYSTEM_PROMPT


def main(argv: list[str] | None = None) -> int:
    """Return 0 for all passes, 1 for scored failures, or 2 for setup/errors.

    JSON configuration independently selects candidate and judge providers,
    constructor settings, and binding options. Both system prompts are bundled.
    CLI overrides cover the entire dataset, vault, repeat count, invocation limit,
    and report directory; case selection and version filtering are absent.
    """
    parser = argparse.ArgumentParser(description="Evaluate every gold dataset case")
    parser.add_argument("--config", type=Path, help="independent model settings JSON")
    parser.add_argument("--dataset-path", type=Path, help="default: bundled dataset")
    parser.add_argument("--vault-path", type=Path, default=DEFAULT_VAULT_PATH)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-api-calls", type=int, default=20)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("evaluation_runs")
    )
    try:
        arguments = parser.parse_args(argv)
        raw_config = (
            json.loads(
                arguments.config.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_object,
            )
            if arguments.config
            else {}
        )
        config = EvaluationConfig.model_validate(raw_config)
        validate_evaluation_inputs(
            dataset_path=arguments.dataset_path,
            vault_path=arguments.vault_path,
            repeats=arguments.repeats,
            max_api_calls=arguments.max_api_calls,
            model_label=config.candidate.model,
        )
        load_dotenv()
        candidate = config.candidate.create_model()
        judge = SemanticJudge(
            model=config.judge.create_model(),
            structured_output_kwargs=config.judge.structured_output_kwargs,
        )
        run = run_evaluation(
            candidate_model=candidate,
            system_prompt=SYSTEM_PROMPT,
            vault_path=arguments.vault_path,
            judge=judge,
            dataset_path=arguments.dataset_path,
            repeats=arguments.repeats,
            max_api_calls=arguments.max_api_calls,
            model_label=config.candidate.model,
            tool_binding_kwargs=config.candidate.tool_binding_kwargs,
            report_destination=arguments.output_directory,
            candidate_config=config.candidate.model_dump(),
            judge_config=config.judge.model_dump(),
        )
    except SystemExit as exc:
        return 0 if exc.code == 0 else 2
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Evaluation setup or persistence failed: {exc}", file=sys.stderr)
        return 2
    print(f"Report: {run.run_directory / 'report.json'}")
    print(f"Summary: {run.run_directory / 'summary.md'}")
    return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
