"""Generate a fresh calibration dataset without invoking any judges."""

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from dotenv import load_dotenv

from src.evaluation.calibration.dataset import (
    CalibrationDataset,
    Candidate,
    ErrorEvidence,
    GenerationRun,
    Question,
    load_dataset,
)
from src.evaluation.config import CandidateConfig
from src.evaluation.reporting import json_value
from src.gold_dataset.validate import DEFAULT_DATASET_PATH, load_dataset_document
from src.harness import SYSTEM_PROMPT, run_in_harness
from src.observability import flush

MODELS = ("gpt-6-luna", "gpt-6.1-terra", "gpt-6.1-sol")
REASONING_EFFORT = "medium"
DATASET_PATH = DEFAULT_DATASET_PATH
VAULT_PATH = Path(__file__).resolve().parents[3] / "tests/extended_test_vault"
OUTPUT_PATH = Path(__file__).with_name("dataset.json")
MAX_API_CALLS = 20
logger = logging.getLogger(__name__)


def _archive_path(output_path: Path) -> Path:
    """Resolve an unused sibling archive from the existing dataset's identity.

    Reject path separators, control characters, reserved filename characters,
    and the identities `.` and `..` so IDs cannot escape the output directory.
    Existing archive entries, including dangling symlinks, are never overwritten.

    Raises:
        ValueError: If the existing dataset or its archive identity is invalid.
        FileExistsError: If the archive destination already exists.
        OSError: If the existing dataset cannot be read.
    """
    previous = load_dataset(path=output_path)
    identity = previous.dataset_id
    if identity in {".", ".."} or re.search(r'[/\\<>:"|?*\x00-\x1f\x7f]', identity):
        raise ValueError("dataset_id contains unsafe archive filename characters")
    archive = output_path.with_name(
        f"{output_path.stem}_{identity}{output_path.suffix}"
    )
    if archive.exists() or archive.is_symlink():
        raise FileExistsError(f"archive already exists: {archive}")
    return archive


def _save_generated_dataset(dataset: CalibrationDataset, output_path: Path) -> None:
    """Stage a complete dataset, archive the old bytes, and atomically publish.

    The current output stays untouched until staging succeeds. After an archive
    rename, a failed publish restores the archive to the output path. Temporary
    files are removed on success and failure. Concurrent edits are unsupported.

    Raises:
        ValueError: If the current dataset or its archive identity is invalid.
        FileExistsError: If the archive destination already exists.
        OSError: If reading, staging, renaming, publishing, or restoring fails.
    """
    archive = None
    if output_path.exists() or output_path.is_symlink():
        archive = _archive_path(output_path=output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(dataset.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if archive is not None:
            output_path.rename(target=archive)
        try:
            temporary.replace(target=output_path)
        except OSError:
            if archive is not None:
                archive.replace(target=output_path)
            raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def generate_dataset(output_path: Path = OUTPUT_PATH) -> CalibrationDataset:
    """Generate one independent harness answer per model per gold request.

    Every invocation creates a new dataset identity. Configuration and harness
    failures are recorded per candidate and do not prevent later attempts.
    Langfuse tracing is disabled for the process before generation.
    No resume or judge calls occur. After generation and staging finish, an
    existing output is renamed to <stem>_<old_dataset_id><suffix>, preserving
    its bytes and human ratings, and the staged dataset is published atomically.
    A failed publish restores the old output; temporary files are cleaned up.
    Invalid existing datasets, unsafe archive identities, and archive collisions
    leave the current output untouched. Concurrent edits are unsupported.

    Raises:
        ValueError: If a dataset, vault path, or archive identity is invalid.
        FileExistsError: If the archive destination already exists.
        OSError: If reading, staging, renaming, publishing, or restoring fails.
    """
    os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

    gold = load_dataset_document(path=DATASET_PATH)
    if not VAULT_PATH.is_dir():
        raise ValueError("vault must be an existing directory")

    configs = {
        name: CandidateConfig(
            provider="openai",
            model=name,
            settings={"reasoning_effort": REASONING_EFFORT, "use_responses_api": True},
            tool_binding_kwargs={"strict": True},
        )
        for name in MODELS
    }
    models = {}
    setup_errors = {}
    for name, config in configs.items():
        try:
            models[name] = config.create_model()
        except Exception as exc:
            setup_errors[name] = exc

    questions = []
    for case in gold.entries:
        candidates = []
        for name, config in configs.items():
            logger.info("Generating %s / %s", case.id, name)
            try:
                if name in setup_errors:
                    raise setup_errors[name]
                result = run_in_harness(
                    chat_model=models[name],
                    user_prompt=case.request,
                    vault_path=str(VAULT_PATH),
                    system_prompt=SYSTEM_PROMPT,
                    max_api_calls=MAX_API_CALLS,
                    model_label=name,
                    tool_binding_kwargs=config.tool_binding_kwargs,
                )
                candidate = Candidate(
                    candidate_id=name,
                    model_id=name,
                    answer=result.final_answer,
                )
            except Exception as exc:
                logger.error("Generation failed %s / %s: %s", case.id, name, exc)
                candidate = Candidate(
                    candidate_id=name,
                    model_id=name,
                    generation_error=ErrorEvidence(
                        type=type(exc).__name__,
                        message=str(exc),
                        evidence=json_value(value=getattr(exc, "trace", None)),
                    ),
                )
            candidates.append(candidate)

        questions.append(
            Question(
                case_id=case.id,
                request=case.request,
                ideal_answer=case.answer_criteria.ideal_answer,
                candidates=candidates,
            )
        )

    dataset = CalibrationDataset(
        dataset_id=uuid4().hex,
        created_at=datetime.now(timezone.utc).isoformat(),
        generation=GenerationRun(
            gold_schema_version=gold.schema_version,
            gold_dataset_path=str(DATASET_PATH),
            vault_path=str(VAULT_PATH),
            system_prompt=SYSTEM_PROMPT,
            max_api_calls=MAX_API_CALLS,
            models=configs,
        ),
        questions=questions,
    )
    _save_generated_dataset(dataset=dataset, output_path=output_path)
    return dataset


def main() -> int:
    """Generate using module defaults and .env; report failures with exit code 1.

    Setup and persistence exceptions propagate. No CLI configuration is accepted.
    """
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    try:
        dataset = generate_dataset()
        failures = [
            (q.case_id, c.model_id)
            for q in dataset.questions
            for c in q.candidates
            if c.generation_error is not None
        ]
        print(f"Saved {len(dataset.questions)} questions to {OUTPUT_PATH}")
        for case_id, model_id in failures:
            print(f"Unsuccessful: {case_id} / {model_id}")
        return 1 if failures else 0
    finally:
        flush()


if __name__ == "__main__":
    raise SystemExit(main())
