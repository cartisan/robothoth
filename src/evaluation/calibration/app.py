"""Rate local answers and calibrate the shared production judge in Streamlit."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.evaluation.calibration.dataset import (
    SPLIT_SEED,
    CalibrationDataset,
    Candidate,
    HumanRatings,
    Question,
    load_dataset,
    save_dataset,
)
from src.evaluation.calibration.generate_dataset import OUTPUT_PATH
from src.evaluation.calibration.runs import (
    RATING_FIELDS,
    RUNS_DIRECTORY,
    JudgeRun,
    SplitContext,
    compute_statistics,
    input_fingerprint,
    load_run,
    run_judges,
)
from src.evaluation.judge import STAGE_PROMPTS, JudgeStage, SemanticJudge

LABELS: dict[JudgeStage, str] = {
    "content": "Content",
    "concision": "Concision",
    "task_usefulness": "Usefulness",
}


def clear_dataset_state() -> None:
    """Clear widgets, selected results and prompt editors before a new load."""
    for key in list(st.session_state):
        if str(key).startswith(("rating_", "prompt_")) or key in (
            "old_results",
            "validation_old_results",
            "load_error",
            "validation_load_error",
        ):
            del st.session_state[key]
    st.session_state["judge_run"] = None
    st.session_state["show_statistics"] = False
    st.session_state["validation_run"] = None
    st.session_state["validation_show_statistics"] = False
    st.session_state["validation_revealed"] = False


def display_question(question: Question) -> None:
    """Display a case ID and visible Markdown blocks for its request and reference."""
    st.subheader(question.case_id)
    st.markdown("**Question**")
    with st.container(border=True):
        st.markdown(question.request)
    st.markdown("**Ideal answer**")
    with st.container(border=True):
        st.markdown(question.ideal_answer)


def display_answer(candidate: Candidate) -> None:
    """Display a bordered Markdown answer or its recorded generation failure."""
    with st.container(border=True):
        if candidate.answer is not None:
            st.markdown(candidate.answer)
        elif candidate.generation_error is not None:
            st.error(f"Generation failed: {candidate.generation_error.message}")


def human_ratings_tab(dataset: CalibrationDataset, path: Path) -> None:
    """Display ordered candidates and nullable ratings; save to the loaded path."""
    for qi, question in enumerate(dataset.questions):
        display_question(question=question)
        for ci, candidate in enumerate(question.candidates):
            st.markdown(f"**{dataset.generation.models[candidate.model_id].model}**")
            answer_column, ratings_column = st.columns(spec=[3, 1])
            with answer_column:
                display_answer(candidate=candidate)
            ratings = {}
            with ratings_column:
                for stage, field in RATING_FIELDS.items():
                    key = f"rating_{qi}_{ci}_{field}"
                    if key not in st.session_state:
                        st.session_state[key] = getattr(candidate.human_ratings, field)
                    value = st.number_input(
                        label=LABELS[stage],
                        min_value=1,
                        max_value=5,
                        step=1,
                        value=None,
                        key=key,
                    )
                    ratings[field] = None if value is None else int(value)
            candidate.human_ratings = HumanRatings(**ratings)
            st.divider()
    if st.button("Save dataset", key="save_dataset"):
        try:
            save_dataset(dataset=dataset, path=path)
        except OSError as exc:
            st.error(f"Could not save dataset: {exc}")
        else:
            st.success(f"Saved dataset to {path}")


def result_loader(
    dataset: CalibrationDataset,
    context: SplitContext,
    validation: bool = False,
) -> None:
    """Offer only matching split checkpoints and restore prompts before rerender."""
    prefix = "validation_" if validation else ""
    run_key = "validation_run" if validation else "judge_run"
    statistics_key = "validation_show_statistics" if validation else "show_statistics"
    files = []
    for path in RUNS_DIRECTORY.glob("*.json"):
        try:
            load_run(path=path, dataset=dataset, split_context=context)
            files.append(path)
        except ValueError, OSError:
            continue
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    selected = st.selectbox(
        "Validation old results" if validation else "Old results",
        options=[str(path) for path in files],
        format_func=lambda value: Path(value).name,
        key=f"{prefix}old_results",
        placeholder="No saved runs" if not files else "Select a run",
    )

    def restore() -> None:
        # Callbacks run before prompt widgets are instantiated on the next render.
        try:
            restored = load_run(path=selected, dataset=dataset, split_context=context)
        except (ValueError, OSError) as exc:
            st.session_state[f"{prefix}load_error"] = str(exc)
        else:
            st.session_state[f"{prefix}load_error"] = None
            st.session_state[run_key] = restored
            st.session_state[statistics_key] = validation
            for stage, text in restored.prompts.items():
                st.session_state[f"prompt_{stage}"] = text

    st.button(
        "Load validation results" if validation else "Load results",
        key=f"{prefix}load_results",
        disabled=selected is None,
        on_click=restore,
    )
    if error := st.session_state.get(f"{prefix}load_error"):
        st.error(f"Could not load results: {error}")


def execute_run(
    dataset: CalibrationDataset,
    prompts: dict[JudgeStage, str],
    context: SplitContext,
) -> None:
    """Replace the selected split run; setup/checkpoint failures clear old results."""
    validation = context.role == "validation"
    run_key = "validation_run" if validation else "judge_run"
    statistics_key = "validation_show_statistics" if validation else "show_statistics"
    st.session_state[run_key] = None
    st.session_state[statistics_key] = False
    bar = st.progress(value=0.0, text="Starting judge run")

    def report_progress(done: int, total: int) -> None:
        bar.progress(value=done / total, text=f"Judging {done}/{total}")

    try:
        load_dotenv()
        run = run_judges(
            dataset=dataset,
            prompts=prompts,
            judge=SemanticJudge(),
            runs_directory=RUNS_DIRECTORY,
            progress=report_progress,
            split_context=context,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        st.error(f"Could not complete judge run: {exc}. Check saved checkpoints.")
    else:
        st.session_state[run_key] = run
        st.session_state[statistics_key] = validation
        st.success(f"Saved judge run {run.run_id}")


def display_statistics(dataset: CalibrationDataset, run: JudgeRun) -> None:
    """Display all six metrics against current human ratings and successful scores."""
    st.caption(
        "Errors are in rating points. Mean signed error = judge − human; "
        "positive values mean the judge scores higher."
    )
    for stage, value in compute_statistics(dataset=dataset, run=run).items():
        st.markdown(f"**{LABELS[stage]}** · {value.pair_count} pairs")
        metrics = {
            "Spearman’s rho": value.spearman_rho,
            "Pearson’s r": value.pearson_r,
            "Linear weighted Cohen’s κ": value.linear_weighted_kappa,
            "Quadratic weighted Cohen’s κ": value.quadratic_weighted_kappa,
            "Mean absolute error": value.mean_absolute_error,
            "Mean signed error": value.mean_signed_error,
        }
        rows = ["| Statistic | Value |", "| --- | --- |"]
        for label, number in metrics.items():
            formatted = "unavailable" if number is None else f"{number:.4f}"
            rows.append(f"| {label} | {formatted} |")
        st.markdown("\n".join(rows))


def display_cases(dataset: CalibrationDataset, run: JudgeRun | None) -> None:
    """Render this split's references, answers, current ratings and judge evidence."""
    results = (
        {(item.stage, item.case_id, item.candidate_id): item for item in run.results}
        if run is not None
        else {}
    )
    for question in dataset.questions:
        display_question(question=question)
        for candidate in question.candidates:
            st.markdown(f"**{dataset.generation.models[candidate.model_id].model}**")
            display_answer(candidate=candidate)
            st.markdown(
                "Human scores — "
                + ", ".join(
                    f"{field}: {getattr(candidate.human_ratings, field)}"
                    for field in RATING_FIELDS.values()
                )
            )
            for stage in STAGE_PROMPTS:
                result = results.get((stage, question.case_id, candidate.candidate_id))
                if result is not None and result.score is not None:
                    st.markdown(
                        f"**Judge {LABELS[stage]}: {result.score.score}** — "
                        f"{result.score.explanation}"
                    )
                elif result is not None and result.error is not None:
                    st.error(f"Judge {LABELS[stage]}: {result.error.message}")
            st.divider()


def display_run(run: JudgeRun | None, prompts: dict[JudgeStage, str]) -> None:
    """Identify displayed snapshots and mark edited prompts as stale."""
    if run is not None:
        st.caption(
            f"Results: {run.run_id} · {len(run.results)} calls · "
            f"{'complete' if run.complete else 'partial checkpoint'}"
        )
        if run.prompts != prompts:
            action = (
                "Validate on held out data"
                if run.split_context is not None
                and run.split_context.role == "validation"
                else "Judge Candidates"
            )
            st.warning(
                "Displayed results use an earlier prompt snapshot. "
                f"{action} to evaluate the edited prompts."
            )


def split_heading(label: str, dataset: CalibrationDataset) -> None:
    """Label a split with whole-question and usable-answer counts."""
    st.header(label)
    answers = sum(c.answer is not None for q in dataset.questions for c in q.candidates)
    st.caption(f"{len(dataset.questions)} questions · {answers} answers")


def calibration_tab(dataset: CalibrationDataset) -> None:
    """Grade test inputs; reveal validation only on an explicit run request."""
    try:
        test, validation = dataset.create_test_val_split()
    except ValueError as exc:
        st.error(f"Cannot calibrate this dataset: {exc}")
        return
    source_fingerprint = input_fingerprint(dataset=dataset)
    test_context = SplitContext(
        role="test", seed=SPLIT_SEED, source_input_fingerprint=source_fingerprint
    )
    validation_context = SplitContext(
        role="validation", seed=SPLIT_SEED, source_input_fingerprint=source_fingerprint
    )
    split_heading(label="Test set", dataset=test)
    result_loader(dataset=test, context=test_context)
    prompts: dict[JudgeStage, str] = {}
    for stage, prompt in STAGE_PROMPTS.items():
        key = f"prompt_{stage}"
        if key not in st.session_state:
            st.session_state[key] = prompt.text
        prompts[stage] = st.text_area(
            label=f"{LABELS[stage]} prompt", key=key, height=280
        )
    blank = any(not text.strip() for text in prompts.values())
    if st.button("Judge Candidates", key="judge_candidates", disabled=blank):
        execute_run(dataset=test, prompts=prompts, context=test_context)
    run: JudgeRun | None = st.session_state.get("judge_run")
    display_run(run=run, prompts=prompts)
    if st.button(
        "Compute statistics",
        key="compute_statistics",
        disabled=run is None or not run.results,
    ):
        st.session_state["show_statistics"] = True
    if run is not None and st.session_state.get("show_statistics"):
        display_statistics(dataset=test, run=run)
    display_cases(dataset=test, run=run)
    if st.button("Validate on held out data", key="validate_held_out", disabled=blank):
        st.session_state["validation_revealed"] = True
        execute_run(dataset=validation, prompts=prompts, context=validation_context)
    if st.session_state.get("validation_revealed"):
        split_heading(label="Validation set", dataset=validation)
        result_loader(dataset=validation, context=validation_context, validation=True)
        validation_run: JudgeRun | None = st.session_state.get("validation_run")
        display_run(run=validation_run, prompts=prompts)
        if validation_run is not None and st.session_state.get(
            "validation_show_statistics"
        ):
            display_statistics(dataset=validation, run=validation_run)
        display_cases(dataset=validation, run=validation_run)


def main() -> None:
    """Disable process Langfuse tracing and render human-rating and judge tabs."""
    os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
    st.set_page_config(page_title="LLM Judge Calibration", layout="wide")
    st.title("LLM Judge Calibration")
    source = st.text_input(
        "Dataset path", value=str(OUTPUT_PATH), key="dataset_path_input"
    )
    if st.button("Load dataset", key="load_dataset"):
        try:
            loaded = load_dataset(path=source)
        except (ValueError, OSError) as exc:
            st.error(f"Could not load dataset: {exc}")
        else:
            clear_dataset_state()
            st.session_state["dataset"] = loaded
            st.session_state["loaded_path"] = Path(source).resolve()
    dataset: CalibrationDataset | None = st.session_state.get("dataset")
    if dataset is None:
        st.info("Load a calibration dataset to begin.")
        return
    st.caption(f"Dataset: {dataset.dataset_id}")
    human, calibration = st.tabs(["Human ratings", "Judge calibration"])
    with human:
        human_ratings_tab(dataset=dataset, path=st.session_state["loaded_path"])
    with calibration:
        calibration_tab(dataset=dataset)


if __name__ == "__main__":
    main()
