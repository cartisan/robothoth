"""Exercise dataset loading, nullable widgets and persisted results in Streamlit."""

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from test_calibration_dataset import balanced_dataset
from test_calibration_runs import PROMPTS, scripted_judge

from src.evaluation.calibration.dataset import load_dataset, save_dataset
from src.evaluation.calibration.runs import SplitContext, input_fingerprint, run_judges

APP_PATH = Path(__file__).resolve().parents[1] / "src/evaluation/calibration/app.py"


@pytest.fixture(autouse=True)
def restore_main_module() -> Iterator[None]:
    """Undo AppTest's main-module replacement so spawn workers stay isolated."""
    original = sys.modules["__main__"]
    yield
    sys.modules["__main__"] = original


def app_for(path: Path) -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    app.text_input(key="dataset_path_input").set_value(str(path))
    app.button(key="load_dataset").click().run()
    assert not app.exception
    return app


def test_load_rate_save_and_switch_dataset_clears_widgets(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    save_dataset(dataset=balanced_dataset(), path=first)
    save_dataset(dataset=balanced_dataset(), path=second)
    app = app_for(path=first)
    assert len(app.tabs) == 2
    assert len(app.number_input) == 18
    assert all(widget.value is None for widget in app.number_input)
    app.number_input[0].set_value(5).run()
    assert (
        app.session_state["dataset"].questions[0].candidates[0].human_ratings.content
        == 5
    )
    app.button(key="save_dataset").click().run()
    assert (
        load_dataset(path=first).questions[0].candidates[0].human_ratings.content == 5
    )
    app.number_input[0].set_value(None).run()
    app.button(key="save_dataset").click().run()
    assert (
        load_dataset(path=first).questions[0].candidates[0].human_ratings.content
        is None
    )
    app.number_input[0].set_value(4).run()
    app.text_input(key="dataset_path_input").set_value(str(second))
    app.button(key="load_dataset").click().run()
    assert all(widget.value is None for widget in app.number_input)
    assert app.session_state["judge_run"] is None


def test_case_blocks_and_adjacent_rating_inputs(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    dataset = balanced_dataset()
    dataset.questions[0].candidates[0].answer = "**Bold answer**\n\n- A list item"
    save_dataset(dataset=dataset, path=path)
    app = app_for(path=path)
    human, calibration = app.tabs
    assert [item.value for item in human.subheader] == ["one", "two"]
    assert [item.value for item in calibration.subheader] == ["two"]
    for tab in (human, calibration):
        assert not tab.expander
        rendered = [item.value for item in tab.markdown]
        assert "**Question**" in rendered
        assert "**Ideal answer**" in rendered
        assert "Held out request" in rendered
        assert "Held out ideal" in rendered
        assert "**model-a**" in rendered
    columns = human.get("column")
    assert len(columns) == 12
    for index, candidate in enumerate(
        candidate for question in dataset.questions for candidate in question.candidates
    ):
        answer, ratings = columns[2 * index : 2 * index + 2]
        assert not answer.number_input
        assert [widget.label for widget in ratings.number_input] == [
            "Content", "Concision", "Usefulness"
        ]
        assert not ratings.markdown
        if candidate.answer is not None:
            assert [item.value for item in answer.markdown] == [candidate.answer]
        else:
            assert "Generation failed: offline" in answer.error[0].value


def test_old_results_restore_prompts_before_widgets_and_track_later_edits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.calibration.runs as runs

    monkeypatch.setattr(runs, "RUNS_DIRECTORY", tmp_path / "runs")
    dataset = balanced_dataset()
    judge, _ = scripted_judge()
    test, _ = dataset.create_test_val_split()
    run_judges(
        dataset=test,
        prompts=PROMPTS,
        judge=judge,
        runs_directory=tmp_path / "runs",
        split_context=SplitContext(
            role="test",
            seed=42,
            source_input_fingerprint=input_fingerprint(dataset=dataset),
        ),
    )
    path = tmp_path / "dataset.json"
    save_dataset(dataset=dataset, path=path)
    app = app_for(path=path)
    assert app.button(key="compute_statistics").disabled
    app.button(key="load_results").click().run()
    assert not app.exception
    assert app.text_area(key="prompt_content").value == "Edited content"
    assert app.text_area(key="prompt_concision").value == "Edited concision"
    assert not app.button(key="compute_statistics").disabled
    app.text_area(key="prompt_content").set_value("New edit").run()
    assert any("earlier prompt snapshot" in item.value for item in app.warning)
    app.button(key="compute_statistics").click().run()
    rendered = " ".join(item.value for item in app.markdown)
    for label in (
        "Spearman",
        "Pearson",
        "Linear weighted",
        "Quadratic weighted",
        "Mean absolute error",
        "Mean signed error",
        "unavailable",
    ):
        assert label in rendered
    assert "judge − human" in " ".join(item.value for item in app.caption)
    app.number_input[9].set_value(5).run()
    assert any(
        "| Mean absolute error | 1.0000 |" in item.value for item in app.markdown
    )
    assert any("| Mean signed error | -1.0000 |" in item.value for item in app.markdown)
    app.number_input[9].set_value(3).run()
    assert any("| Mean signed error | 1.0000 |" in item.value for item in app.markdown)
    app.button(key="load_dataset").click().run()
    assert app.session_state["judge_run"] is None
    assert app.text_area(key="prompt_content").value != "New edit"


def test_judge_button_delivers_edits_and_displays_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.calibration.runs as runs
    from src.evaluation.judge import SemanticJudge

    monkeypatch.setattr(runs, "RUNS_DIRECTORY", tmp_path / "runs")
    judge, messages = scripted_judge()
    monkeypatch.setattr(SemanticJudge, "__init__", lambda self: None)
    monkeypatch.setattr(SemanticJudge, "grade_stage", judge.grade_stage)
    # Keep the shared judge's real schema/boundary behavior, replacing its API model.
    monkeypatch.setattr(SemanticJudge, "model", judge.model, raising=False)
    monkeypatch.setattr(
        SemanticJudge,
        "structured_output_kwargs",
        judge.structured_output_kwargs,
        raising=False,
    )
    path = tmp_path / "dataset.json"
    save_dataset(dataset=balanced_dataset(), path=path)
    app = app_for(path=path)
    app.text_area(key="prompt_content").set_value("Custom prompt").run()
    app.button(key="judge_candidates").click().run()
    assert not app.exception
    assert len(messages) == 6
    assert messages[0][0].content == "Custom prompt"
    assert any("Matches rubric" in item.value for item in app.markdown)
    assert any("offline" in item.value for item in app.error)
    assert len(list((tmp_path / "runs").glob("*.json"))) == 1


def test_invalid_load_shows_error_without_losing_current_dataset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.json"
    save_dataset(dataset=balanced_dataset(), path=path)
    app = app_for(path=path)
    app.text_input(key="dataset_path_input").set_value(str(tmp_path / "missing.json"))
    app.button(key="load_dataset").click().run()
    assert not app.exception
    assert app.error
    assert len(app.number_input) == 18


def test_app_disables_langfuse_on_startup_and_reruns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(name="LANGFUSE_TRACING_ENABLED", value="true")
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    assert not app.exception
    assert os.environ["LANGFUSE_TRACING_ENABLED"] == "false"
    monkeypatch.setenv(name="LANGFUSE_TRACING_ENABLED", value="true")
    app.run()
    assert not app.exception
    assert os.environ["LANGFUSE_TRACING_ENABLED"] == "false"


def install_scripted_judge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list:
    """Replace the external API boundary while exercising production grading."""
    import src.evaluation.calibration.runs as runs
    from src.evaluation.judge import SemanticJudge

    judge, messages = scripted_judge()
    monkeypatch.setattr(runs, "RUNS_DIRECTORY", tmp_path / "runs")
    monkeypatch.setattr(SemanticJudge, "__init__", lambda self: None)
    monkeypatch.setattr(SemanticJudge, "grade_stage", judge.grade_stage)
    monkeypatch.setattr(SemanticJudge, "model", judge.model, raising=False)
    monkeypatch.setattr(
        SemanticJudge,
        "structured_output_kwargs",
        judge.structured_output_kwargs,
        raising=False,
    )
    return messages


def calibration_text(app: AppTest) -> str:
    return " ".join(
        item.value
        for item in (
            *app.tabs[1].markdown,
            *app.tabs[1].subheader,
            *app.tabs[1].caption,
        )
    )


def test_validation_is_explicit_repeatable_and_independent_of_test_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    messages = install_scripted_judge(monkeypatch=monkeypatch, tmp_path=tmp_path)
    dataset = balanced_dataset()
    path = tmp_path / "dataset.json"
    save_dataset(dataset=dataset, path=path)
    app = app_for(path=path)
    initial = calibration_text(app=app)
    assert "Held out request" in initial  # Seed 42 puts case two in test.
    assert "Question" not in [item.value for item in app.tabs[1].markdown]
    assert "Answer A" not in initial
    assert not messages
    assert not app.button(key="validate_held_out").disabled
    assert not any(widget.key == "validation_old_results" for widget in app.selectbox)
    app.text_area(key="prompt_content").set_value("Validation edit").run()
    app.button(key="validate_held_out").click().run()
    assert not app.exception
    assert len(messages) == 6
    assert messages[0][0].content == "Validation edit"
    assert {json.loads(pair[1].content)["request"] for pair in messages} == {"Question"}
    assert app.session_state["judge_run"] is None
    first = app.session_state["validation_run"]
    assert first.split_context.role == "validation"
    assert "one" in [item.value for item in app.tabs[1].subheader]
    assert "Mean signed error" in calibration_text(app=app)
    assert any("offline" in error.value for error in app.tabs[1].error)
    app.number_input[0].set_value(5).run()
    assert "| Mean signed error | -1.0000 |" in calibration_text(app=app)
    app.text_area(key="prompt_content").set_value("Another edit").run()
    assert len(messages) == 6
    assert "one" in [item.value for item in app.tabs[1].subheader]
    assert app.warning
    app.button(key="validate_held_out").click().run()
    assert len(messages) == 12
    assert app.session_state["validation_run"].run_id != first.run_id
    assert messages[6][0].content == "Another edit"
    app.button(key="judge_candidates").click().run()
    assert len(messages) == 18
    assert {json.loads(pair[1].content)["request"] for pair in messages[12:]} == {
        "Held out request"
    }
    assert app.session_state["validation_run"].run_id != first.run_id
    app.button(key="load_dataset").click().run()
    assert not app.session_state["validation_revealed"]
    assert app.session_state["validation_run"] is None
    assert app.session_state["judge_run"] is None
    assert "one" not in [item.value for item in app.tabs[1].subheader]


def test_invalid_split_prevents_judging_and_blank_prompts_disable_validation(
    tmp_path: Path,
) -> None:
    from test_calibration_dataset import make_dataset

    path = tmp_path / "dataset.json"
    save_dataset(dataset=make_dataset(), path=path)
    app = app_for(path=path)
    assert any("even question count" in error.value for error in app.tabs[1].error)
    assert not app.tabs[1].button
    save_dataset(dataset=balanced_dataset(), path=path)
    app.button(key="load_dataset").click().run()
    app.text_area(key="prompt_content").set_value("  ").run()
    assert app.button(key="validate_held_out").disabled
    assert app.button(key="judge_candidates").disabled


def test_failed_revalidation_does_not_display_previous_run_as_new(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.evaluation.judge import SemanticJudge

    install_scripted_judge(monkeypatch=monkeypatch, tmp_path=tmp_path)
    path = tmp_path / "dataset.json"
    save_dataset(dataset=balanced_dataset(), path=path)
    app = app_for(path=path)
    app.button(key="validate_held_out").click().run()
    assert app.session_state["validation_run"] is not None

    def fail(self: SemanticJudge) -> None:
        raise RuntimeError("setup unavailable")

    monkeypatch.setattr(SemanticJudge, "__init__", fail)
    app.button(key="validate_held_out").click().run()
    assert not app.exception
    assert app.session_state["validation_run"] is None
    assert "Mean signed error" not in calibration_text(app=app)
    assert any("setup unavailable" in error.value for error in app.tabs[1].error)


def test_saved_run_options_match_split_and_full_source_and_validation_is_gated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge, _ = scripted_judge()
    dataset = balanced_dataset()
    test, validation = dataset.create_test_val_split()
    directory = tmp_path / "runs"
    fingerprint = input_fingerprint(dataset=dataset)
    saved = {}
    for role, view in (("test", test), ("validation", validation)):
        saved[role] = run_judges(
            dataset=view,
            prompts=PROMPTS,
            judge=judge,
            runs_directory=directory,
            split_context=SplitContext(
                role="test" if role == "test" else "validation",
                seed=42,
                source_input_fingerprint=fingerprint,
            ),
        )
    run_judges(dataset=test, prompts=PROMPTS, judge=judge, runs_directory=directory)
    run_judges(
        dataset=test,
        prompts=PROMPTS,
        judge=judge,
        runs_directory=directory,
        split_context=SplitContext(
            role="test", seed=42, source_input_fingerprint="wrong source"
        ),
    )
    # Held-out changes invalidate a test run with otherwise identical test inputs.
    changed = dataset.model_copy(deep=True)
    changed.questions[0].request = "Changed validation request"
    changed_test, _ = changed.create_test_val_split()
    run_judges(
        dataset=changed_test,
        prompts=PROMPTS,
        judge=judge,
        runs_directory=directory,
        split_context=SplitContext(
            role="test",
            seed=42,
            source_input_fingerprint=input_fingerprint(dataset=changed),
        ),
    )
    messages = install_scripted_judge(monkeypatch=monkeypatch, tmp_path=tmp_path)
    path = tmp_path / "dataset.json"
    save_dataset(dataset=dataset, path=path)
    app = app_for(path=path)
    assert app.selectbox(key="old_results").options == [f"{saved['test'].run_id}.json"]
    assert not messages
    app.button(key="validate_held_out").click().run()
    options = app.selectbox(key="validation_old_results").options
    assert len(options) == 2
    assert f"{saved['validation'].run_id}.json" in options
    app.selectbox(key="validation_old_results").select(
        str(directory / f"{saved['validation'].run_id}.json")
    ).run()
    app.button(key="validation_load_results").click().run()
    assert not app.exception
    assert app.session_state["validation_run"].run_id == saved["validation"].run_id
    assert app.text_area(key="prompt_content").value == "Edited content"
    assert len(messages) == 6  # Loading never judges.
    assert "Mean signed error" in calibration_text(app=app)


@pytest.mark.parametrize("checkpoint", [1, 2])
def test_validation_persistence_failure_clears_previous_results(
    checkpoint: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.calibration.runs as runs

    messages = install_scripted_judge(monkeypatch=monkeypatch, tmp_path=tmp_path)
    path = tmp_path / "dataset.json"
    save_dataset(dataset=balanced_dataset(), path=path)
    app = app_for(path=path)
    app.button(key="validate_held_out").click().run()
    assert app.session_state["validation_run"] is not None
    original = runs.save_run
    calls = 0

    def fail_save(run: runs.JudgeRun, path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == checkpoint:
            raise OSError("disk unavailable")
        original(run=run, path=path)

    monkeypatch.setattr(runs, "save_run", fail_save)
    app.button(key="validate_held_out").click().run()
    assert not app.exception
    assert app.session_state["validation_run"] is None
    assert not app.session_state["validation_show_statistics"]
    assert len(messages) == 6 + checkpoint - 1
    assert "Mean signed error" not in calibration_text(app=app)
    assert any("disk unavailable" in error.value for error in app.tabs[1].error)


def test_bundled_app_schedules_63_calls_per_split_and_separate_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    messages = install_scripted_judge(monkeypatch=monkeypatch, tmp_path=tmp_path)
    dataset = load_dataset(path=Path("src/evaluation/calibration/dataset.json"))
    test, validation = dataset.create_test_val_split()
    for q in dataset.questions:
        for c in q.candidates:
            c.human_ratings.content = 1 if q in test.questions else 5
    path = tmp_path / "dataset.json"
    save_dataset(dataset=dataset, path=path)
    app = app_for(path=path)
    before = calibration_text(app=app)
    for q in validation.questions:
        assert q.request not in before
        assert q.ideal_answer not in before
        for c in q.candidates:
            assert c.answer is not None
            assert c.answer not in before
    app.button(key="judge_candidates").click().run()
    assert not app.exception
    assert len(messages) == 63
    assert {json.loads(pair[1].content)["request"] for pair in messages} == {
        q.request for q in test.questions
    }
    app.button(key="compute_statistics").click().run()
    assert "| Mean signed error | 3.0000 |" in calibration_text(app=app)
    app.button(key="validate_held_out").click().run()
    assert not app.exception
    assert len(messages) == 126
    assert {json.loads(pair[1].content)["request"] for pair in messages[63:]} == {
        q.request for q in validation.questions
    }
    text = calibration_text(app=app)
    assert "| Mean signed error | 3.0000 |" in text
    assert "| Mean signed error | -1.0000 |" in text
    assert len(app.session_state["judge_run"].results) == 63
    assert len(app.session_state["validation_run"].results) == 63
