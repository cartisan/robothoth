"""Judge runs isolate evidence, checkpoint failures and compare current ratings."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from test_calibration_dataset import make_dataset
from test_semantic_judge import structured_response

from src.evaluation.calibration.dataset import HumanRatings
from src.evaluation.calibration.runs import (
    compute_statistics,
    input_fingerprint,
    load_run,
    run_judges,
)
from src.evaluation.judge import JudgeResponseSchema, JudgeStage, SemanticJudge

PROMPTS: dict[JudgeStage, str] = {
    "content": "Edited content",
    "concision": "Edited concision",
    "task_usefulness": "Edited usefulness",
}


def scripted_judge() -> tuple[SemanticJudge, list[Any]]:
    model = MagicMock()
    model.model_dump.return_value = {"model": "judge", "api_key": "hidden"}
    messages: list[Any] = []

    def invoke(input: list[Any]) -> dict:
        messages.append(input)
        payload = json.loads(input[1].content)
        if len(messages) == 2:
            raise ConnectionError("offline")
        return structured_response(
            raw=AIMessage(content="raw evidence"),
            scores=[
                {
                    "rubric": payload["rubric"],
                    "score": 4,
                    "explanation": "Matches rubric",
                }
            ],
        )

    model.with_structured_output.return_value.invoke.side_effect = invoke
    return SemanticJudge(model=model), messages


def test_run_rubric_order_inputs_errors_and_checkpoints(tmp_path: Path) -> None:
    dataset = make_dataset()
    dataset.questions[0].candidates[0].human_ratings = HumanRatings(content=1)
    judge, messages = scripted_judge()
    checkpoints = []

    def progress(done: int, total: int) -> None:
        saved = json.loads(next(tmp_path.glob("*.json")).read_text())
        checkpoints.append((done, total, len(saved["results"])))

    run = run_judges(
        dataset=dataset,
        prompts=PROMPTS,
        judge=judge,
        runs_directory=tmp_path,
        progress=progress,
    )
    assert [item.stage for item in run.results] == [
        "content",
        "content",
        "concision",
        "concision",
        "task_usefulness",
        "task_usefulness",
    ]
    assert [(item.case_id, item.candidate_id) for item in run.results] == [
        ("one", "a"),
        ("one", "b"),
    ] * 3
    assert run.results[1].error is not None
    assert run.results[1].error.message == "content: offline"
    assert run.results[1].calls[0]["raw_response"] is None
    assert run.results[0].score is not None
    assert run.results[0].score.score == 4
    assert run.results[0].calls[0]["raw_response"]["content"] == "raw evidence"
    assert checkpoints == [(i, 6, i) for i in range(1, 7)]
    assert run.complete
    assert "hidden" not in run.model_dump_json()
    assert run.output_schema == JudgeResponseSchema.model_json_schema()
    for index, pair in enumerate(messages):
        assert len(pair) == 2
        assert (
            pair[0].content
            == ["Edited content", "Edited concision", "Edited usefulness"][index // 2]
        )
        payload = json.loads(pair[1].content)
        assert set(payload) == (
            {"request", "final_answer", "rubric", "ideal_answer"}
            if index < 2
            else {"request", "final_answer", "rubric"}
        )
        assert payload["final_answer"] == ["Answer A", "Answer B"][index % 2]
    assert len({id(pair) for pair in messages}) == 6
    assert load_run(path=next(tmp_path.glob("*.json")), dataset=dataset) == run


def test_loading_ignores_human_ratings_but_rejects_changed_inputs(
    tmp_path: Path,
) -> None:
    dataset = make_dataset()
    judge, _ = scripted_judge()
    run = run_judges(
        dataset=dataset, prompts=PROMPTS, judge=judge, runs_directory=tmp_path
    )
    fingerprint = input_fingerprint(dataset=dataset)
    dataset.questions[0].candidates[0].human_ratings.content = 3
    assert input_fingerprint(dataset=dataset) == fingerprint
    path = next(tmp_path.glob("*.json"))
    assert load_run(path=path, dataset=dataset) == run
    for field in ("request", "ideal_answer", "case_id"):
        changed = dataset.model_copy(deep=True)
        setattr(changed.questions[0], field, "different")
        with pytest.raises(ValueError, match="match"):
            load_run(path=path, dataset=changed)
    dataset.questions[0].candidates[0].answer = "different"
    with pytest.raises(ValueError, match="match"):
        load_run(path=path, dataset=dataset)


def test_new_run_and_prompt_snapshot_do_not_share_mutable_editors(
    tmp_path: Path,
) -> None:
    dataset = make_dataset()
    judge, _ = scripted_judge()
    prompts = dict(PROMPTS)
    first = run_judges(
        dataset=dataset, prompts=prompts, judge=judge, runs_directory=tmp_path
    )
    prompts["content"] = "Later edit"
    second = run_judges(
        dataset=dataset, prompts=prompts, judge=judge, runs_directory=tmp_path
    )
    assert first.prompts["content"] == "Edited content"
    assert second.prompts["content"] == "Later edit"
    assert first.run_id != second.run_id
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_partial_run_is_readable_after_callback_interruption(tmp_path: Path) -> None:
    dataset = make_dataset()
    judge, _ = scripted_judge()

    def interrupt(done: int, total: int) -> None:
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        run_judges(
            dataset=dataset,
            prompts=PROMPTS,
            judge=judge,
            runs_directory=tmp_path,
            progress=interrupt,
        )
    saved = load_run(path=next(tmp_path.glob("*.json")), dataset=dataset)
    assert len(saved.results) == 1
    assert saved.complete is False


@pytest.mark.parametrize(
    "human,judge_scores,want,pairs",
    [
        ([1, 2, 2, 4], [1, 3, 3, 5], 1.0, 4),
        ([1, None, 4, 5], [5, 3, None, 1], -1.0, 2),
        ([1, None], [2, 3], None, 1),
        ([1, 1], [2, 3], None, 2),
        ([1, 2], [3, 3], None, 2),
    ],
)
def test_spearman_ties_missing_and_constant_inputs(
    human: list,
    judge_scores: list,
    want: float | None,
    pairs: int,
) -> None:
    from src.evaluation.calibration.runs import rating_statistics

    actual = rating_statistics(human=human, judge=judge_scores)
    assert actual.pair_count == pairs
    assert (
        actual.spearman_rho == pytest.approx(want)
        if want is not None
        else actual.spearman_rho is None
    )


def test_correlations_use_current_human_ratings_and_successes(tmp_path: Path) -> None:
    dataset = make_dataset()
    judge, _ = scripted_judge()
    run = run_judges(
        dataset=dataset, prompts=PROMPTS, judge=judge, runs_directory=tmp_path
    )
    dataset.questions[0].candidates[0].human_ratings = HumanRatings(
        content=2, concision=1
    )
    dataset.questions[0].candidates[1].human_ratings = HumanRatings(
        content=4, concision=2
    )
    values = compute_statistics(dataset=dataset, run=run)
    assert values["content"].pair_count == 1
    assert values["content"].mean_signed_error == 2.0
    assert values["content"].mean_absolute_error == 2.0
    assert values["concision"].pair_count == 2
    assert values["concision"].spearman_rho is None
    assert values["concision"].mean_absolute_error == 2.5
    assert values["concision"].linear_weighted_kappa == 0.0
    assert values["task_usefulness"].pair_count == 0
    assert values["task_usefulness"].mean_absolute_error is None


def test_run_configuration_filters_binding_credentials(tmp_path: Path) -> None:
    judge, _ = scripted_judge()
    judge.structured_output_kwargs["extra_body"] = {"api_key": "binding-secret"}
    run = run_judges(
        dataset=make_dataset(), prompts=PROMPTS, judge=judge, runs_directory=tmp_path
    )
    assert "binding-secret" not in run.model_dump_json()


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_score",
        "reordered",
        "fingerprint",
        "missing_prompt",
        "false_complete",
        "coerced_version",
    ],
)
def test_run_load_rejects_invalid_records(mutation: str, tmp_path: Path) -> None:
    dataset = make_dataset()
    judge, _ = scripted_judge()
    run = run_judges(
        dataset=dataset, prompts=PROMPTS, judge=judge, runs_directory=tmp_path
    )
    raw = run.model_dump()
    if mutation == "wrong_score":
        raw["results"][0]["score"]["rubric"] = "concision"
    elif mutation == "reordered":
        raw["results"] = list(reversed(raw["results"]))
    elif mutation == "fingerprint":
        raw["input_fingerprint"] = "bad"
    elif mutation == "missing_prompt":
        del raw["prompts"]["content"]
    elif mutation == "false_complete":
        raw["results"] = []
    else:
        raw["schema_version"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_run(path=path, dataset=dataset)


@pytest.mark.parametrize(
    "human,judge_scores,expected",
    [
        (
            [1, 2, 4, 5],
            [1, 3, 2, 5],
            (0.8, 0.7483314773547883, 4 / 7, 14 / 19, 0.75, -0.25),
        ),
        ([1, 2, 5], [1, 5, 2], (0.5, -1 / 26, -1 / 8, -1 / 26, 2.0, 0.0)),
        ([1, 2, 5], [1, 2, 5], (1.0, 1.0, 1.0, 1.0, 0.0, 0.0)),
        ([1, 5], [5, 1], (-1.0, -1.0, -1.0, -1.0, 4.0, 0.0)),
        ([1, None, 4, 5], [5, 3, None, 1], (-1.0, -1.0, -1.0, -1.0, 4.0, 0.0)),
        ([1, 1], [2, 2], (None, None, 0.0, 0.0, 1.0, 1.0)),
        ([3, 3], [3, 3], (None, None, None, None, 0.0, 0.0)),
        ([None, 3], [2, 4], (None, None, 0.0, 0.0, 1.0, 1.0)),
        ([3], [3], (None, None, None, None, 0.0, 0.0)),
        ([], [], (None, None, None, None, None, None)),
        ([None], [4], (None, None, None, None, None, None)),
    ],
)
def test_rating_statistics_agreement_bias_and_undefined_metrics(
    human: list,
    judge_scores: list,
    expected: tuple,
) -> None:
    from src.evaluation.calibration.runs import rating_statistics

    result = rating_statistics(human=human, judge=judge_scores)
    actual = (
        result.spearman_rho,
        result.pearson_r,
        result.linear_weighted_kappa,
        result.quadratic_weighted_kappa,
        result.mean_absolute_error,
        result.mean_signed_error,
    )
    for value, want in zip(actual, expected, strict=True):
        if want is None:
            assert value is None
        else:
            assert value == pytest.approx(want)


def test_rating_statistics_rejects_unpaired_inputs() -> None:
    from src.evaluation.calibration.runs import rating_statistics

    with pytest.raises(ValueError):
        rating_statistics(human=[1], judge=[1, 2])


def test_split_context_rejects_wrong_role_seed_source_and_legacy(
    tmp_path: Path,
) -> None:
    from src.evaluation.calibration.runs import SplitContext

    dataset = make_dataset()
    context = SplitContext(role="test", seed=42, source_input_fingerprint="source")
    judge, _ = scripted_judge()
    run = run_judges(
        dataset=dataset,
        prompts=PROMPTS,
        judge=judge,
        runs_directory=tmp_path,
        split_context=context,
    )
    path = next(tmp_path.glob("*.json"))
    assert run.schema_version == 3
    assert load_run(path=path, dataset=dataset, split_context=context) == run
    for changes in (
        {"role": "validation"},
        {"seed": 43},
        {"source_input_fingerprint": "other"},
    ):
        wrong = SplitContext.model_validate(obj={**context.model_dump(), **changes})
        with pytest.raises(ValueError, match="split context"):
            load_run(path=path, dataset=dataset, split_context=wrong)
    with pytest.raises(ValueError, match="split context"):
        load_run(path=path, dataset=dataset)
    for version in (1, 2):
        raw = run.model_dump()
        raw["schema_version"] = version
        del raw["split_context"]
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="schema_version"):
            load_run(path=path, dataset=dataset)
        with pytest.raises(ValueError, match="schema_version"):
            load_run(path=path, dataset=dataset, split_context=context)
