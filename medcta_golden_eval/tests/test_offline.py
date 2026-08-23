import json
from pathlib import Path

import pytest

import medcta_golden_eval.pipeline as pipeline_module
from medcta_golden_eval import llm
import shared.llm as shared_llm
from medcta_golden_eval.adapter import ALLOWED_TOOLS, SOURCE_REVISION, STARTER_CASE_IDS, adapt_raw_dataset
from medcta_golden_eval.config import DATA_PATH, PROJECT_DIR
from medcta_golden_eval.data import load_dataset, validate_dataset
from medcta_golden_eval.metrics import build_report
from medcta_golden_eval.pipeline import report_existing_run, run_pipeline
from medcta_golden_eval.prompts import build_user_prompt
from medcta_golden_eval.runner import _judge_answer_correctness as _real_judge_answer_correctness
from medcta_golden_eval.runner import _judge_answer_equivalence as _real_judge_answer_equivalence
from medcta_golden_eval.runner import run_case
from medcta_golden_eval.schema import (
    ACTIONS,
    safe_json_loads,
    validate,
)


@pytest.fixture(autouse=True)
def _stub_judges(monkeypatch):
    """Keep offline tests from making real judge calls (.env has live Azure creds).

    Individual tests may override either target afterwards via their own
    monkeypatch.setattr(...) call, which wins for the duration of that test.
    """
    monkeypatch.setattr(
        "medcta_golden_eval.runner._judge_answer_correctness",
        lambda answer, accepted, **_: 1.0 if answer in (accepted or []) else 0.0,
    )
    monkeypatch.setattr(
        "medcta_golden_eval.runner._judge_answer_equivalence",
        lambda previous, current: 1.0 if previous == current else 0.0,
    )


def _tool(name):
    return {"name": name, "description": f"Description for {name}", "inputs": []}


def _case(
    case_id="case-1",
    reference_tools=("OCR",),
    available_tools=None,
    answer="gold answer",
):
    names = list(available_tools or dict.fromkeys(reference_tools))
    steps = [
        {
            "step_index": index,
            "action": "CALL_TOOL",
            "tool_name": name,
            "reference_arguments": {"image": "image/example.jpg"},
            "reference_observation": f"observation-{index}-{name}",
            "reference_answer": None,
        }
        for index, name in enumerate(reference_tools)
    ]
    steps.append(
        {
            "step_index": len(steps),
            "action": "FINAL_ANSWER",
            "tool_name": None,
            "reference_arguments": None,
            "reference_observation": None,
            "reference_answer": answer,
        }
    )
    return {
        "case_id": str(case_id),
        "question": f"Question for {case_id}?",
        "image_path": "image/example.jpg",
        "image_reference": "https://example.test/image.jpg",
        "available_tools": [_tool(name) for name in names],
        "reference_steps": steps,
        "ground_truth": {"accepted_answers": [answer], "blacklist": None},
    }


def _payload(*cases):
    return {
        "schema_version": 1,
        "dataset": {
            "name": "test/MedCTA",
            "license": "apache-2.0",
            "source_page": "https://example.test",
            "source_url": "https://example.test/raw.json",
            "source_revision": "test-revision",
            "image_base_url": "https://example.test",
        },
        "selected_case_ids": [case["case_id"] for case in cases],
        "cases": list(cases),
    }


def _response(action, tool_name=None, answer=None, reasoning="working hypothesis", **extra):
    if answer is None:
        answer = "provisional hypothesis"
    return json.dumps(
        {
            "action": action,
            "tool_name": tool_name,
            "answer": answer,
            "reasoning": reasoning,
            **extra,
        }
    )


def _scripted(*responses):
    iterator = iter(responses)
    return lambda _system, _user, _image: next(iterator)


def _raw_case():
    return {
        "tools": [
            {
                "name": "OCR",
                "description": "Read text",
                "inputs": [{"type": "image", "name": "image"}],
            }
        ],
        "files": [{"type": "image", "path": "image/image_1.jpg", "url": None}],
        "dialogs": [
            {"role": "user", "content": "What is shown?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "OCR",
                            "arguments": {"image": "image/image_1.jpg"},
                        },
                    }
                ],
                "thought": "THIS MUST NOT BE ADAPTED",
            },
            {
                "role": "tool",
                "name": "OCR",
                "content": {"type": "text", "content": "visible label"},
            },
            {"role": "assistant", "content": "gold answer"},
        ],
        "gt_answer": {"whitelist": [["gold answer"]], "blacklist": None},
    }


def test_action_ontology_is_the_two_action_contract():
    assert ACTIONS == ("CALL_TOOL", "FINAL_ANSWER")


def test_validate_accepts_call_tool_and_final_answer():
    called, error = validate(
        {
            "action": "CALL_TOOL",
            "tool_name": "OCR",
            "answer": "provisional hypothesis",
            "reasoning": "need to read the label first",
        },
        available_tools=["OCR"],
    )
    assert error is None
    assert called is not None
    assert called.tool_name == "OCR"
    assert called.answer == "provisional hypothesis"
    assert called.reasoning == "need to read the label first"

    final, error = validate(
        {
            "action": "FINAL_ANSWER",
            "tool_name": None,
            "answer": "Liver",
            "reasoning": "consistent with all evidence",
        },
        available_tools=["OCR"],
    )
    assert error is None
    assert final is not None
    assert final.answer == "Liver"
    assert final.reasoning == "consistent with all evidence"


@pytest.mark.parametrize(
    "raw, expected_error",
    [
        (
            {"action": "CALL_TOOL", "tool_name": "NotATool", "answer": "hyp", "reasoning": "r"},
            "invalid or unavailable",
        ),
        (
            {"action": "CALL_TOOL", "tool_name": "OCR", "answer": None, "reasoning": "r"},
            "requires a nonempty current-best answer",
        ),
        (
            {"action": "FINAL_ANSWER", "tool_name": "OCR", "answer": "text", "reasoning": "r"},
            "requires tool_name null",
        ),
        (
            {"action": "FINAL_ANSWER", "tool_name": None, "answer": None, "reasoning": "r"},
            "requires a nonempty answer",
        ),
        (
            {"action": "UNKNOWN", "tool_name": None, "answer": None, "reasoning": "r"},
            "invalid action",
        ),
        (
            {
                "action": "CALL_TOOL",
                "tool_name": "OCR",
                "answer": None,
                "reason": "extra",
            },
            "response keys mismatch",
        ),
        (
            {"action": "CALL_TOOL", "tool_name": "OCR", "answer": "hyp", "reasoning": None},
            "reasoning is required",
        ),
        (
            {"action": "CALL_TOOL", "tool_name": "OCR", "answer": "hyp", "reasoning": "   "},
            "reasoning is required",
        ),
    ],
)
def test_validate_rejects_invalid_actions_and_cross_field_values(raw, expected_error):
    parsed, error = validate(raw, available_tools=["OCR"])
    assert parsed is None and expected_error in error


def test_safe_json_loads_never_raises():
    parsed, error = safe_json_loads("not json")
    assert parsed is None and error.startswith("json_parse:")



def test_pinned_subset_contains_exact_ids_tools_and_no_thoughts():
    subset_path = PROJECT_DIR / "data" / "medcta" / "subset_v1.json"
    payload = load_dataset(subset_path)
    assert payload["selected_case_ids"] == list(STARTER_CASE_IDS)
    assert {tool["name"] for case in payload["cases"] for tool in case["available_tools"]} == ALLOWED_TOOLS
    assert all(SOURCE_REVISION in case["image_reference"] for case in payload["cases"])
    assert all(case["image_reference"].startswith("https://") for case in payload["cases"])
    assert [
        len([step for step in case["reference_steps"] if step["action"] == "CALL_TOOL"])
        for case in payload["cases"]
    ] == [2, 3, 3, 5, 4, 2, 5, 2, 5, 3, 3]
    assert "thought" not in json.dumps(payload).casefold()


def test_adapter_retains_arguments_and_observations_but_drops_thoughts():
    adapted = adapt_raw_dataset({"0": _raw_case()}, ["0"])
    case = adapted["cases"][0]
    assert case["reference_steps"][0]["reference_arguments"] == {
        "image": "image/image_1.jpg"
    }
    assert case["reference_steps"][0]["reference_observation"] == "visible label"
    assert case["ground_truth"]["accepted_answers"] == ["gold answer"]
    assert "THIS MUST NOT BE ADAPTED" not in json.dumps(adapted)


def test_adapter_rejects_mismatched_tool_observation():
    raw_case = _raw_case()
    raw_case["dialogs"][2]["name"] = "ImageDescription"
    with pytest.raises(ValueError, match="call/observation tool mismatch"):
        adapt_raw_dataset({"0": raw_case}, ["0"])


def test_dataset_validation_rejects_nonterminal_final_action():
    case = _case()
    case["reference_steps"].insert(0, case["reference_steps"].pop())
    payload = _payload(case)
    payload["cases"][0]["reference_steps"] = [
        {**step, "step_index": index}
        for index, step in enumerate(payload["cases"][0]["reference_steps"])
    ]
    with pytest.raises(ValueError, match="exactly one terminal FINAL_ANSWER"):
        validate_dataset(payload)


def test_dataset_validation_requires_a_terminal_final_action():
    case = _case(reference_tools=("OCR", "ImageDescription"))
    case["reference_steps"] = case["reference_steps"][:-1]
    payload = _payload(case)
    with pytest.raises(ValueError, match="exactly one terminal FINAL_ANSWER"):
        validate_dataset(payload)


def test_prompt_contains_visible_state_but_not_hidden_reference_data():
    case = _case(answer="HIDDEN_GOLD")
    case["reference_steps"][0]["reference_observation"] = "FUTURE_OBSERVATION"
    prompt = build_user_prompt(case, [], [])
    assert case["question"] in prompt
    assert case["image_reference"] in prompt
    assert "OCR" in prompt
    assert "HIDDEN_GOLD" not in prompt
    assert "FUTURE_OBSERVATION" not in prompt


def test_prompt_includes_actual_prior_actions_and_only_revealed_observations():
    case = _case(reference_tools=("OCR", "ImageDescription"))
    prompt = build_user_prompt(
        case,
        [{"step_index": 0, "action": "CALL_TOOL", "tool_name": "ImageDescription", "answer": None}],
        [{"step_index": 0, "reference_tool_name": "OCR", "content": "revealed now"}],
    )
    assert "ImageDescription" in prompt
    assert "revealed now" in prompt
    assert "observation-1-ImageDescription" not in prompt


def test_prompt_hides_prior_answer_and_reasoning_to_avoid_anchoring():
    case = _case(reference_tools=("OCR", "ImageDescription"))
    prompt = build_user_prompt(
        case,
        [
            {
                "step_index": 0,
                "action": "CALL_TOOL",
                "tool_name": "OCR",
                "answer": "SECRET_PRIOR_GUESS",
                "reasoning": "SECRET_PRIOR_REASONING",
            }
        ],
        [],
    )
    assert "OCR" in prompt
    assert "SECRET_PRIOR_GUESS" not in prompt
    assert "SECRET_PRIOR_REASONING" not in prompt


def test_perfect_replay_matches_tools_finalization_and_answer(monkeypatch):
    monkeypatch.setattr("medcta_golden_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
    case = _case(reference_tools=("OCR", "ImageDescription"))
    trace = run_case(
        case,
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR"),
            _response("CALL_TOOL", "ImageDescription"),
            _response("FINAL_ANSWER", answer="Gold answer."),
        ),
    )
    assert trace["status"] == "completed"
    assert trace["trajectory_exact_match"] is True
    assert trace["final_answer_match"] is True
    assert [step["reference_tool_match"] for step in trace["steps"][:2]] == [True, True]
    assert "observation-0-OCR" in trace["steps"][1]["user_prompt"]


def test_wrong_tool_at_non_terminal_step_is_still_forced_onto_the_golden_tool():
    """A CALL_TOOL attempt at the wrong tool is not an early exit -- it is
    still overridden, exactly like an attempted FINAL_ANSWER would be: the
    golden tool's observation is revealed and model_tool_sequence records
    the forced (golden) tool, not the model's real attempted one.
    """
    case = _case(
        reference_tools=("OCR",),
        available_tools=("OCR", "ImageDescription"),
    )
    trace = run_case(
        case,
        call_fn=_scripted(
            _response("CALL_TOOL", "ImageDescription"),
            _response("FINAL_ANSWER", answer="gold answer"),
        ),
    )
    assert trace["status"] == "completed"
    assert trace["model_tool_sequence"] == ["OCR"]
    assert trace["trajectory_exact_match"] is True
    assert trace["attempted_trajectory_exact_match"] is False
    assert trace["attempted_model_actions"][0] == {
        "step_index": 0,
        "action": "CALL_TOOL",
        "tool_name": "ImageDescription",
    }
    first = trace["steps"][0]
    assert first["reference_tool_match"] is False
    assert first["attempted_early_exit"] is False
    assert first["reference_observation_revealed"]["reference_tool_name"] == "OCR"
    second_prompt = trace["steps"][1]["user_prompt"]
    assert "observation-0-OCR" in second_prompt


def test_attempted_early_exit_never_stops_the_case_and_is_recorded():
    """A model that tries FINAL_ANSWER at a non-terminal reference step still
    gets the next reference observation revealed, the case still reaches
    status == 'completed' at the true final step, and
    attempted_premature_finalization is True.
    """
    trace = run_case(
        _case(reference_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(
            _response("FINAL_ANSWER", answer="early guess 1"),
            _response("FINAL_ANSWER", answer="early guess 2"),
            _response("FINAL_ANSWER", answer="true final answer"),
        ),
    )
    assert trace["status"] == "completed"
    assert trace["reference_step_count"] == 3
    assert trace["attempted_steps"] == 3
    assert [step["attempted_early_exit"] for step in trace["steps"]] == [True, True, False]
    assert trace["attempted_premature_finalization"] is True
    assert trace["model_tool_sequence"] == ["OCR", "ImageDescription"]
    assert trace["steps"][0]["reference_observation_revealed"]["reference_tool_name"] == "OCR"
    assert trace["steps"][1]["reference_observation_revealed"]["reference_tool_name"] == "ImageDescription"
    assert trace["final_answer"] == "true final answer"


def test_prior_actions_history_reflects_forced_golden_action_not_the_real_attempt():
    """The PRIOR MODEL ACTIONS shown back to the model on later turns must
    always record the forced/golden action, never the model's real declared
    action or tool_name -- even when that real attempt was FINAL_ANSWER or a
    non-golden tool.
    """
    case = _case(reference_tools=("OCR", "ImageDescription"))
    prompts = []

    def call_fn(_system, user, _image):
        prompts.append(user)
        if len(prompts) == 1:
            return _response("FINAL_ANSWER", answer="early guess")
        if len(prompts) == 2:
            return _response("CALL_TOOL", "OCR", answer="wrong tool attempt")
        return _response("FINAL_ANSWER", answer="final answer")

    trace = run_case(case, call_fn=call_fn)
    assert trace["status"] == "completed"
    third_prompt = prompts[2]
    history = third_prompt.split("PRIOR MODEL ACTIONS:\n", 1)[1].split(
        "\n\nPRIOR REFERENCE", 1
    )[0]
    actions = json.loads(history)
    assert actions == [
        {"step_index": 0, "action": "CALL_TOOL", "tool_name": "OCR"},
        {"step_index": 1, "action": "CALL_TOOL", "tool_name": "ImageDescription"},
    ]
    assert trace["model_actions"] == actions + [
        {"step_index": 2, "action": "FINAL_ANSWER", "tool_name": None}
    ]
    assert trace["attempted_model_actions"] == [
        {"step_index": 0, "action": "FINAL_ANSWER", "tool_name": None},
        {"step_index": 1, "action": "CALL_TOOL", "tool_name": "OCR"},
        {"step_index": 2, "action": "FINAL_ANSWER", "tool_name": None},
    ]
    assert trace["attempted_trajectory_exact_match"] is False


def test_extra_tool_call_at_final_step_still_uses_the_models_answer(monkeypatch):
    """There is no further golden step to force into at the reference's
    final step, so a stray CALL_TOOL there is only recorded as a diagnostic
    (attempted_extra_tool_call); the model's own answer at that step is
    still used as the case's final answer and the case still completes.
    """
    monkeypatch.setattr("medcta_golden_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
    trace = run_case(
        _case(reference_tools=("OCR",), available_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR"),
            _response("CALL_TOOL", "ImageDescription", answer="answer despite calling a tool"),
        ),
    )
    assert trace["status"] == "completed"
    assert trace["steps"][-1]["attempted_extra_tool_call"] is True
    assert trace["steps"][-1]["reference_observation_revealed"] is None
    assert trace["final_answer"] == "answer despite calling a tool"
    assert trace["final_answer_match"] is True
    assert trace["model_tool_sequence"] == ["OCR"]


def test_premature_finalization_progress_set_correctly_on_the_trace(monkeypatch):
    monkeypatch.setattr(
        "medcta_golden_eval.runner._judge_answer_correctness",
        lambda answer, accepted, **_: 1.0,
    )
    bails_immediately = run_case(
        _case("early", reference_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(
            _response("FINAL_ANSWER", answer="e1"),
            _response("FINAL_ANSWER", answer="e2"),
            _response("FINAL_ANSWER", answer="e3"),
        ),
    )
    bails_after_one_tool = run_case(
        _case("late", reference_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR"),
            _response("FINAL_ANSWER", answer="l2"),
            _response("FINAL_ANSWER", answer="l3"),
        ),
    )
    never_bails = run_case(
        _case("never", reference_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR"),
            _response("CALL_TOOL", "ImageDescription"),
            _response("FINAL_ANSWER", answer="n3"),
        ),
    )
    assert bails_immediately["premature_finalization_progress"] == 0.0
    assert bails_after_one_tool["premature_finalization_progress"] == 0.5
    assert never_bails["premature_finalization_progress"] is None
    assert never_bails["attempted_premature_finalization"] is False


def test_initial_invalid_response_is_repaired_and_still_counted():
    trace = run_case(
        _case(),
        call_fn=_scripted(
            _response("CALL_TOOL", "ocr"),
            _response("CALL_TOOL", "OCR"),
            _response("FINAL_ANSWER", answer="gold answer"),
        ),
    )
    first_output = trace["steps"][0]["model_output"]
    assert trace["status"] == "completed"
    assert first_output["repaired"] is True
    assert first_output["initial_invalid"] is True
    assert trace["initial_invalid_steps"] == 1


def test_twice_invalid_response_stops_case():
    trace = run_case(
        _case(),
        call_fn=_scripted(
            _response("CALL_TOOL", "not-a-tool"),
            _response("CALL_TOOL", "still-not-a-tool"),
        ),
    )
    assert trace["status"] == "invalid_action"
    assert trace["attempted_steps"] == 1
    assert trace["steps"][0]["model_output"]["validation_error"]


def test_inference_failure_is_recorded_without_becoming_invalid_action():
    def fail(*_args):
        raise RuntimeError("provider unavailable")

    trace = run_case(_case(), call_fn=fail)
    output = trace["steps"][0]["model_output"]
    assert trace["status"] == "inference_failure"
    assert output["inference_failure"] is True
    assert output["initial_response_received"] is False
    assert output["initial_invalid"] is False


def test_repair_inference_failure_preserves_initial_invalidity():
    calls = iter([_response("CALL_TOOL", "invalid")])

    def stub(*_args):
        try:
            return next(calls)
        except StopIteration as exc:
            raise RuntimeError("repair provider failure") from exc

    trace = run_case(_case(), call_fn=stub)
    output = trace["steps"][0]["model_output"]
    assert trace["status"] == "inference_failure"
    assert output["initial_invalid"] is True
    assert output["repaired"] is True


def _score_by_answer_text(mapping):
    """Judge stub: score looked up by the literal answer text, default 0.0."""
    return lambda answer, accepted, **_: mapping.get(answer, 0.0)


def test_mean_score_improvement_and_first_step_answer_score(monkeypatch):
    monkeypatch.setattr(
        "medcta_golden_eval.runner._judge_answer_correctness",
        _score_by_answer_text({"A0": 0.2, "A1": 0.8, "B0": 0.0, "B1": 0.0, "B2": 1.0}),
    )
    case_a = run_case(
        _case("a", reference_tools=("OCR",)),
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="A0"),
            _response("FINAL_ANSWER", answer="A1"),
        ),
    )
    case_b = run_case(
        _case("b", reference_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="B0"),
            _response("CALL_TOOL", "ImageDescription", answer="B1"),
            _response("FINAL_ANSWER", answer="B2"),
        ),
    )
    report = build_report([case_a, case_b], model="test")

    # a: (0.8 - 0.2) / 1 forced tool call = 0.6 ; b: (1.0 - 0.0) / 2 forced tool calls = 0.5
    assert report["mean_score_improvement"] == {
        "mean_score": round((0.6 + 0.5) / 2, 4),
        "n_scored": 2,
        "n_evaluable": 2,
    }
    # first-step scores: a=0.2, b=0.0
    assert report["first_step_answer_score"] == {
        "mean_score": round((0.2 + 0.0) / 2, 4),
        "n_scored": 2,
        "n_evaluable": 2,
    }


def test_mean_score_before_premature_finalization_scores_first_attempt_not_eventual_final(
    monkeypatch,
):
    """A model that attempts an early exit with answer text X, is forced to
    continue, and eventually finalizes with different answer text Y: confirms
    mean_score_before_premature_finalization scores X, not Y.
    """
    monkeypatch.setattr(
        "medcta_golden_eval.runner._judge_answer_correctness",
        _score_by_answer_text({"X": 0.3, "Y": 0.9}),
    )
    trace = run_case(
        _case(reference_tools=("OCR",)),
        call_fn=_scripted(
            _response("FINAL_ANSWER", answer="X"),
            _response("FINAL_ANSWER", answer="Y"),
        ),
    )
    assert trace["status"] == "completed"
    assert trace["attempted_premature_finalization"] is True
    assert trace["final_answer"] == "Y"

    report = build_report([trace], model="test")
    assert report["mean_score_before_premature_finalization"] == {
        "mean_score": 0.3,
        "n_scored": 1,
        "n_evaluable": 1,
    }
    assert report["final_answer_mean_score"]["mean_score"] == 0.9


def test_requested_metrics_computed_correctly_across_a_mixed_set(monkeypatch):
    monkeypatch.setattr(
        "medcta_golden_eval.runner._judge_answer_correctness",
        _score_by_answer_text(
            {"P0": 0.4, "P1": 0.9, "B0": 0.5, "B1": 0.6, "B2": 1.0, "C0": 0.2, "C1": 0.0}
        ),
    )
    perfect = run_case(
        _case("perfect", reference_tools=("OCR",)),
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="P0"),
            _response("FINAL_ANSWER", answer="P1"),
        ),
    )
    bail = run_case(
        _case("bail", reference_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(
            _response("FINAL_ANSWER", answer="B0"),
            _response("FINAL_ANSWER", answer="B1"),
            _response("FINAL_ANSWER", answer="B2"),
        ),
    )
    never = run_case(
        _case("never", reference_tools=("OCR",)),
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="C0"),
            _response("FINAL_ANSWER", answer="C1"),
        ),
    )
    report = build_report([perfect, bail, never], model="test")

    assert report["trace_status_counts"] == {"completed": 3}
    # final_answer_match: perfect (0.9>=0.8) True, bail (1.0) True, never (0.0) False
    assert report["final_answer_accuracy"] == {
        "rate": 2 / 3,
        "numerator": 2,
        "denominator": 3,
    }
    assert report["final_answer_mean_score"] == {
        "mean_score": round((0.9 + 1.0 + 0.0) / 3, 4),
        "n_scored": 3,
        "n_evaluable": 3,
    }
    assert report["first_step_answer_score"] == {
        "mean_score": round((0.4 + 0.5 + 0.2) / 3, 4),
        "n_scored": 3,
        "n_evaluable": 3,
    }
    # perfect: (0.9-0.4)/1=0.5 ; bail: (1.0-0.5)/2=0.25 ; never: (0.0-0.2)/1=-0.2
    assert report["mean_score_improvement"] == {
        "mean_score": round((0.5 + 0.25 - 0.2) / 3, 4),
        "n_scored": 3,
        "n_evaluable": 3,
    }
    # only "bail" ever attempted an early exit, at step 0 of 2 reference tools
    assert report["premature_finalization_rate"] == {
        "rate": 1 / 3,
        "numerator": 1,
        "denominator": 3,
    }
    assert report["premature_finalization_progress"] == {"mean_rate": 0.0, "n_cases": 1}
    assert report["mean_score_before_premature_finalization"] == {
        "mean_score": 0.5,
        "n_scored": 1,
        "n_evaluable": 1,
    }


def test_invalid_and_inference_metrics_remain_separate():
    repaired = run_case(
        _case("repaired"),
        call_fn=_scripted(
            "not json",
            _response("CALL_TOOL", "OCR"),
            _response("FINAL_ANSWER", answer="gold answer"),
        ),
    )

    def fail(*_args):
        raise RuntimeError("offline")

    failed = run_case(_case("failed"), call_fn=fail)
    report = build_report([repaired, failed], model="test")
    assert report["inference_failure_count"] == 1
    assert report["inference_failure_case_count"] == 1
    assert report["n_evaluable_cases"] == 1
    assert report["repair_count"] == 1


def test_inference_only_run_has_no_data_denominators():
    def fail(*_args):
        raise RuntimeError("provider unavailable")

    failed = run_case(_case("failed"), call_fn=fail)
    report = build_report([failed], model="test")

    assert report["n_selected_cases"] == 1
    assert report["n_evaluable_cases"] == 0
    assert report["inference_failure_count"] == 1
    for metric in (
        "final_answer_accuracy",
        "successful_revision_rate",
        "missed_revision_rate",
        "overreaction_rate",
        "kept_correct_rate",
        "premature_finalization_rate",
    ):
        assert report[metric] == {"rate": None, "numerator": 0, "denominator": 0}
    for metric in (
        "final_answer_mean_score",
        "first_step_answer_score",
        "mean_score_improvement",
        "mean_score_before_premature_finalization",
    ):
        assert report[metric] == {"mean_score": None, "n_scored": 0, "n_evaluable": 0}
    assert report["premature_finalization_progress"] == {"mean_rate": None, "n_cases": 0}


def test_inference_failure_does_not_dilute_evaluable_case_metrics(monkeypatch):
    monkeypatch.setattr("medcta_golden_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
    perfect = run_case(
        _case("perfect"),
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR"),
            _response("FINAL_ANSWER", answer="gold answer"),
        ),
    )

    def fail(*_args):
        raise RuntimeError("provider unavailable")

    failed = run_case(_case("failed"), call_fn=fail)
    report = build_report([perfect, failed], model="test")

    assert report["n_selected_cases"] == 2
    assert report["n_evaluable_cases"] == 1
    assert report["final_answer_accuracy"]["rate"] == 1.0
    assert report["final_answer_accuracy"]["denominator"] == 1


def test_equivalence_short_circuit_skips_judge_call_for_identical_text(monkeypatch):
    """Exercises the real _judge_answer_equivalence, not the autouse stub.

    The autouse `_stub_judges` fixture replaces
    `medcta_golden_eval.runner._judge_answer_equivalence` wholesale, so without
    restoring the real function here, patching `medcta_golden_eval.llm.judge_equivalence`
    below would be dead code and this test would pass even if the short
    circuit were deleted.
    """

    def _boom(*_args, **_kwargs):
        raise AssertionError("judge_equivalence should not be called for identical text")

    monkeypatch.setattr(
        "medcta_golden_eval.runner._judge_answer_equivalence", _real_judge_answer_equivalence
    )
    monkeypatch.setattr("medcta_golden_eval.llm.judge_equivalence", _boom)
    monkeypatch.setattr("medcta_golden_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
    case = _case(reference_tools=("OCR",))
    trace = run_case(
        case,
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="Same Hypothesis", reasoning="r1"),
            _response("FINAL_ANSWER", answer="Same Hypothesis", reasoning="r2"),
        ),
    )
    assert trace["status"] == "completed"
    assert len(trace["stage_transitions"]) == 1
    transition = trace["stage_transitions"][0]
    assert transition["answer_equivalence_score"] == 1.0
    assert transition["answer_changed"] is False


def test_equivalence_short_circuit_calls_judge_for_differing_text(monkeypatch):
    """Companion to the identical-text test: proves the judge IS invoked

    (with the two differing answers, in order) once text differs, so the
    short circuit is verified on both sides rather than just "never called".
    """
    calls = []

    def _spy(previous, current):
        calls.append((previous, current))
        return 0.2

    monkeypatch.setattr(
        "medcta_golden_eval.runner._judge_answer_equivalence", _real_judge_answer_equivalence
    )
    monkeypatch.setattr("medcta_golden_eval.llm.judge_equivalence", _spy)
    monkeypatch.setattr("medcta_golden_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
    case = _case(reference_tools=("OCR",))
    trace = run_case(
        case,
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="Hypothesis A", reasoning="r1"),
            _response("FINAL_ANSWER", answer="Hypothesis B", reasoning="r2"),
        ),
    )
    assert calls == [("Hypothesis A", "Hypothesis B")]
    transition = trace["stage_transitions"][0]
    assert transition["answer_equivalence_score"] == 0.2
    assert transition["answer_changed"] is True


def test_judge_answer_correctness_uses_single_merged_judge_with_all_accepted_answers(
    monkeypatch,
):
    """CALL_TOOL and FINAL_ANSWER steps both route through the one merged

    llm.judge_answer function/prompt (no more is_final_answer routing), and
    every accepted answer is passed through, not just the first — matching
    any one of them is sufficient.
    """
    calls = []

    def _spy(golds, pred):
        calls.append((golds, pred))
        return 0.6

    monkeypatch.setattr("medcta_golden_eval.llm.judge_answer", _spy)
    assert _real_judge_answer_correctness("x", ["gold-a", "gold-b"]) == 0.6
    assert calls == [(["gold-a", "gold-b"], "x")]


def test_correctness_cache_skips_judge_call_for_repeated_identical_answer_text(monkeypatch):
    """Exercises the real _judge_answer_correctness_cached, not the autouse stub.

    The autouse `_stub_judges` fixture replaces
    `medcta_golden_eval.runner._judge_answer_correctness` wholesale, so without
    restoring the real function here, patching llm.judge_answer below would
    be dead code and this test would pass even if the cache were deleted.
    All three steps (two CALL_TOOL, one FINAL_ANSWER) share the same
    hypothesis text, so they must share exactly one cache entry and one
    judge call — there's no longer a separate cache slot for the
    FINAL_ANSWER step, since one merged judge function/prompt is used
    throughout.
    """
    calls = []

    def _spy(golds, pred):
        calls.append((golds, pred))
        return 0.9

    monkeypatch.setattr(
        "medcta_golden_eval.runner._judge_answer_correctness", _real_judge_answer_correctness
    )
    monkeypatch.setattr("medcta_golden_eval.llm.judge_answer", _spy)
    case = _case(reference_tools=("OCR", "ImageDescription"))
    trace = run_case(
        case,
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="Same Hypothesis", reasoning="r1"),
            _response("CALL_TOOL", "ImageDescription", answer="Same Hypothesis", reasoning="r2"),
            _response("FINAL_ANSWER", answer="Same Hypothesis", reasoning="r3"),
        ),
    )
    assert trace["status"] == "completed"
    assert calls == [(["gold answer"], "Same Hypothesis")]
    assert [step["current_answer_score"] for step in trace["steps"]] == [0.9, 0.9, 0.9]
    assert trace["final_answer_score"] == 0.9


def test_correctness_cache_calls_judge_separately_for_differing_answer_text(monkeypatch):
    """Companion to the identical-text test: proves the judge IS invoked

    once per distinct answer, so the cache is verified on both sides rather
    than just "never called".
    """
    calls = []

    def _spy(golds, pred):
        calls.append((golds, pred))
        return 0.5

    monkeypatch.setattr(
        "medcta_golden_eval.runner._judge_answer_correctness", _real_judge_answer_correctness
    )
    monkeypatch.setattr("medcta_golden_eval.llm.judge_answer", _spy)
    case = _case(reference_tools=("OCR", "ImageDescription"))
    run_case(
        case,
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="Hypothesis A", reasoning="r1"),
            _response("CALL_TOOL", "ImageDescription", answer="Hypothesis B", reasoning="r2"),
            _response("FINAL_ANSWER", answer="Hypothesis C", reasoning="r3"),
        ),
    )
    assert calls == [
        (["gold answer"], "Hypothesis A"),
        (["gold answer"], "Hypothesis B"),
        (["gold answer"], "Hypothesis C"),
    ]


def test_correctness_cache_key_normalizes_whitespace_and_case(monkeypatch):
    calls = []

    def _spy(golds, pred):
        calls.append(pred)
        return 0.7

    monkeypatch.setattr(
        "medcta_golden_eval.runner._judge_answer_correctness", _real_judge_answer_correctness
    )
    monkeypatch.setattr("medcta_golden_eval.llm.judge_answer", _spy)
    case = _case(reference_tools=("OCR", "ImageDescription"))
    run_case(
        case,
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="Same   Hypothesis", reasoning="r1"),
            _response("CALL_TOOL", "ImageDescription", answer="same hypothesis", reasoning="r2"),
            _response("FINAL_ANSWER", answer="SAME HYPOTHESIS", reasoning="r3"),
        ),
    )
    assert len(calls) == 1


def _two_step_transition(prev_correct, new_correct, monkeypatch):
    monkeypatch.setattr(
        "medcta_golden_eval.runner._judge_answer_correctness",
        lambda answer, accepted, **_: (
            1.0 if (answer == "prev" and prev_correct) or (answer == "new" and new_correct) else 0.0
        ),
    )
    monkeypatch.setattr("medcta_golden_eval.runner._judge_answer_equivalence", lambda previous, current: 0.0)
    case = _case(reference_tools=("OCR",))
    trace = run_case(
        case,
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="prev", reasoning="r1"),
            _response("FINAL_ANSWER", answer="new", reasoning="r2"),
        ),
    )
    assert len(trace["stage_transitions"]) == 1
    return trace["stage_transitions"][0]


def test_transition_label_successful_revision(monkeypatch):
    transition = _two_step_transition(False, True, monkeypatch)
    assert transition["label"] == "successful_revision"
    assert transition["prev_correct"] is False
    assert transition["new_correct"] is True


def test_transition_label_missed_revision(monkeypatch):
    transition = _two_step_transition(False, False, monkeypatch)
    assert transition["label"] == "missed_revision"


def test_transition_label_overreaction(monkeypatch):
    transition = _two_step_transition(True, False, monkeypatch)
    assert transition["label"] == "overreaction"


def test_transition_label_kept_correct(monkeypatch):
    transition = _two_step_transition(True, True, monkeypatch)
    assert transition["label"] == "kept_correct"


def test_maintained_wrong_true_when_missed_revision_and_unchanged(monkeypatch):
    monkeypatch.setattr("medcta_golden_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 0.0)
    case = _case(reference_tools=("OCR",))
    trace = run_case(
        case,
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="same wrong answer", reasoning="r1"),
            _response("FINAL_ANSWER", answer="same wrong answer", reasoning="r2"),
        ),
    )
    transition = trace["stage_transitions"][0]
    assert transition["label"] == "missed_revision"
    assert transition["answer_changed"] is False
    assert transition["maintained_wrong"] is True


def test_maintained_wrong_false_when_missed_revision_but_changed(monkeypatch):
    monkeypatch.setattr("medcta_golden_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 0.0)
    monkeypatch.setattr("medcta_golden_eval.runner._judge_answer_equivalence", lambda previous, current: 0.0)
    case = _case(reference_tools=("OCR",))
    trace = run_case(
        case,
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="wrong A", reasoning="r1"),
            _response("FINAL_ANSWER", answer="wrong B", reasoning="r2"),
        ),
    )
    transition = trace["stage_transitions"][0]
    assert transition["label"] == "missed_revision"
    assert transition["answer_changed"] is True
    assert transition["maintained_wrong"] is False


def test_answer_stable_throughout_is_none_with_fewer_than_two_scored_steps(monkeypatch):
    monkeypatch.setattr("medcta_golden_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
    trace = run_case(
        _case(reference_tools=("OCR",)),
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR"),
            "not json",
            "still not json",
        ),
    )
    assert trace["status"] == "invalid_action"
    assert trace["stage_transitions"] == []
    assert trace["answer_stable_throughout"] is None


def test_answer_stable_throughout_true_when_no_transition_changes(monkeypatch):
    monkeypatch.setattr("medcta_golden_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
    case = _case(reference_tools=("OCR",))
    trace = run_case(
        case,
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="same answer", reasoning="r1"),
            _response("FINAL_ANSWER", answer="same answer", reasoning="r2"),
        ),
    )
    assert len(trace["stage_transitions"]) == 1
    assert trace["answer_stable_throughout"] is True


def test_answer_stable_throughout_false_when_a_transition_changes(monkeypatch):
    monkeypatch.setattr("medcta_golden_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
    monkeypatch.setattr("medcta_golden_eval.runner._judge_answer_equivalence", lambda previous, current: 0.0)
    case = _case(reference_tools=("OCR",))
    trace = run_case(
        case,
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="answer A", reasoning="r1"),
            _response("FINAL_ANSWER", answer="answer B", reasoning="r2"),
        ),
    )
    assert len(trace["stage_transitions"]) == 1
    assert trace["answer_stable_throughout"] is False


def test_revision_rate_metrics_wire_up_from_stage_transitions(monkeypatch):
    """Light wiring check that build_report's four revision-rate metrics
    (duplicated verbatim from medcta_eval.metrics, not derived as 1 - x)
    correctly aggregate the stage_transitions labels already covered by the
    test_transition_label_* tests above.
    """
    monkeypatch.setattr(
        "medcta_golden_eval.runner._judge_answer_correctness",
        lambda answer, accepted, **_: 1.0 if answer == "right" else 0.0,
    )
    case = _case(reference_tools=("OCR",))
    trace = run_case(
        case,
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="wrong", reasoning="r1"),
            _response("FINAL_ANSWER", answer="right", reasoning="r2"),
        ),
    )
    assert trace["stage_transitions"][0]["label"] == "successful_revision"
    report = build_report([trace], model="test")
    assert report["successful_revision_rate"] == {"rate": 1.0, "numerator": 1, "denominator": 1}
    assert report["missed_revision_rate"] == {"rate": 0.0, "numerator": 0, "denominator": 1}
    assert report["overreaction_rate"] == {"rate": None, "numerator": 0, "denominator": 0}
    assert report["kept_correct_rate"] == {"rate": None, "numerator": 0, "denominator": 0}


class PerfectBackend:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = 0

    def __call__(self, _system, user, image_url):
        self.calls += 1
        assert image_url == "https://example.test/image.jpg"
        case_id = next(
            case_id for case_id in self.mapping if f"CASE ID:\n{case_id}\n" in user
        )
        history = user.split("PRIOR MODEL ACTIONS:\n", 1)[1].split(
            "\n\nPRIOR REFERENCE", 1
        )[0]
        actions = json.loads(history)
        tool, answer = self.mapping[case_id]
        if not actions:
            return _response("CALL_TOOL", tool)
        return _response("FINAL_ANSWER", answer=answer)


def _write_payload(path: Path, *cases):
    path.write_text(json.dumps(_payload(*cases)), encoding="utf-8")


def test_builtin_backend_fails_before_inference_without_fixed_judge(monkeypatch):
    monkeypatch.setattr(pipeline_module, "JUDGE_MODEL", None)

    with pytest.raises(RuntimeError, match="MEDCTA_JUDGE_MODEL"):
        pipeline_module._select_backend(None, "candidate/model", "candidate-key")


def test_pipeline_writes_manifest_report_and_one_trace_per_case(tmp_path):
    data_path = tmp_path / "cases.json"
    _write_payload(data_path, _case("A"), _case("B"))
    backend = PerfectBackend({"A": ("OCR", "gold answer"), "B": ("OCR", "gold answer")})

    report, traces = run_pipeline(
        n=2,
        model="test-model",
        out_dir=tmp_path / "runs",
        cases_path=data_path,
        call_fn=backend,
        verbose=False,
    )
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "report.json").is_file()
    assert not (run_dir / "partial_report.json").exists()
    assert sorted(path.name for path in run_dir.glob("trace_*.json")) == [
        "trace_A.json",
        "trace_B.json",
    ]
    assert report["final_answer_accuracy"]["rate"] == 1.0
    assert [trace["case_id"] for trace in traces] == ["A", "B"]


def test_pipeline_resume_skips_verified_existing_traces(tmp_path):
    data_path = tmp_path / "cases.json"
    _write_payload(data_path, _case("A"), _case("B"))
    mapping = {"A": ("OCR", "gold answer"), "B": ("OCR", "gold answer")}
    first_backend = PerfectBackend(mapping)
    run_pipeline(
        n=2,
        model="test-model",
        out_dir=tmp_path / "runs",
        cases_path=data_path,
        call_fn=first_backend,
        verbose=False,
    )
    run_dir = next((tmp_path / "runs").iterdir())

    resumed_backend = PerfectBackend(mapping)
    report, traces = run_pipeline(
        n=2,
        model="test-model",
        resume_dir=run_dir,
        cases_path=data_path,
        call_fn=resumed_backend,
        verbose=False,
    )
    assert resumed_backend.calls == 0
    assert report["n_selected_cases"] == 2
    assert [trace["case_id"] for trace in traces] == ["A", "B"]


def test_pipeline_resume_rejects_model_provenance_mismatch(tmp_path):
    data_path = tmp_path / "cases.json"
    _write_payload(data_path, _case("A"))
    mapping = {"A": ("OCR", "gold answer")}
    run_pipeline(
        n=1,
        model="first-model",
        out_dir=tmp_path / "runs",
        cases_path=data_path,
        call_fn=PerfectBackend(mapping),
        verbose=False,
    )
    run_dir = next((tmp_path / "runs").iterdir())
    with pytest.raises(ValueError, match="resume provenance"):
        run_pipeline(
            n=1,
            model="other-model",
            resume_dir=run_dir,
            cases_path=data_path,
            call_fn=PerfectBackend(mapping),
            verbose=False,
        )


def test_pipeline_continues_after_one_case_inference_failure(tmp_path):
    data_path = tmp_path / "cases.json"
    _write_payload(data_path, _case("A"), _case("B"))

    class MixedBackend:
        def __call__(self, _system, user, _image):
            if "CASE ID:\nA\n" in user:
                raise RuntimeError("case A failed")
            if "PRIOR MODEL ACTIONS:\n[]" in user:
                return _response("CALL_TOOL", "OCR")
            return _response("FINAL_ANSWER", answer="gold answer")

    report, traces = run_pipeline(
        n=2,
        model="test-model",
        out_dir=tmp_path / "runs",
        cases_path=data_path,
        call_fn=MixedBackend(),
        verbose=False,
    )
    assert [trace["status"] for trace in traces] == ["inference_failure", "completed"]
    assert report["inference_failure_count"] == 1
    assert report["n_evaluable_cases"] == 1
    assert report["trace_status_counts"]["completed"] == 1


def test_interrupted_run_keeps_reproducible_partial_report(tmp_path):
    data_path = tmp_path / "cases.json"
    _write_payload(data_path, _case("A"), _case("B"))

    class InterruptBackend:
        def __call__(self, _system, user, _image):
            if "CASE ID:\nB\n" in user:
                raise KeyboardInterrupt()
            if "PRIOR MODEL ACTIONS:\n[]" in user:
                return _response("CALL_TOOL", "OCR")
            return _response("FINAL_ANSWER", answer="gold answer")

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            n=2,
            model="test-model",
            out_dir=tmp_path / "runs",
            cases_path=data_path,
            call_fn=InterruptBackend(),
            verbose=False,
        )

    run_dir = next((tmp_path / "runs").iterdir())
    partial_path = run_dir / "partial_report.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    assert partial["run_complete"] is False
    assert partial["n_planned_cases"] == 2
    assert partial["n_selected_cases"] == 1
    assert partial["missing_case_ids"] == ["B"]

    partial_path.unlink()
    regenerated, traces = report_existing_run(run_dir, verbose=False)
    assert partial_path.is_file()
    assert regenerated["report_source"] == "recomputed_from_saved_traces"
    assert regenerated["missing_case_ids"] == ["B"]
    assert [trace["case_id"] for trace in traces] == ["A"]


def test_call_json_sends_vision_input_through_shared_responses_call(monkeypatch):
    captured = {}

    def _spy(system, request_input, *, model, max_output_tokens, api_key=None):
        captured.update(
            system=system,
            request_input=request_input,
            model=model,
            max_output_tokens=max_output_tokens,
        )
        return '{"ok":true}'

    monkeypatch.setattr(llm, "call_responses_json", _spy)
    monkeypatch.setattr(llm, "resolve_image_url", lambda url, *, ttl_seconds: url)
    result = llm.call_json(
        "system",
        "user",
        "https://example.test/image.jpg",
        model="vision-model",
    )
    assert result == '{"ok":true}'
    assert captured["system"] == "system"
    assert captured["model"] == "vision-model"
    assert captured["request_input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "user"},
                {"type": "input_image", "image_url": "https://example.test/image.jpg"},
            ],
        }
    ]


def test_call_json_sends_plain_text_input_when_no_image(monkeypatch):
    captured = {}

    def _spy(system, request_input, *, model, max_output_tokens, api_key=None):
        captured.update(system=system, request_input=request_input, model=model)
        return "{}"

    monkeypatch.setattr(llm, "call_responses_json", _spy)
    llm.call_json("system", "user", None, model="text-model")
    assert captured["request_input"] == "user"
    assert captured["model"] == "text-model"


def test_run_judge_parses_and_clamps_score(monkeypatch):
    monkeypatch.setattr(llm, "JUDGE_MODEL", "gpt-5.4")
    monkeypatch.setattr(
        llm, "call_responses_json", lambda *_args, **_kwargs: '{"score": 1.5}'
    )
    assert llm._run_judge("system", "user") == 1.0

    monkeypatch.setattr(
        llm, "call_responses_json", lambda *_args, **_kwargs: '{"score": -0.5}'
    )
    assert llm._run_judge("system", "user") == 0.0


def test_run_judge_returns_none_on_missing_score_or_failure(monkeypatch):
    monkeypatch.setattr(llm, "call_responses_json", lambda *_args, **_kwargs: "{}")
    assert llm._run_judge("system", "user") is None

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(llm, "call_responses_json", _boom)
    assert llm._run_judge("system", "user") is None


def test_judge_answer_includes_every_accepted_answer_in_the_judge_prompt(monkeypatch):
    """judge_answer must forward all accepted answers, not just the first,

    since the gold answer for a case can have multiple valid phrasings and
    matching any single one of them is meant to be sufficient.
    """
    captured = {}

    def _spy(
        system,
        user,
        *,
        model,
        max_output_tokens,
        api_key=None,
        provider="openrouter",
        base_url=None,
        client_profile="candidate",
    ):
        captured.update(
            system=system,
            user=user,
            provider=provider,
            base_url=base_url,
            client_profile=client_profile,
        )
        return '{"score": 1.0}'

    monkeypatch.setattr(llm, "call_responses_json", _spy)
    monkeypatch.setattr(llm, "JUDGE_MODEL", "gpt-5.4")
    score = llm.judge_answer(["Liver mass", "Hepatic lesion"], "Hepatic lesion")
    assert score == 1.0
    assert "Liver mass" in captured["user"]
    assert "Hepatic lesion" in captured["user"]


def test_retry_delay_parser_supports_minutes_and_long_wait_cap():
    assert shared_llm._parse_retry_after_message("Please try again in 20m33.5s") == 1233.5
    assert shared_llm._parse_retry_after_message("try again in 4.25s") == 4.25
    assert 1233.5 > shared_llm.MAX_RETRY_WAIT_SECONDS
