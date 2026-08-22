import json
from pathlib import Path

import pytest

import medcta_eval.pipeline as pipeline_module
from medcta_eval import llm
import shared.llm as shared_llm
from medcta_eval.adapter import ALLOWED_TOOLS, SOURCE_REVISION, STARTER_CASE_IDS, adapt_raw_dataset
from medcta_eval.config import DATA_PATH, PROJECT_DIR
from medcta_eval.data import load_dataset, validate_dataset
from medcta_eval.metrics import build_report
from medcta_eval.pipeline import report_existing_run, run_pipeline
from medcta_eval.prompts import build_user_prompt
from medcta_eval.runner import _judge_answer_correctness as _real_judge_answer_correctness
from medcta_eval.runner import _judge_answer_equivalence as _real_judge_answer_equivalence
from medcta_eval.runner import run_case
from medcta_eval.schema import (
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
        "medcta_eval.runner._judge_answer_correctness",
        lambda answer, accepted, **_: 1.0 if answer in (accepted or []) else 0.0,
    )
    monkeypatch.setattr(
        "medcta_eval.runner._judge_answer_equivalence",
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
    monkeypatch.setattr("medcta_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
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


def test_wrong_tool_continues_with_expected_reference_observation():
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
    assert trace["trajectory_exact_match"] is False
    first = trace["steps"][0]
    assert first["reference_tool_match"] is False
    assert first["reference_observation_revealed"]["reference_tool_name"] == "OCR"
    second_prompt = trace["steps"][1]["user_prompt"]
    assert "observation-0-OCR" in second_prompt
    assert '"tool_name": "ImageDescription"' in second_prompt


def test_premature_finalization_stops_and_keeps_answer_accuracy_separate(monkeypatch):
    monkeypatch.setattr("medcta_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
    trace = run_case(
        _case(reference_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(_response("FINAL_ANSWER", answer="gold answer")),
    )
    assert trace["status"] == "premature_finalization"
    assert trace["attempted_steps"] == 1
    assert trace["reference_step_count"] == 3
    assert trace["final_answer_match"] is True
    assert trace["trajectory_exact_match"] is False
    assert trace["stage_transitions"] == []
    assert trace["answer_stable_throughout"] is None


def test_call_at_reference_final_step_is_missed_finalization():
    trace = run_case(
        _case(reference_tools=("OCR",), available_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR"),
            _response("CALL_TOOL", "ImageDescription"),
        ),
    )
    assert trace["status"] == "missed_finalization"
    assert trace["model_tool_sequence"] == ["OCR", "ImageDescription"]
    assert trace["steps"][-1]["reference_observation_revealed"] is None


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


def test_requested_metrics_use_the_locked_denominators(monkeypatch):
    monkeypatch.setattr("medcta_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
    perfect = run_case(
        _case("perfect", ("OCR",), ("OCR", "ImageDescription")),
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR"),
            _response("FINAL_ANSWER", answer="gold answer"),
        ),
    )
    premature = run_case(
        _case("premature", ("OCR", "ImageDescription"), ("OCR", "ImageDescription")),
        call_fn=_scripted(_response("FINAL_ANSWER", answer="gold answer")),
    )
    missed = run_case(
        _case("missed", ("OCR",), ("OCR", "ImageDescription")),
        call_fn=_scripted(
            _response("CALL_TOOL", "ImageDescription"),
            _response("CALL_TOOL", "OCR"),
        ),
    )
    report = build_report([perfect, premature, missed], model="test")

    assert report["next_tool_accuracy"] == {
        "rate": 1 / 4,
        "numerator": 1,
        "denominator": 4,
    }
    assert report["trajectory_step_accuracy"] == {
        "rate": 2 / 7,
        "numerator": 2,
        "denominator": 7,
    }
    assert report["premature_finalization_rate"]["numerator"] == 1
    assert report["missed_finalization_rate"] == {
        "rate": 0.5,
        "numerator": 1,
        "denominator": 2,
    }
    assert report["tool_precision"] == {
        "rate": 1 / 3,
        "numerator": 1,
        "denominator": 3,
    }
    assert report["unnecessary_tool_rate"] == {
        "rate": 1 / 3,
        "numerator": 1,
        "denominator": 3,
    }
    assert report["trajectory_exact_match_rate"]["numerator"] == 1
    assert report["final_answer_accuracy"]["numerator"] == 2
    assert report["invalid_action_rate"] == {
        "rate": 0.0,
        "numerator": 0,
        "denominator": 5,
    }
    assert report["inference_failure_count"] == 0


def _score_by_answer_text(mapping):
    """Judge stub: score looked up by the literal answer text, default 0.0."""
    return lambda answer, accepted, **_: mapping.get(answer, 0.0)


def test_mean_score_improvement_and_first_step_answer_score(monkeypatch):
    monkeypatch.setattr(
        "medcta_eval.runner._judge_answer_correctness",
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

    # a: (0.8 - 0.2) / 1 tool call = 0.6 ; b: (1.0 - 0.0) / 2 tool calls = 0.5
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


def test_mean_score_improvement_excludes_zero_tool_call_cases(monkeypatch):
    monkeypatch.setattr(
        "medcta_eval.runner._judge_answer_correctness",
        _score_by_answer_text({"X": 0.5, "N0": 0.1, "N1": 0.9}),
    )
    immediate = run_case(
        _case("immediate", reference_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(_response("FINAL_ANSWER", answer="X")),
    )
    normal = run_case(
        _case("normal", reference_tools=("OCR",)),
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="N0"),
            _response("FINAL_ANSWER", answer="N1"),
        ),
    )
    assert immediate["status"] == "premature_finalization"
    assert immediate["model_tool_sequence"] == []

    report = build_report([immediate, normal], model="test")
    # immediate has tools_called == 0, so it's excluded from the mean entirely
    # (not counted as a 0.0 delta) -- only "normal"'s (0.9-0.1)/1 = 0.8 counts.
    assert report["mean_score_improvement"] == {
        "mean_score": 0.8,
        "n_scored": 1,
        "n_evaluable": 2,
    }


def test_premature_finalization_progress_uses_a_rate_not_raw_step_index(monkeypatch):
    monkeypatch.setattr(
        "medcta_eval.runner._judge_answer_correctness",
        _score_by_answer_text({"E": 0.5, "L": 0.5, "gold answer": 1.0}),
    )
    early = run_case(
        _case("early", reference_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(_response("FINAL_ANSWER", answer="E")),
    )
    late = run_case(
        _case("late", reference_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR"),
            _response("FINAL_ANSWER", answer="L"),
        ),
    )
    completed = run_case(
        _case("completed", reference_tools=("OCR",)),
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR"),
            _response("FINAL_ANSWER", answer="gold answer"),
        ),
    )
    assert early["status"] == "premature_finalization"
    assert late["status"] == "premature_finalization"
    assert completed["status"] == "completed"

    report = build_report([early, late, completed], model="test")
    # early bails at step 0 of 2 reference tools -> 0/2 = 0.0
    # late bails at step 1 of 2 reference tools (1 already called) -> 1/2 = 0.5
    # completed is excluded from the cohort entirely (not counted as 0 or 1)
    assert report["premature_finalization_progress"] == {
        "mean_rate": round((0.0 + 0.5) / 2, 4),
        "n_cases": 2,
    }


def test_premature_finalization_mean_score_restricted_to_that_cohort(monkeypatch):
    monkeypatch.setattr(
        "medcta_eval.runner._judge_answer_correctness",
        _score_by_answer_text({"gold answer": 1.0, "bad guess": 0.3}),
    )
    completed = run_case(
        _case("completed", reference_tools=("OCR",)),
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR"),
            _response("FINAL_ANSWER", answer="gold answer"),
        ),
    )
    premature = run_case(
        _case("premature", reference_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(_response("FINAL_ANSWER", answer="bad guess")),
    )
    report = build_report([completed, premature], model="test")

    # final_answer_mean_score spans both cases (completed + premature);
    # premature_finalization_mean_score is restricted to just the premature one.
    assert report["final_answer_mean_score"]["n_scored"] == 2
    assert report["premature_finalization_mean_score"] == {
        "mean_score": 0.3,
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
    assert report["invalid_action_rate"] == {
        "rate": 0.5,
        "numerator": 1,
        "denominator": 2,
    }
    assert report["inference_failure_count"] == 1
    assert report["inference_failure_case_count"] == 1
    assert report["n_evaluable_cases"] == 1
    assert report["repair_count"] == 1


def test_inference_only_run_has_no_routing_rate_denominators():
    def fail(*_args):
        raise RuntimeError("provider unavailable")

    failed = run_case(_case("failed"), call_fn=fail)
    report = build_report([failed], model="test")

    assert report["n_selected_cases"] == 1
    assert report["n_evaluable_cases"] == 0
    assert report["inference_failure_count"] == 1
    for metric in (
        "next_tool_accuracy",
        "trajectory_step_accuracy",
        "premature_finalization_rate",
        "missed_finalization_rate",
        "tool_precision",
        "unnecessary_tool_rate",
        "trajectory_exact_match_rate",
        "final_answer_accuracy",
        "invalid_action_rate",
        "stage_answer_accuracy",
        "successful_revision_rate",
        "missed_revision_rate",
        "overreaction_rate",
        "kept_correct_rate",
        "answer_change_rate",
        "maintenance_rate",
        "maintained_wrong_rate",
        "answer_stability",
        "premature_finalization_wrong",
        "early_correct_finalization",
        "unnecessary_tool_calls",
    ):
        assert report[metric] == {"rate": None, "numerator": 0, "denominator": 0}
    for metric in ("final_answer_mean_score", "first_step_answer_score", "mean_score_improvement"):
        assert report[metric] == {"mean_score": None, "n_scored": 0, "n_evaluable": 0}
    assert report["premature_finalization_progress"] == {"mean_rate": None, "n_cases": 0}
    assert report["premature_finalization_mean_score"] == {
        "mean_score": None,
        "n_scored": 0,
        "n_evaluable": 0,
    }


def test_inference_failure_does_not_dilute_evaluable_case_metrics(monkeypatch):
    monkeypatch.setattr("medcta_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
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
    assert report["next_tool_accuracy"]["rate"] == 1.0
    assert report["next_tool_accuracy"]["denominator"] == 1
    assert report["final_answer_accuracy"]["rate"] == 1.0
    assert report["final_answer_accuracy"]["denominator"] == 1


def test_equivalence_short_circuit_skips_judge_call_for_identical_text(monkeypatch):
    """Exercises the real _judge_answer_equivalence, not the autouse stub.

    The autouse `_stub_judges` fixture replaces
    `medcta_eval.runner._judge_answer_equivalence` wholesale, so without
    restoring the real function here, patching `medcta_eval.llm.judge_equivalence`
    below would be dead code and this test would pass even if the short
    circuit were deleted.
    """

    def _boom(*_args, **_kwargs):
        raise AssertionError("judge_equivalence should not be called for identical text")

    monkeypatch.setattr(
        "medcta_eval.runner._judge_answer_equivalence", _real_judge_answer_equivalence
    )
    monkeypatch.setattr("medcta_eval.llm.judge_equivalence", _boom)
    monkeypatch.setattr("medcta_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
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
        "medcta_eval.runner._judge_answer_equivalence", _real_judge_answer_equivalence
    )
    monkeypatch.setattr("medcta_eval.llm.judge_equivalence", _spy)
    monkeypatch.setattr("medcta_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
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

    monkeypatch.setattr("medcta_eval.llm.judge_answer", _spy)
    assert _real_judge_answer_correctness("x", ["gold-a", "gold-b"]) == 0.6
    assert calls == [(["gold-a", "gold-b"], "x")]


def test_correctness_cache_skips_judge_call_for_repeated_identical_answer_text(monkeypatch):
    """Exercises the real _judge_answer_correctness_cached, not the autouse stub.

    The autouse `_stub_judges` fixture replaces
    `medcta_eval.runner._judge_answer_correctness` wholesale, so without
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
        "medcta_eval.runner._judge_answer_correctness", _real_judge_answer_correctness
    )
    monkeypatch.setattr("medcta_eval.llm.judge_answer", _spy)
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
        "medcta_eval.runner._judge_answer_correctness", _real_judge_answer_correctness
    )
    monkeypatch.setattr("medcta_eval.llm.judge_answer", _spy)
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
        "medcta_eval.runner._judge_answer_correctness", _real_judge_answer_correctness
    )
    monkeypatch.setattr("medcta_eval.llm.judge_answer", _spy)
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
        "medcta_eval.runner._judge_answer_correctness",
        lambda answer, accepted, **_: (
            1.0 if (answer == "prev" and prev_correct) or (answer == "new" and new_correct) else 0.0
        ),
    )
    monkeypatch.setattr("medcta_eval.runner._judge_answer_equivalence", lambda previous, current: 0.0)
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
    monkeypatch.setattr("medcta_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 0.0)
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
    monkeypatch.setattr("medcta_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 0.0)
    monkeypatch.setattr("medcta_eval.runner._judge_answer_equivalence", lambda previous, current: 0.0)
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
    monkeypatch.setattr("medcta_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
    trace = run_case(
        _case(reference_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(_response("FINAL_ANSWER", answer="gold answer")),
    )
    assert trace["status"] == "premature_finalization"
    assert trace["stage_transitions"] == []
    assert trace["answer_stable_throughout"] is None


def test_answer_stable_throughout_true_when_no_transition_changes(monkeypatch):
    monkeypatch.setattr("medcta_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
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
    monkeypatch.setattr("medcta_eval.runner._judge_answer_correctness", lambda answer, accepted, **_: 1.0)
    monkeypatch.setattr("medcta_eval.runner._judge_answer_equivalence", lambda previous, current: 0.0)
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


def test_stage_answer_accuracy_counts_all_scored_steps(monkeypatch):
    monkeypatch.setattr(
        "medcta_eval.runner._judge_answer_correctness",
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
    report = build_report([trace], model="test")
    assert report["stage_answer_accuracy"] == {"rate": 0.5, "numerator": 1, "denominator": 2}


def test_premature_finalization_wrong_and_early_correct_finalization_partition(monkeypatch):
    monkeypatch.setattr(
        "medcta_eval.runner._judge_answer_correctness",
        lambda answer, accepted, **_: 1.0 if answer == "gold answer" else 0.0,
    )
    wrong = run_case(
        _case("wrong", reference_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(_response("FINAL_ANSWER", answer="bad guess")),
    )
    right = run_case(
        _case("right", reference_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(_response("FINAL_ANSWER", answer="gold answer")),
    )
    report = build_report([wrong, right], model="test")
    assert report["premature_finalization_wrong"] == {"rate": 0.5, "numerator": 1, "denominator": 2}
    assert report["early_correct_finalization"] == {"rate": 0.5, "numerator": 1, "denominator": 2}


def test_unnecessary_tool_calls_counts_tool_calls_despite_correct_answer(monkeypatch):
    monkeypatch.setattr(
        "medcta_eval.runner._judge_answer_correctness",
        lambda answer, accepted, **_: 1.0 if answer == "correct" else 0.0,
    )
    case = _case(reference_tools=("OCR", "ImageDescription"))
    trace = run_case(
        case,
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="correct", reasoning="already know it"),
            _response("CALL_TOOL", "ImageDescription", answer="correct", reasoning="still checking"),
            _response("FINAL_ANSWER", answer="correct", reasoning="confirmed"),
        ),
    )
    report = build_report([trace], model="test")
    # Step 0 (OCR) scores correct=True but has no revealed evidence yet, so it's
    # excluded entirely (see test below). Step 1 (ImageDescription) has one piece
    # of evidence, scores correct, and still chose CALL_TOOL: counts as unnecessary.
    # Step 2 (FINAL_ANSWER) has evidence and scores correct, but didn't call a tool.
    assert report["unnecessary_tool_calls"] == {"rate": 0.5, "numerator": 1, "denominator": 2}


def test_unnecessary_tool_calls_excludes_zero_evidence_step_zero_guess(monkeypatch):
    """A step-0 guess made before any evidence is revealed is a lucky guess,

    not a sign the model was ready to stop — it must not count toward
    unnecessary_tool_calls even when the judge marks it correct, or a model
    that dutifully follows the entire reference tool sequence would be
    flagged as making "unnecessary" calls purely for guessing right early.
    """
    monkeypatch.setattr(
        "medcta_eval.runner._judge_answer_correctness",
        lambda answer, accepted, **_: 1.0 if answer == "correct" else 0.0,
    )
    case = _case(reference_tools=("OCR",))
    trace = run_case(
        case,
        call_fn=_scripted(
            _response("CALL_TOOL", "OCR", answer="correct", reasoning="guessing before evidence"),
            _response("FINAL_ANSWER", answer="correct", reasoning="confirmed"),
        ),
    )
    assert trace["steps"][0]["evidence_shown"] == []
    report = build_report([trace], model="test")
    assert report["unnecessary_tool_calls"] == {"rate": 0.0, "numerator": 0, "denominator": 1}


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
    assert report["trajectory_exact_match_rate"]["rate"] == 1.0
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
    assert report["n_completed_cases"] == 1


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

    def _spy(system, user, *, model, max_output_tokens, api_key=None):
        captured.update(system=system, user=user)
        return '{"score": 1.0}'

    monkeypatch.setattr(llm, "call_responses_json", _spy)
    score = llm.judge_answer(["Liver mass", "Hepatic lesion"], "Hepatic lesion")
    assert score == 1.0
    assert "Liver mass" in captured["user"]
    assert "Hepatic lesion" in captured["user"]


def test_retry_delay_parser_supports_minutes_and_long_wait_cap():
    assert shared_llm._parse_retry_after_message("Please try again in 20m33.5s") == 1233.5
    assert shared_llm._parse_retry_after_message("try again in 4.25s") == 4.25
    assert 1233.5 > shared_llm.MAX_RETRY_WAIT_SECONDS
