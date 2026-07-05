"""
Offline test suite — no network, no LLM calls.

Run with:  cd MedRouteBench && python -m pytest staged_eval/tests -q
"""
import pytest

from staged_eval.schema import (
    ACTIONS, NONFINAL_ALLOWED, FINAL_ALLOWED,
    AgentOutput, safe_json_loads, validate,
)
from staged_eval.data import (
    load_cases, load_ground_truth,
    n_stages, revealed_pairs, stratified_sample, warn_if_degenerate_labels,
)
from staged_eval.prompts import build_user_prompt, SYSTEM_PROMPT, REPAIR_TEMPLATE
from staged_eval.metrics import (
    action_accuracy_by_stage, final_answer_accuracy, revision_correctness,
    premature_answer_rate, missed_revision_rate, abstention_rate, build_report,
)
from staged_eval.runner import run_case


# ── helpers ───────────────────────────────────────────────────────────────────

_SMOKE_N = 5


def _cases_and_gt():
    return load_cases(limit=_SMOKE_N), load_ground_truth()


# ── schema ────────────────────────────────────────────────────────────────────

def test_validate_good():
    out, err = validate({
        "action": "ANSWER", "answer": "yes", "confidence": 0.8,
        "reason_for_action": "clear evidence", "needed_information": None,
    })
    assert isinstance(out, AgentOutput) and err is None
    assert out.action == "ANSWER" and out.answer == "yes"


def test_validate_missing_key():
    _, err = validate({})
    assert err is not None and "missing key" in err


def test_validate_bad_action():
    _, err = validate({
        "action": "CALL_TOOL", "answer": None, "confidence": 0.5,
        "reason_for_action": "x", "needed_information": None,
    })
    assert err is not None and "invalid action" in err


def test_validate_unknown_answer():
    _, err = validate({
        "action": "ANSWER", "answer": "dunno", "confidence": 0.5,
        "reason_for_action": "x", "needed_information": None,
    })
    assert err is not None


def test_validate_normalises_case():
    out, err = validate({
        "action": "answer", "answer": "YES", "confidence": 0.9,
        "reason_for_action": "check", "needed_information": None,
    })
    assert err is None
    assert out.action == "ANSWER" and out.answer == "yes"


def test_validate_confidence_clamp():
    out, err = validate({
        "action": "ABSTAIN", "answer": None, "confidence": 1.5,
        "reason_for_action": "overshoot", "needed_information": None,
    })
    assert err is None and out.confidence == 1.0


def test_safe_json_loads_good():
    obj, err = safe_json_loads('{"ok": true}')
    assert obj == {"ok": True} and err is None


def test_safe_json_loads_bad():
    obj, err = safe_json_loads("not json{{{")
    assert obj is None and err is not None


# ── data ──────────────────────────────────────────────────────────────────────

def test_load_cases():
    cases, _ = _cases_and_gt()
    assert len(cases) == _SMOKE_N
    assert all("pmid" in c and "CONTEXTS" in c and "QUESTION" in c for c in cases)


def test_n_stages():
    cases, _ = _cases_and_gt()
    c = cases[0]
    assert n_stages(c) == len(c["CONTEXTS"]) + 1


def test_revealed_pairs_stage0_empty():
    cases, _ = _cases_and_gt()
    assert revealed_pairs(cases[0], 0) == []


def test_revealed_pairs_stage1_one_pair():
    cases, _ = _cases_and_gt()
    pairs = revealed_pairs(cases[0], 1)
    assert len(pairs) == 1
    assert isinstance(pairs[0][0], str) and isinstance(pairs[0][1], str)


def test_stratified_sample_size_and_diversity():
    cases = load_cases()
    gt = load_ground_truth()
    sample = stratified_sample(cases, gt, 15)
    assert len(sample) == 15
    labels = {gt.get(c["pmid"]) for c in sample}
    assert len(labels) > 1, "stratified sample should cover multiple labels"


# ── prompts ───────────────────────────────────────────────────────────────────

def test_prompt_nonfinal_no_final_marker():
    cases, _ = _cases_and_gt()
    c, total = cases[0], n_stages(cases[0])
    p = build_user_prompt(c, 1, total, None)
    assert "FINAL STAGE" not in p
    assert c["QUESTION"] in p


def test_prompt_final_marker_present():
    cases, _ = _cases_and_gt()
    c, total = cases[0], n_stages(cases[0])
    p = build_user_prompt(c, total - 1, total, None)
    assert "FINAL STAGE" in p
    assert "FOLLOW_UP is FORBIDDEN" in p


def test_prompt_reveals_labels():
    cases, _ = _cases_and_gt()
    c, total = cases[0], n_stages(cases[0])
    p = build_user_prompt(c, 2, total, None)
    assert "LABEL " in p


# ── metrics ───────────────────────────────────────────────────────────────────

_TOY = [{
    "pmid": "TEST", "gt": "yes", "n_stages": 3, "n_contexts": 2,
    "stages": [
        {"stage": 0, "is_final": False, "revealed_labels": [],
         "parsed": {"action": "FOLLOW_UP", "answer": None, "confidence": 0.3,
                    "reason_for_action": "need context", "needed_information": "results"},
         "parse_error": False, "raw": ""},
        {"stage": 1, "is_final": False, "revealed_labels": ["BACKGROUND"],
         "parsed": {"action": "ANSWER", "answer": "no", "confidence": 0.4,
                    "reason_for_action": "initial read", "needed_information": None},
         "parse_error": False, "raw": ""},
        {"stage": 2, "is_final": True, "revealed_labels": ["BACKGROUND", "RESULTS"],
         "parsed": {"action": "REVISE_ANSWER", "answer": "yes", "confidence": 0.85,
                    "reason_for_action": "results change conclusion", "needed_information": None},
         "parse_error": False, "raw": ""},
    ],
}]
_TOY_GT = {"TEST": "yes"}


def test_action_accuracy_all_correct():
    r = action_accuracy_by_stage(_TOY)
    assert r["by_role"]["nonfinal"] == 1.0
    assert r["by_role"]["final"] == 1.0


def test_final_answer_accuracy():
    r = final_answer_accuracy(_TOY, _TOY_GT)
    assert r["accuracy"] == 1.0 and r["committed"] == 1


def test_revision_correctness():
    r = revision_correctness(_TOY, _TOY_GT)
    assert r["n_revisions"] == 1 and r["correctness"] == 1.0


def test_premature_answer_rate():
    # Stage 1: ANSWER "no" (wrong) at non-final → premature
    assert premature_answer_rate(_TOY, _TOY_GT) == 1.0


def test_missed_revision_rate():
    # Prior answer "no" wrong, but agent revised → rate = 0.0
    r = missed_revision_rate(_TOY, _TOY_GT)
    assert r["rate"] == 0.0


def test_abstention_rate_zero():
    assert abstention_rate(_TOY) == 0.0


def test_build_report_keys():
    report = build_report(_TOY, _TOY_GT, model="test")
    expected = {
        "model", "n_cases", "sample_label_counts",
        "action_accuracy_by_stage", "final_answer_accuracy",
        "revision_correctness", "premature_answer_rate",
        "missed_revision_rate", "abstention_rate",
    }
    assert set(report.keys()) == expected


# ── runner (mocked LLM) ───────────────────────────────────────────────────────

def _stub_responses():
    """Cyclic stub: returns valid JSON strings for each stage call."""
    _bank = [
        '{"action":"FOLLOW_UP","answer":null,"confidence":0.0,"reason_for_action":"need more","needed_information":"background"}',
        '{"action":"ANSWER","answer":"yes","confidence":0.7,"reason_for_action":"partial evidence","needed_information":null}',
        '{"action":"FOLLOW_UP","answer":null,"confidence":0.0,"reason_for_action":"want results","needed_information":"results"}',
        '{"action":"ANSWER","answer":"yes","confidence":0.85,"reason_for_action":"clear results","needed_information":null}',
        '{"action":"ANSWER","answer":"yes","confidence":0.75,"reason_for_action":"all data reviewed","needed_information":null}',
        '{"action":"ANSWER","answer":"yes","confidence":0.8,"reason_for_action":"comprehensive","needed_information":null}',
        '{"action":"ANSWER","answer":"yes","confidence":0.9,"reason_for_action":"done","needed_information":null}',
    ]
    idx = [0]

    def stub(system, user):
        r = _bank[idx[0] % len(_bank)]
        idx[0] += 1
        return r

    return stub


def test_run_case_mocked_structure():
    cases, gt = _cases_and_gt()
    case = cases[0]
    trace = run_case(case, gt.get(case["pmid"]), call_fn=_stub_responses())
    assert trace["pmid"] == case["pmid"]
    assert len(trace["stages"]) == n_stages(case)
    assert trace["stages"][-1]["is_final"] is True


def test_run_case_mocked_no_parse_errors():
    cases, gt = _cases_and_gt()
    trace = run_case(cases[0], gt.get(cases[0]["pmid"]), call_fn=_stub_responses())
    assert all(not s["parse_error"] for s in trace["stages"])
