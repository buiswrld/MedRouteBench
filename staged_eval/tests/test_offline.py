"""Offline tests for the fixed two-stage PubMedQA revision experiment."""

import json

import pytest

from staged_eval.data import (
    eligible_cases,
    load_cases,
    load_ground_truth,
    normalized_labels,
    split_evidence,
    stratified_sample,
)
from staged_eval.metrics import (
    build_report,
    final_abstention_rate,
    final_answer_accuracy,
    kept_correct_rate,
    maintenance_rate,
    missed_revision_rate,
    overreaction_rate,
    stage1_answer_accuracy,
    successful_revision_rate,
)
from staged_eval.pipeline import run_pipeline
from staged_eval.prompts import (
    STAGE1_SYSTEM_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    build_user_prompt,
)
from staged_eval.runner import run_case
from staged_eval.schema import ACTIONS, AgentOutput, safe_json_loads, validate


def _case(
    pmid="TEST",
    contexts=None,
    labels=None,
    **extra,
):
    return {
        "pmid": pmid,
        "QUESTION": "Does the intervention improve outcomes?",
        "CONTEXTS": contexts
        or ["Background evidence.", "Methods evidence.", "Result evidence."],
        "LABELS": labels or ["BACKGROUND", "METHODS", "RESULTS"],
        **extra,
    }


def _raw(action, answer, confidence=0.8, reason="because the evidence supports it"):
    return {
        "action": action,
        "answer": answer,
        "confidence": confidence,
        "reason_for_action": reason,
    }


def _output(action, answer, confidence=0.8, reason="because the evidence supports it"):
    return json.dumps(_raw(action, answer, confidence, reason))


def _trace(pmid, gold, stage1, action2, answer2, label):
    return {
        "pmid": pmid,
        "pubmedqa_gold_label": gold,
        "stage1_model_output": {
            "raw": "",
            "parsed": {"action": "ANSWER", "answer": stage1},
            "valid": True,
            "repaired": False,
            "validation_error": None,
        },
        "stage2_model_output": {
            "raw": "",
            "parsed": {"action": action2, "answer": answer2},
            "valid": True,
            "repaired": False,
            "validation_error": None,
        },
        "status": "completed",
        "label": label,
    }


# Schema and cross-stage consistency


def test_action_ontology_is_fixed_two_stage_contract():
    assert ACTIONS == ["ANSWER", "KEEP_ANSWER", "REVISE_ANSWER", "ABSTAIN"]


def test_validate_normalizes_stage1_output():
    output, error = validate(_raw("answer", "YES"), stage=1)
    assert error is None
    assert output == AgentOutput(
        action="ANSWER",
        answer="yes",
        confidence=0.8,
        reason_for_action="because the evidence supports it",
    )


@pytest.mark.parametrize("key", ["confidence", "reason_for_action"])
def test_validate_requires_confidence_and_reason(key):
    raw = _raw("ANSWER", "yes")
    del raw[key]
    _, error = validate(raw, stage=1)
    assert f"missing key '{key}'" in error


def test_validate_rejects_non_numeric_confidence():
    _, error = validate(_raw("ANSWER", "yes", confidence="high"), stage=1)
    assert "confidence not a number" in error


def test_validate_rejects_empty_reason():
    _, error = validate(_raw("ANSWER", "yes", reason="   "), stage=1)
    assert "empty reason_for_action" in error


def test_validate_clamps_confidence():
    output, _ = validate(_raw("ANSWER", "yes", confidence=1.7), stage=1)
    assert output.confidence == 1.0


@pytest.mark.parametrize("action", ["KEEP_ANSWER", "REVISE_ANSWER", "ABSTAIN"])
def test_stage1_requires_answer_action(action):
    _, error = validate(_raw(action, None), stage=1)
    assert "Stage 1 action must be ANSWER" in error


def test_stage1_requires_concrete_answer():
    _, error = validate(_raw("ANSWER", None), stage=1)
    assert "requires yes, no, or maybe" in error


def test_stage2_keep_must_match_stage1():
    output, error = validate(
        _raw("KEEP_ANSWER", "no"),
        stage=2,
        prior_answer="no",
    )
    assert error is None and output.answer == "no"

    _, error = validate(
        _raw("KEEP_ANSWER", "yes"),
        stage=2,
        prior_answer="no",
    )
    assert "match" in error


def test_stage2_revision_must_change_answer():
    output, error = validate(
        _raw("REVISE_ANSWER", "yes"),
        stage=2,
        prior_answer="no",
    )
    assert error is None and output.answer == "yes"

    _, error = validate(
        _raw("REVISE_ANSWER", "no"),
        stage=2,
        prior_answer="no",
    )
    assert "different" in error


def test_stage2_abstain_requires_null():
    output, error = validate(
        _raw("ABSTAIN", None),
        stage=2,
        prior_answer="maybe",
    )
    assert error is None and output.answer is None

    _, error = validate(
        _raw("ABSTAIN", "maybe"),
        stage=2,
        prior_answer="maybe",
    )
    assert "answer null" in error


def test_safe_json_loads_never_raises():
    assert safe_json_loads('{"ok": true}') == ({"ok": True}, None)
    parsed, error = safe_json_loads("not json")
    assert parsed is None and error.startswith("json_parse:")


# Evidence split


def test_normalized_labels_are_readable_unique_and_aligned():
    case = _case(
        contexts=["a", "b", "c", "d"],
        labels=[" study methods ", "study-methods", "", None],
    )
    assert normalized_labels(case) == [
        "STUDY METHODS",
        "STUDY METHODS 2",
        "SECTION 3",
        "SECTION 4",
    ]


def test_split_uses_first_label_containing_result():
    case = _case(
        contexts=["background", "methods", "primary result", "conclusion"],
        labels=["BACKGROUND", "METHODS", "Primary Results", "CONCLUSIONS"],
    )
    split = split_evidence(case)
    assert split["strategy"] == "first_results_section"
    assert [item["context"] for item in split["stage1_evidence"]] == [
        "background",
        "methods",
    ]
    assert [item["context"] for item in split["stage2_added_evidence"]] == [
        "primary result",
        "conclusion",
    ]


def test_split_returns_none_without_results_section():
    case = _case(
        contexts=["one", "two", "three", "four", "five"],
        labels=["A", "B", "C", "D", "E"],
    )
    assert split_evidence(case) is None


def test_results_at_first_section_is_excluded():
    assert split_evidence(
        _case(contexts=["results", "discussion"], labels=["RESULTS", "DISCUSSION"])
    ) is None


def test_empty_results_context_is_excluded():
    assert split_evidence(
        _case(
            contexts=["background", "methods", ""],
            labels=["BACKGROUND", "METHODS", "RESULTS"],
        )
    ) is None


@pytest.mark.parametrize(
    "contexts,labels",
    [
        (["only one"], ["RESULTS"]),
        (["background", ""], ["BACKGROUND", "RESULTS"]),
        (["", "results"], ["BACKGROUND", "RESULTS"]),
    ],
)
def test_split_skips_cases_without_two_nonempty_evidence_groups(contexts, labels):
    assert split_evidence(_case(contexts=contexts, labels=labels)) is None


def test_eligible_cases_requires_official_gold_and_valid_split():
    cases = [
        _case("A"),
        _case("B", contexts=["one"], labels=["RESULTS"]),
        _case("C"),
    ]
    assert [case["pmid"] for case in eligible_cases(cases, {"A": "yes", "B": "no"})] == [
        "A"
    ]


def test_real_loader_and_stratified_sample_are_gold_labelled():
    cases = load_cases()
    gold = load_ground_truth()
    eligible = eligible_cases(cases, gold)
    sample = stratified_sample(eligible, gold, min(6, len(eligible)))
    assert sample
    assert all(gold[case["pmid"]] in {"yes", "no", "maybe"} for case in sample)


def test_fixture_fallback_must_be_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "staged_eval.data.TEST_SET_PATH",
        tmp_path / "missing-test.json",
    )
    monkeypatch.setattr(
        "staged_eval.data.ORI_PQAL_PATH",
        tmp_path / "missing-cases.json",
    )
    monkeypatch.setattr(
        "staged_eval.data.GROUND_TRUTH_PATH",
        tmp_path / "missing-ground-truth.json",
    )

    with pytest.raises(FileNotFoundError, match="--use-fixtures"):
        load_cases()
    with pytest.raises(FileNotFoundError, match="--use-fixtures"):
        load_ground_truth()

    fixture_cases = load_cases(use_fixtures=True)
    fixture_gold = load_ground_truth(use_fixtures=True)
    assert fixture_cases
    assert all(case["pmid"] in fixture_gold for case in fixture_cases)


# Prompt privacy and fixed stages


def test_stage1_prompt_contains_only_preliminary_evidence():
    case = _case(contexts=["PRELIM", "NEW RESULT"], labels=["BACKGROUND", "RESULTS"])
    prompt = build_user_prompt(case, 1, split_evidence(case))
    assert "PRELIM" in prompt
    assert "NEW RESULT" not in prompt
    assert "ANSWER" in prompt


def test_stage2_prompt_contains_prior_output_and_full_context():
    case = _case(contexts=["PRELIM", "NEW RESULT"], labels=["BACKGROUND", "RESULTS"])
    prompt = build_user_prompt(
        case,
        2,
        split_evidence(case),
        {"action": "ANSWER", "answer": "no"},
    )
    assert "PRELIM" in prompt and "NEW RESULT" in prompt
    assert '"answer": "no"' in prompt
    assert all(action in prompt for action in ("KEEP_ANSWER", "REVISE_ANSWER", "ABSTAIN"))


def test_prompts_never_expose_forbidden_pubmedqa_fields():
    case = _case(
        contexts=["visible preliminary", "visible result"],
        labels=["BACKGROUND", "RESULTS"],
        LONG_ANSWER="SECRET_LONG_ANSWER",
        final_decision="SECRET_FINAL_DECISION",
        test_ground_truth="SECRET_TEST_GROUND_TRUTH",
    )
    split = split_evidence(case)
    prompts = [
        STAGE1_SYSTEM_PROMPT,
        STAGE2_SYSTEM_PROMPT,
        build_user_prompt(case, 1, split),
        build_user_prompt(case, 2, split, {"action": "ANSWER", "answer": "yes"}),
    ]
    for prompt in prompts:
        assert "SECRET_LONG_ANSWER" not in prompt
        assert "SECRET_FINAL_DECISION" not in prompt
        assert "SECRET_TEST_GROUND_TRUTH" not in prompt


# Runner and trace bookkeeping


def test_run_case_creates_requested_trace_and_successful_revision_label():
    responses = iter([_output("ANSWER", "no"), _output("REVISE_ANSWER", "yes")])
    trace = run_case(_case(), "yes", call_fn=lambda _system, _user: next(responses))
    required = {
        "pmid",
        "pubmedqa_gold_label",
        "stage1_evidence_shown",
        "stage1_model_output",
        "stage2_added_evidence",
        "stage2_full_context_shown",
        "stage2_model_output",
        "label",
    }
    assert required <= trace.keys()
    assert trace["status"] == "completed"
    assert trace["label"] == "successful_revision"
    assert trace["stage1_model_output"]["parsed"]["action"] == "ANSWER"
    assert trace["stage2_model_output"]["parsed"]["action"] == "REVISE_ANSWER"


def test_conflicting_outputs_are_repaired_once():
    responses = iter(
        [
            _output("KEEP_ANSWER", "no"),
            _output("ANSWER", "no"),
            _output("KEEP_ANSWER", "yes"),
            _output("KEEP_ANSWER", "no"),
        ]
    )
    trace = run_case(_case(), "no", call_fn=lambda _system, _user: next(responses))
    assert trace["status"] == "completed"
    assert trace["stage1_model_output"]["repaired"] is True
    assert trace["stage2_model_output"]["repaired"] is True
    assert trace["label"] == "kept_correct"


def test_twice_invalid_stage1_is_marked_invalid_and_stops():
    calls = []

    def stub(_system, _user):
        calls.append(1)
        return _output("ABSTAIN", None)

    trace = run_case(_case(), "yes", call_fn=stub)
    assert len(calls) == 2
    assert trace["status"] == "invalid_stage1"
    assert trace["label"] == "invalid"
    assert trace["stage2_model_output"] is None


def test_unsplittable_case_is_skipped_without_model_call():
    trace = run_case(
        _case(contexts=["only"], labels=["RESULTS"]),
        "yes",
        call_fn=lambda *_: pytest.fail("model must not be called"),
    )
    assert trace["status"] == "skipped_unsplittable"
    assert trace["label"] == "skipped"


@pytest.mark.parametrize(
    "stage1,stage2_action,stage2_answer,gold,expected",
    [
        ("no", "REVISE_ANSWER", "yes", "yes", "successful_revision"),
        ("no", "REVISE_ANSWER", "maybe", "yes", "missed_revision"),
        ("yes", "REVISE_ANSWER", "no", "yes", "overreaction"),
        ("yes", "KEEP_ANSWER", "yes", "yes", "kept_correct"),
        ("maybe", "ABSTAIN", None, "yes", "abstention"),
    ],
)
def test_trace_classification(stage1, stage2_action, stage2_answer, gold, expected):
    responses = iter([_output("ANSWER", stage1), _output(stage2_action, stage2_answer)])
    trace = run_case(_case(), gold, call_fn=lambda _system, _user: next(responses))
    assert trace["label"] == expected


# Metrics


TRACES = [
    _trace("1", "yes", "no", "REVISE_ANSWER", "yes", "successful_revision"),
    _trace("2", "yes", "no", "REVISE_ANSWER", "maybe", "missed_revision"),
    _trace("3", "yes", "no", "ABSTAIN", None, "abstention"),
    _trace("4", "yes", "yes", "REVISE_ANSWER", "no", "overreaction"),
    _trace("5", "yes", "yes", "KEEP_ANSWER", "yes", "kept_correct"),
    _trace("6", "yes", "no", "KEEP_ANSWER", "no", "missed_revision"),
]


@pytest.mark.parametrize(
    "metric,expected_rate,expected_denominator",
    [
        (stage1_answer_accuracy, 2 / 6, 6),
        (final_answer_accuracy, 2 / 6, 6),
        (successful_revision_rate, 1 / 4, 4),
        (missed_revision_rate, 2 / 3, 3),
        (overreaction_rate, 1 / 2, 2),
        (kept_correct_rate, 1 / 2, 2),
        (final_abstention_rate, 1 / 6, 6),
        (maintenance_rate, 2 / 5, 5),
    ],
)
def test_requested_metric_definitions(metric, expected_rate, expected_denominator):
    result = metric(TRACES)
    assert result["rate"] == pytest.approx(expected_rate)
    assert result["denominator"] == expected_denominator


def test_empty_metric_denominators_are_explicitly_null():
    assert successful_revision_rate([]) == {
        "rate": None,
        "numerator": 0,
        "denominator": 0,
    }


def test_invalid_stage2_is_excluded_from_completed_denominators():
    # task-0707 denominator policy: invalid/incomplete cases are dropped from
    # completed-based denominators rather than counted as wrong. A valid Stage 1
    # still counts toward the Stage 1 denominator.
    invalid = _trace("2", "yes", "no", "KEEP_ANSWER", "no", "invalid")
    invalid["stage2_model_output"] = {
        "raw": "invalid",
        "parsed": None,
        "valid": False,
        "repaired": True,
        "validation_error": "invalid",
    }
    invalid["status"] = "invalid_stage2"

    report = build_report([TRACES[4], invalid], model="test")
    # Stage 1 denominator still includes the invalid case (its Stage 1 was valid).
    assert report["stage1_answer_accuracy"] == {
        "rate": 0.5,
        "numerator": 1,
        "denominator": 2,
    }
    # Final accuracy is over completed cases only, so the invalid case is dropped.
    assert report["final_answer_accuracy"] == {
        "rate": 1.0,
        "numerator": 1,
        "denominator": 1,
    }
    # No completed wrong-Stage-1 cases remain -> empty revision denominators.
    assert report["successful_revision_rate"]["denominator"] == 0
    assert report["missed_revision_rate"] == {
        "rate": None,
        "numerator": 0,
        "denominator": 0,
    }


def test_build_report_contains_exact_requested_metrics_and_counts():
    report = build_report(TRACES, model="test", sample_counts={"yes": 6})
    expected_metrics = {
        "stage1_answer_accuracy",
        "final_answer_accuracy",
        "successful_revision_rate",
        "missed_revision_rate",
        "overreaction_rate",
        "kept_correct_rate",
        "final_abstention_rate",
        "maintenance_rate",
    }
    assert expected_metrics <= report.keys()
    assert report["n_selected_cases"] == report["n_completed_cases"] == 6
    assert report["trace_label_counts"]["missed_revision"] == 2


def test_pipeline_writes_report_and_one_trace_per_case(monkeypatch, tmp_path):
    cases = [_case("A"), _case("B")]
    monkeypatch.setattr(
        "staged_eval.pipeline.load_cases",
        lambda path=None, limit=None: cases,
    )
    monkeypatch.setattr(
        "staged_eval.pipeline.load_ground_truth",
        lambda path=None: {"A": "yes", "B": "no"},
    )

    def stub(_system, user):
        if user.startswith("STAGE 1"):
            return _output("ANSWER", "no")
        return _output("KEEP_ANSWER", "no")

    report, traces = run_pipeline(
        n=2,
        stratify=False,
        out_dir=tmp_path,
        verbose=False,
        call_fn=stub,
    )
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "manifest.json").exists()
    assert (run_dirs[0] / "report.json").exists()
    assert len(list(run_dirs[0].glob("trace_*.json"))) == 2
    assert not list(run_dirs[0].glob(".*.tmp"))
    assert report["run_id"] == run_dirs[0].name
    assert report["provenance"]["model"] == report["model"]
    assert len(traces) == 2 and report["n_completed_cases"] == 2


def test_pipeline_resume_skips_existing_traces(monkeypatch, tmp_path):
    cases = [_case("A"), _case("B")]
    monkeypatch.setattr(
        "staged_eval.pipeline.load_cases",
        lambda path=None, limit=None: cases,
    )
    monkeypatch.setattr(
        "staged_eval.pipeline.load_ground_truth",
        lambda path=None: {"A": "yes", "B": "no"},
    )
    calls = []

    def stub(_system, user):
        calls.append(user)
        if user.startswith("STAGE 1"):
            return _output("ANSWER", "no")
        return _output("KEEP_ANSWER", "no")

    _, first_traces = run_pipeline(
        n=1,
        stratify=False,
        out_dir=tmp_path,
        verbose=False,
        call_fn=stub,
    )
    run_dir = next(tmp_path.iterdir())
    assert len(first_traces) == 1 and len(calls) == 2

    report, resumed_traces = run_pipeline(
        n=2,
        stratify=False,
        resume_dir=run_dir,
        verbose=False,
        call_fn=stub,
    )
    assert len(calls) == 4
    assert [trace["pmid"] for trace in resumed_traces] == ["A", "B"]
    assert report["n_completed_cases"] == 2


def test_builtin_backend_receives_the_reported_model(monkeypatch, tmp_path):
    cases = [_case("A")]
    monkeypatch.setattr(
        "staged_eval.pipeline.load_cases",
        lambda path=None, limit=None: cases,
    )
    monkeypatch.setattr(
        "staged_eval.pipeline.load_ground_truth",
        lambda path=None: {"A": "yes"},
    )
    seen_models = []

    def fake_call(_system, user, *, model):
        seen_models.append(model)
        if user.startswith("STAGE 1"):
            return _output("ANSWER", "yes")
        return _output("KEEP_ANSWER", "yes")

    monkeypatch.setattr("staged_eval.pipeline._call_llm_json", fake_call)
    report, _ = run_pipeline(
        n=1,
        stratify=False,
        model="chosen-model",
        out_dir=tmp_path,
        verbose=False,
    )

    assert seen_models == ["chosen-model", "chosen-model"]
    assert report["model"] == "chosen-model"
    assert report["backend"] == "azure"


def test_run_directories_are_unique(monkeypatch, tmp_path):
    cases = [_case("A")]
    monkeypatch.setattr(
        "staged_eval.pipeline.load_cases",
        lambda path=None, limit=None: cases,
    )
    monkeypatch.setattr(
        "staged_eval.pipeline.load_ground_truth",
        lambda path=None: {"A": "yes"},
    )

    def stub(_system, user):
        if user.startswith("STAGE 1"):
            return _output("ANSWER", "yes")
        return _output("KEEP_ANSWER", "yes")

    run_pipeline(
        n=1,
        stratify=False,
        model="stub-model",
        out_dir=tmp_path,
        verbose=False,
        call_fn=stub,
    )
    run_pipeline(
        n=1,
        stratify=False,
        model="stub-model",
        out_dir=tmp_path,
        verbose=False,
        call_fn=stub,
    )

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 2
    assert run_dirs[0].name != run_dirs[1].name


def test_resume_rejects_model_provenance_mismatch(monkeypatch, tmp_path):
    cases = [_case("A")]
    monkeypatch.setattr(
        "staged_eval.pipeline.load_cases",
        lambda path=None, limit=None: cases,
    )
    monkeypatch.setattr(
        "staged_eval.pipeline.load_ground_truth",
        lambda path=None: {"A": "yes"},
    )

    def stub(_system, user):
        if user.startswith("STAGE 1"):
            return _output("ANSWER", "yes")
        return _output("KEEP_ANSWER", "yes")

    run_pipeline(
        n=1,
        stratify=False,
        model="model-a",
        out_dir=tmp_path,
        verbose=False,
        call_fn=stub,
    )
    run_dir = next(tmp_path.iterdir())

    with pytest.raises(ValueError, match="provenance does not match"):
        run_pipeline(
            n=1,
            stratify=False,
            model="model-b",
            resume_dir=run_dir,
            verbose=False,
            call_fn=stub,
        )


def test_resume_recomputes_an_unreadable_trace(monkeypatch, tmp_path):
    cases = [_case("A")]
    monkeypatch.setattr(
        "staged_eval.pipeline.load_cases",
        lambda path=None, limit=None: cases,
    )
    monkeypatch.setattr(
        "staged_eval.pipeline.load_ground_truth",
        lambda path=None: {"A": "yes"},
    )

    calls = []

    def stub(_system, user):
        calls.append((_system, user))
        if user.startswith("STAGE 1"):
            return _output("ANSWER", "yes")
        return _output("KEEP_ANSWER", "yes")

    run_pipeline(
        n=1,
        stratify=False,
        model="stub-model",
        out_dir=tmp_path,
        verbose=False,
        call_fn=stub,
    )
    run_dir = next(tmp_path.iterdir())
    (run_dir / "trace_A.json").write_text("{", encoding="utf-8")
    calls.clear()

    report, traces = run_pipeline(
        n=1,
        stratify=False,
        model="stub-model",
        resume_dir=run_dir,
        verbose=False,
        call_fn=stub,
    )

    assert len(calls) == 2
    assert traces[0]["status"] == "completed"
    assert report["n_completed_cases"] == 1


# Concurrency

_METRIC_KEYS = {
    "stage1_answer_accuracy",
    "final_answer_accuracy",
    "successful_revision_rate",
    "missed_revision_rate",
    "overreaction_rate",
    "kept_correct_rate",
    "final_abstention_rate",
    "maintenance_rate",
}


def _strip_volatile(trace):
    return {k: v for k, v in trace.items() if k != "_elapsed_seconds"}


def test_concurrency_matches_sequential_and_preserves_metric_logs(monkeypatch, tmp_path):
    cases = [_case(p) for p in ["A", "B", "C", "D", "E", "F"]]
    gt = {"A": "yes", "B": "no", "C": "maybe", "D": "yes", "E": "no", "F": "maybe"}
    monkeypatch.setattr(
        "staged_eval.pipeline.load_cases",
        lambda path=None, limit=None: cases,
    )
    monkeypatch.setattr(
        "staged_eval.pipeline.load_ground_truth",
        lambda path=None: gt,
    )

    def stub(_system, user):
        if user.startswith("STAGE 1"):
            return _output("ANSWER", "no")
        return _output("KEEP_ANSWER", "no")

    report_seq, traces_seq = run_pipeline(
        n=6, stratify=False, out_dir=tmp_path / "seq", workers=1,
        verbose=False, call_fn=stub,
    )
    report_conc, traces_conc = run_pipeline(
        n=6, stratify=False, out_dir=tmp_path / "conc", workers=4,
        verbose=False, call_fn=stub,
    )

    # Deterministic ordering and identical traces regardless of worker count.
    assert [t["pmid"] for t in traces_conc] == ["A", "B", "C", "D", "E", "F"]
    assert [_strip_volatile(t) for t in traces_seq] == [
        _strip_volatile(t) for t in traces_conc
    ]
    # Per-case timing is logged.
    assert all("_elapsed_seconds" in t for t in traces_conc)

    # All 8 metrics logged, each with its eligible-count denominator, and stable
    # across worker counts.
    for key in _METRIC_KEYS:
        assert key in report_conc
        assert set(report_conc[key]) >= {"rate", "numerator", "denominator"}
        assert report_conc[key] == report_seq[key]
    assert report_conc["n_completed_cases"] == report_seq["n_completed_cases"] == 6


def test_invalid_workers_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "staged_eval.pipeline.load_cases",
        lambda path=None, limit=None: [_case("A")],
    )
    monkeypatch.setattr(
        "staged_eval.pipeline.load_ground_truth",
        lambda path=None: {"A": "yes"},
    )
    with pytest.raises(ValueError, match="workers must be >= 1"):
        run_pipeline(
            n=1, stratify=False, out_dir=tmp_path, workers=0,
            verbose=False, call_fn=lambda *_: _output("ANSWER", "yes"),
        )


# Cross-run consistency


def test_cross_run_consistency_metrics(tmp_path):
    from staged_eval.cross_run_consistency import analyze

    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    run1.mkdir()
    run2.mkdir()

    def write(run_dir, pmid, gold, action2, answer2):
        trace = _trace(pmid, gold, "no", action2, answer2, "x")
        (run_dir / f"trace_{pmid}.json").write_text(
            json.dumps(trace), encoding="utf-8"
        )

    # P1: agree + correct | P2: disagree | P3: agree + abstain
    write(run1, "P1", "no", "KEEP_ANSWER", "no")
    write(run2, "P1", "no", "KEEP_ANSWER", "no")
    write(run1, "P2", "no", "REVISE_ANSWER", "yes")
    write(run2, "P2", "no", "KEEP_ANSWER", "no")
    write(run1, "P3", "maybe", "ABSTAIN", None)
    write(run2, "P3", "maybe", "ABSTAIN", None)
    # Present in only one run → excluded from the common set.
    write(run1, "P4", "yes", "KEEP_ANSWER", "no")

    result = analyze([run1, run2], verbose=False)
    assert result["n_cases"] == 3
    assert result["mean_per_case_correctness"] == pytest.approx(0.5)
    assert result["proportion_same_stage2_action"] == pytest.approx(2 / 3)
    assert result["proportion_same_final_answer"] == pytest.approx(2 / 3)
