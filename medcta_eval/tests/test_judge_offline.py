import csv
import json

from medcta_eval.answer_scoring import normalize_answer, strict_answer_match
from medcta_eval.judge import score_existing_run, validate_completed_human_sample
from medcta_eval.metrics import strict_final_answer_match_rate


def _judge_response():
    return json.dumps(
        {
            "label": "correct",
            "matched_reference_index": 0,
            "material_contradiction": False,
            "unsupported_clinical_claim": False,
            "rationale": "The clinical meaning matches the accepted reference.",
        }
    )


def _write_inference_run(path):
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "source-run",
                "provenance": {
                    "model": "gpt-5-mini",
                    "selected_case_ids": ["A", "B"],
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "trace_A.json").write_text(
        json.dumps(
            {
                "case_id": "A",
                "question": "What is the diagnosis?",
                "accepted_ground_truth_answers": ["Pneumonia", "Pulmonary infection"],
                "final_answer": "This is pneumonia.",
                "final_answer_match": False,
                "final_answer_match_method": "normalized_exact_whitelist",
                "status": "premature_finalization",
            }
        ),
        encoding="utf-8",
    )
    (path / "trace_B.json").write_text(
        json.dumps(
            {
                "case_id": "B",
                "question": "What should the agent do?",
                "accepted_ground_truth_answers": ["Call OCR"],
                "final_answer": None,
                "final_answer_match": False,
                "final_answer_match_method": "normalized_exact_whitelist",
                "status": "missed_finalization",
            }
        ),
        encoding="utf-8",
    )


def test_strict_scorer_is_transparent_and_uses_every_reference():
    assert normalize_answer("  Pulmonary-infection. ") == "pulmonary infection"
    assert strict_answer_match("Pulmonary infection!", ["Pneumonia", "Pulmonary infection"])
    assert not strict_answer_match("This is pneumonia.", ["Pneumonia"])


def test_strict_report_recomputes_legacy_judge_fields_instead_of_trusting_them():
    legacy = {
        "status": "premature_finalization",
        "final_answer": "Different diagnosis",
        "accepted_ground_truth_answers": ["Pneumonia"],
        "final_answer_match": True,
        "final_answer_match_method": "llm_judge_threshold_0.8",
    }
    assert strict_final_answer_match_rate([legacy]) == {
        "rate": 0.0,
        "numerator": 0,
        "denominator": 1,
    }


def test_medcta_judging_is_offline_from_inference_and_writes_human_sample(tmp_path):
    source_run = tmp_path / "source"
    _write_inference_run(source_run)
    report, results, judge_run = score_existing_run(
        source_run,
        judge_model="claude-test-judge",
        judge_family="anthropic",
        repeats=3,
        human_sample_size=1,
        call_fn=lambda *_args: _judge_response(),
        verbose=False,
    )
    assert report["n_items"] == 1
    assert report["provenance"]["source"]["n_source_traces"] == 2
    assert report["provenance"]["source"]["n_answers_submitted_to_judge"] == 1
    assert report["pass_rate"]["rate"] == 1.0
    assert report["medcta_semantic_final_answer_accuracy"] == {
        "status": "provisional_pending_human_validation",
        "rate": 0.5,
        "numerator": 1,
        "denominator": 2,
        "no_final_answer_counted_incorrect": 1,
        "unresolved_judge_item_count": 0,
        "note": (
            "The denominator is every evaluable candidate case; cases without "
            "a final answer count as incorrect. The rate is withheld when any "
            "answer lacks a conclusive judge result."
        ),
    }
    assert len(results) == 1
    assert (judge_run / "human_validation_sample.csv").is_file()
    assert (judge_run / "report.json").is_file()
    assert not any(source_run.glob("trace_*.json.tmp"))


def test_completed_human_labels_update_judge_report(tmp_path):
    source_run = tmp_path / "source"
    _write_inference_run(source_run)
    _report, _results, judge_run = score_existing_run(
        source_run,
        judge_model="claude-test-judge",
        judge_family="anthropic",
        repeats=1,
        human_sample_size=1,
        call_fn=lambda *_args: _judge_response(),
        verbose=False,
    )
    sample = judge_run / "human_validation_sample.csv"
    with open(sample, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["human_label_1"] = "correct"
    rows[0]["human_label_2"] = "correct"
    with open(sample, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    validation = validate_completed_human_sample(judge_run, sample)
    assert validation["judge_human"]["percent_agreement"] == 1.0
    updated = json.loads((judge_run / "report.json").read_text(encoding="utf-8"))
    assert updated["human_validation"]["status"] == "complete"
