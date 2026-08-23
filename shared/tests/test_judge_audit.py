import csv
import json

from shared.judge_audit import export_blinded_sample


def _write_run(run_dir, *, model, answer, score):
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "model": model,
                    "judge": {"provider": "azure", "model": "gpt-5.4"},
                }
            }
        )
    )
    (run_dir / "trace_7.json").write_text(
        json.dumps(
            {
                "case_id": "7",
                "question": "What finding is shown?",
                "accepted_ground_truth_answers": ["finding A"],
                "final_answer": answer,
                "final_answer_score": score,
                "final_answer_match": score >= 0.8,
            }
        )
    )


def test_export_blinds_model_and_judge_outcome(tmp_path):
    first = tmp_path / "run_a"
    second = tmp_path / "run_b"
    _write_run(first, model="anthropic/model", answer="finding A", score=1.0)
    _write_run(second, model="meta/model", answer="finding B", score=0.2)
    out_dir = tmp_path / "audit"

    annotation_path, key_path = export_blinded_sample(
        [first, second], out_dir=out_dir, sample_size=30, seed=42
    )

    with open(annotation_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert "candidate_model" not in rows[0]
    assert "judge_score" not in rows[0]
    assert all(row["human_label"] == "" for row in rows)

    key = json.loads(key_path.read_text())
    assert {item["candidate_model"] for item in key["selected_items"]} == {
        "anthropic/model",
        "meta/model",
    }
