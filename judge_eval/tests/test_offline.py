import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from judge_eval.agreement import (
    agreement_report,
    cohen_kappa,
    percent_agreement,
    quadratic_weighted_kappa,
)
from judge_eval.client import JudgeClientConfig, OpenAIJudgeBackend
from judge_eval.evaluator import JudgeItem, evaluate_item, infer_model_family
from judge_eval.human import validate_human_sample, write_blinded_human_sample
from judge_eval.pipeline import run_judge_pipeline
from judge_eval.prompts import build_system_prompt, build_user_prompt
from judge_eval.rubric import load_rubric
from judge_eval.schema import safe_json_object, validate_judgment


RUBRIC_PATH = (
    Path(__file__).resolve().parents[1]
    / "rubrics"
    / "clinical_correctness_v1.json"
)


def _rubric():
    return load_rubric(RUBRIC_PATH)


def _item(**overrides):
    values = {
        "item_id": "case-1",
        "task": "What diagnosis is supported?",
        "candidate_answer": "The finding is pneumonia.",
        "reference_answers": ("Pneumonia", "Pulmonary infection"),
        "candidate_model": "gpt-5-mini",
        "metadata": {"strict_final_answer_match": False},
    }
    values.update(overrides)
    return JudgeItem(**values)


def _response(
    label="correct",
    matched_reference_index=0,
    material_contradiction=False,
    unsupported_clinical_claim=False,
    rationale="The candidate is clinically equivalent to the reference.",
):
    return json.dumps(
        {
            "label": label,
            "matched_reference_index": matched_reference_index,
            "material_contradiction": material_contradiction,
            "unsupported_clinical_claim": unsupported_clinical_claim,
            "rationale": rationale,
        }
    )


def test_rubric_is_versioned_hashed_and_complete():
    rubric = _rubric()
    assert rubric.rubric_id == "clinical_answer_correctness"
    assert rubric.version == "1.0.0"
    assert len(rubric.sha256) == 64
    assert rubric.pass_labels == ("correct",)
    assert rubric.score_for("not_scorable") is None


def test_schema_rejects_correct_with_a_material_defect():
    rubric = _rubric()
    raw, error = safe_json_object(
        _response(material_contradiction=True)
    )
    parsed, validation_error = validate_judgment(raw, rubric, reference_count=2)
    assert error is None
    assert parsed is None
    assert "material defect" in validation_error


def test_prompt_blinds_model_identity_and_marks_candidate_as_untrusted():
    item = _item(candidate_model="SECRET-CANDIDATE-MODEL")
    system = build_system_prompt(_rubric())
    user = build_user_prompt(
        task=item.task,
        candidate_answer=item.candidate_answer,
        reference_answers=list(item.reference_answers),
    )
    assert "SECRET-CANDIDATE-MODEL" not in system + user
    assert "untrusted" in system.casefold()
    assert "candidate_answer" in user


def test_repeated_judgments_rotate_references_and_map_indexes_back():
    result = evaluate_item(_item(), _rubric(), lambda *_args: _response(), repeats=3)
    assert result["aggregate"]["label"] == "correct"
    assert result["aggregate"]["unanimous"] is True
    assert [repeat["reference_order"] for repeat in result["repetitions"]] == [
        [0, 1],
        [1, 0],
        [1, 0],
    ]
    assert [
        repeat["parsed"]["matched_reference_index"]
        for repeat in result["repetitions"]
    ] == [0, 1, 1]


def test_inconsistent_repeated_judgments_require_human_review():
    outputs = iter(
        [
            _response("correct", 0),
            _response("incorrect", None, rationale="The answer conflicts."),
            _response("correct", 0),
        ]
    )
    result = evaluate_item(_item(), _rubric(), lambda *_args: next(outputs), repeats=3)
    assert result["aggregate"]["label"] == "correct"
    assert result["aggregate"]["consistency_rate"] == 2 / 3
    assert result["aggregate"]["needs_human_review"] is True


def test_invalid_response_is_repaired_once_and_preserved():
    outputs = iter(["not json", _response()])
    result = evaluate_item(_item(), _rubric(), lambda *_args: next(outputs), repeats=1)
    repeat = result["repetitions"][0]
    assert repeat["initial_raw"] == "not json"
    assert repeat["repaired"] is True
    assert repeat["valid"] is True


def test_family_inference_and_pipeline_self_judge_guard(tmp_path):
    assert infer_model_family("gpt-5-mini") == "openai"
    assert infer_model_family("Kimi-K2.6") == "moonshot"
    with pytest.raises(ValueError, match="same model family"):
        run_judge_pipeline(
            [_item()],
            _rubric(),
            lambda *_args: _response(),
            judge_model="gpt-5.4",
            judge_family="openai",
            judge_backend="test",
            judge_generation={"temperature": 0},
            out_dir=tmp_path,
        )


def test_pipeline_writes_raw_prompts_manifest_and_report(tmp_path):
    report, results, run_dir = run_judge_pipeline(
        [_item()],
        _rubric(),
        lambda *_args: _response(),
        judge_model="claude-test-judge",
        judge_family="anthropic",
        judge_backend="test",
        judge_generation={"temperature": 0},
        out_dir=tmp_path,
        repeats=3,
        verbose=False,
    )
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "report.json").is_file()
    assert len(list(run_dir.glob("judgment_*.json"))) == 1
    assert report["pass_rate"]["rate"] == 1.0
    assert report["pass_rate"]["numerator"] == 1
    assert report["pass_rate"]["denominator"] == 1
    assert report["pass_rate"]["confidence_interval"]["method"] == (
        "Wilson score interval"
    )
    assert report["human_validation"]["status"] == "pending"
    assert results[0]["repetitions"][0]["raw"] == _response()


def test_openai_backend_pins_temperature_and_json_mode():
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=_response()))]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    config = JudgeClientConfig(
        endpoint="https://example.test/openai/v1",
        api_key="secret",
        model="judge-model",
        model_family="other",
    )
    raw = OpenAIJudgeBackend(config, client=client)("system", "user")
    assert json.loads(raw)["label"] == "correct"
    assert captured["temperature"] == 0.0
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["model"] == "judge-model"


def test_blinded_human_sample_omits_judge_and_model_identity(tmp_path):
    result = evaluate_item(_item(), _rubric(), lambda *_args: _response(), repeats=1)
    path = write_blinded_human_sample([result], tmp_path / "sample.csv", n=1)
    text = path.read_text(encoding="utf-8")
    assert "gpt-5-mini" not in text
    assert "clinically equivalent" not in text
    with open(path, encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["human_label_1"] == ""
    assert row["human_label_2"] == ""


def test_human_validation_requires_adjudication_and_reports_agreement(tmp_path):
    result = evaluate_item(_item(), _rubric(), lambda *_args: _response(), repeats=1)
    path = write_blinded_human_sample([result], tmp_path / "sample.csv", n=1)
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["human_label_1"] = "correct"
    rows[0]["human_label_2"] = "correct"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = validate_human_sample(path, [result])
    assert report["human_human"]["percent_agreement"] == 1.0
    assert report["judge_human"]["percent_agreement"] == 1.0


def test_agreement_metrics_include_kappa_confusion_and_fixed_bootstrap():
    first = ["correct", "correct", "incorrect", "incorrect"]
    second = ["correct", "incorrect", "incorrect", "incorrect"]
    assert percent_agreement(first, second) == 0.75
    assert cohen_kappa(first, second) == 0.5
    report = agreement_report(first, second, rater_a="a", rater_b="b", seed=7)
    assert report["cohen_kappa"] == 0.5
    assert quadratic_weighted_kappa(first, second)["value"] == 0.5
    assert report["confusion_matrix"]["correct"]["incorrect"] == 1
    assert report["agreement_confidence_interval"]["seed"] == 7
