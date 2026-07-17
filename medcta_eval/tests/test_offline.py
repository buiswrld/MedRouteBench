import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import medcta_eval.pipeline as pipeline_module
from medcta_eval import llm
from medcta_eval.adapter import ALLOWED_TOOLS, adapt_raw_dataset
from medcta_eval.config import DATA_PATH, SOURCE_REVISION, STARTER_CASE_IDS
from medcta_eval.data import load_dataset, validate_dataset
from medcta_eval.metrics import build_report
from medcta_eval.pipeline import report_existing_run, run_pipeline
from medcta_eval.prompts import build_user_prompt
from medcta_eval.runner import run_case
from medcta_eval.schema import (
    ACTIONS,
    answer_matches,
    normalize_answer,
    safe_json_loads,
    validate,
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


def _response(action, tool_name=None, answer=None, **extra):
    return json.dumps(
        {"action": action, "tool_name": tool_name, "answer": answer, **extra}
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
        {"action": "CALL_TOOL", "tool_name": "OCR", "answer": None},
        available_tools=["OCR"],
    )
    assert error is None and called.tool_name == "OCR"

    final, error = validate(
        {"action": "FINAL_ANSWER", "tool_name": None, "answer": "Liver"},
        available_tools=["OCR"],
    )
    assert error is None and final.answer == "Liver"


@pytest.mark.parametrize(
    "raw, expected_error",
    [
        (
            {"action": "CALL_TOOL", "tool_name": "NotATool", "answer": None},
            "invalid or unavailable",
        ),
        (
            {"action": "CALL_TOOL", "tool_name": "OCR", "answer": "text"},
            "requires answer null",
        ),
        (
            {"action": "FINAL_ANSWER", "tool_name": "OCR", "answer": "text"},
            "requires tool_name null",
        ),
        (
            {"action": "FINAL_ANSWER", "tool_name": None, "answer": None},
            "requires a nonempty answer",
        ),
        (
            {"action": "UNKNOWN", "tool_name": None, "answer": None},
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
    ],
)
def test_validate_rejects_invalid_actions_and_cross_field_values(raw, expected_error):
    parsed, error = validate(raw, available_tools=["OCR"])
    assert parsed is None and expected_error in error


def test_safe_json_loads_never_raises():
    parsed, error = safe_json_loads("not json")
    assert parsed is None and error.startswith("json_parse:")


def test_final_answer_matching_is_transparent_normalized_exact():
    assert normalize_answer("  GOBLET  Cells. ") == "goblet cells"
    assert answer_matches("GOBLET cells!", ["goblet cells."])
    assert not answer_matches("goblet", ["goblet cells"])


def test_pinned_subset_contains_exact_ids_tools_and_no_thoughts():
    payload = load_dataset(DATA_PATH)
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


def test_perfect_replay_matches_tools_finalization_and_answer():
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


def test_premature_finalization_stops_and_keeps_answer_accuracy_separate():
    trace = run_case(
        _case(reference_tools=("OCR", "ImageDescription")),
        call_fn=_scripted(_response("FINAL_ANSWER", answer="gold answer")),
    )
    assert trace["status"] == "premature_finalization"
    assert trace["attempted_steps"] == 1
    assert trace["reference_step_count"] == 3
    assert trace["final_answer_match"] is True
    assert trace["trajectory_exact_match"] is False


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


def test_requested_metrics_use_the_locked_denominators():
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
        "rate": 0.25,
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
    ):
        assert report[metric] == {"rate": None, "numerator": 0, "denominator": 0}


def test_inference_failure_does_not_dilute_evaluable_case_metrics():
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


def test_model_preflight_checks_visibility_without_inference(monkeypatch):
    monkeypatch.setattr(
        llm,
        "preflight_model",
        lambda model: {"model": model, "available": True},
    )
    assert pipeline_module._preflight_groq_model("vision-model") == {
        "model": "vision-model",
        "available": True,
        "check": "model_list_only",
    }

    monkeypatch.setattr(
        llm,
        "preflight_model",
        lambda model: {"model": model, "available": False},
    )
    with pytest.raises(RuntimeError, match="not available"):
        pipeline_module._preflight_groq_model("missing-model")


def test_builtin_pipeline_preflight_fails_before_artifacts_or_inference(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "cases.json"
    _write_payload(data_path, _case("A"))
    inference_calls = []

    def unavailable(model):
        assert model == "missing-model"
        raise RuntimeError("configured model unavailable")

    monkeypatch.setattr(pipeline_module, "_preflight_groq_model", unavailable)
    monkeypatch.setattr(
        pipeline_module,
        "_call_groq_json",
        lambda *_args, **_kwargs: inference_calls.append(True),
    )

    with pytest.raises(RuntimeError, match="configured model unavailable"):
        run_pipeline(
            n=1,
            model="missing-model",
            out_dir=tmp_path / "runs",
            cases_path=data_path,
            verbose=False,
        )

    assert inference_calls == []
    assert not (tmp_path / "runs").exists()


def test_groq_adapter_sends_image_url_and_json_mode(monkeypatch):
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(llm, "get_client", lambda: client)
    monkeypatch.setattr(llm, "resolve_image_url", lambda url: url)
    result = llm.call_json(
        "system",
        "user",
        "https://example.test/image.jpg",
        model="vision-model",
    )
    assert result == '{"ok":true}'
    assert captured["model"] == "vision-model"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["reasoning_effort"] == "none"
    content = captured["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "user"}
    assert content[1]["image_url"]["url"] == "https://example.test/image.jpg"


def test_huggingface_image_resolution_is_used_without_changing_other_hosts(monkeypatch):
    assert llm.resolve_image_url("https://example.test/image.jpg") == (
        "https://example.test/image.jpg"
    )

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "image/jpeg"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def geturl(self):
            return "https://cdn.example.test/resolved.jpg"

    llm.clear_image_url_cache()
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    resolved = llm.resolve_image_url("https://huggingface.co/datasets/x/resolve/y/image.jpg")
    assert resolved == "https://cdn.example.test/resolved.jpg"


def test_huggingface_image_resolution_cache_expires_before_signed_url(monkeypatch):
    clock = [100.0]
    requests = []

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "image/jpeg"}

        def __init__(self, resolved):
            self.resolved = resolved

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def geturl(self):
            return self.resolved

    def fake_urlopen(*_args, **_kwargs):
        requests.append(len(requests) + 1)
        return FakeResponse(f"https://cdn.example.test/resolved-{len(requests)}.jpg")

    llm.clear_image_url_cache()
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(llm, "IMAGE_URL_CACHE_TTL_SECONDS", 10.0)
    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    source = "https://huggingface.co/datasets/x/resolve/y/image.jpg"

    first = llm.resolve_image_url(source)
    clock[0] = 109.9
    cached = llm.resolve_image_url(source)
    clock[0] = 110.0
    refreshed = llm.resolve_image_url(source)

    assert first == cached == "https://cdn.example.test/resolved-1.jpg"
    assert refreshed == "https://cdn.example.test/resolved-2.jpg"
    assert len(requests) == 2


def test_groq_json_validation_failure_falls_back_to_locally_validated_text(monkeypatch):
    captured = []

    class Completions:
        def create(self, **kwargs):
            captured.append(kwargs)
            if len(captured) == 1:
                raise RuntimeError("provider code=json_validate_failed")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=_response("CALL_TOOL", "OCR")
                        )
                    )
                ]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(llm, "get_client", lambda: client)
    monkeypatch.setattr(llm, "resolve_image_url", lambda url: url)
    raw = llm.call_json("system", "user", None, model="vision-model")
    assert json.loads(raw)["tool_name"] == "OCR"
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in captured[1]


def test_retry_delay_parser_supports_minutes_and_long_wait_cap():
    assert llm._parse_retry_after_message("Please try again in 20m33.5s") == 1233.5
    assert llm._parse_retry_after_message("try again in 4.25s") == 4.25
    assert 1233.5 > llm.MAX_RETRY_WAIT_SECONDS
