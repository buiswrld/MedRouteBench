"""Offline semantic scoring for saved MedCTA final answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional

from judge_eval.client import OpenAIJudgeBackend, config_from_env
from judge_eval.evaluator import JudgeItem, infer_model_family
from judge_eval.human import validate_human_sample, write_blinded_human_sample
from judge_eval.pipeline import run_judge_pipeline, sha256_file, write_json_atomic
from judge_eval.rubric import load_rubric


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_RUBRIC = (
    PACKAGE_DIR.parent
    / "judge_eval"
    / "rubrics"
    / "clinical_correctness_v1.json"
)


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_inference_run(run_dir) -> tuple[dict, list[dict]]:
    resolved = Path(run_dir).resolve()
    manifest_path = resolved / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MedCTA run lacks manifest.json: {resolved}")
    manifest = _load_json(manifest_path)
    provenance = manifest.get("provenance") or {}
    selected = [str(item) for item in provenance.get("selected_case_ids") or []]
    if not selected:
        raise ValueError("MedCTA manifest lacks selected_case_ids")
    traces_by_id = {}
    for path in resolved.glob("trace_*.json"):
        trace = _load_json(path)
        case_id = str(trace.get("case_id"))
        if case_id in traces_by_id:
            raise ValueError(f"duplicate trace for case_id {case_id}")
        traces_by_id[case_id] = trace
    missing = [item_id for item_id in selected if item_id not in traces_by_id]
    if missing:
        raise ValueError(
            "MedCTA run is incomplete; semantic scoring requires every selected "
            f"trace. Missing case IDs: {missing}"
        )
    traces = [traces_by_id[item_id] for item_id in selected if item_id in traces_by_id]
    return manifest, traces


def _items_from_traces(traces: list[dict], candidate_model: str) -> list[JudgeItem]:
    items = []
    for trace in traces:
        answer = trace.get("final_answer")
        references = trace.get("accepted_ground_truth_answers")
        if not isinstance(answer, str) or not answer.strip():
            continue
        if not isinstance(references, list) or not references:
            raise ValueError(f"trace {trace.get('case_id')} lacks accepted answers")
        items.append(
            JudgeItem(
                item_id=str(trace["case_id"]),
                task=str(trace.get("question") or ""),
                candidate_answer=answer,
                reference_answers=tuple(str(value) for value in references),
                candidate_model=candidate_model,
                metadata={
                    "trajectory_status": trace.get("status"),
                    "strict_final_answer_match": bool(trace.get("final_answer_match")),
                    "strict_match_method": trace.get("final_answer_match_method"),
                },
            )
        )
    if not items:
        raise ValueError("MedCTA run contains no final answers to judge")
    return items


def score_existing_run(
    run_dir,
    *,
    judge_model: Optional[str] = None,
    judge_family: Optional[str] = None,
    rubric_path=DEFAULT_RUBRIC,
    repeats: int = 3,
    human_sample_size: int = 30,
    human_sample_seed: int = 0,
    allow_same_family: bool = False,
    call_fn: Optional[Callable[[str, str], str]] = None,
    judge_backend_name: Optional[str] = None,
    judge_generation: Optional[dict] = None,
    out_dir=None,
    verbose: bool = True,
) -> tuple[dict, list[dict], Path]:
    resolved_run = Path(run_dir).resolve()
    manifest, traces = _load_inference_run(resolved_run)
    source_provenance = manifest.get("provenance") or {}
    candidate_model = source_provenance.get("model")
    if not isinstance(candidate_model, str) or not candidate_model.strip():
        raise ValueError("MedCTA source manifest lacks the candidate model identity")
    candidate_model = candidate_model.strip()
    evaluable_source = [
        trace for trace in traces if trace.get("status") != "inference_failure"
    ]
    items = _items_from_traces(evaluable_source, candidate_model)
    rubric = load_rubric(rubric_path)

    if call_fn is None:
        client_config = config_from_env(model=judge_model, model_family=judge_family)
        backend = OpenAIJudgeBackend(client_config)
        effective_call = backend
        effective_model = client_config.model
        effective_family = client_config.model_family
        backend_name = "azure-openai-compatible"
        generation = client_config.public_dict()
    else:
        if not judge_model or not judge_family:
            raise ValueError("custom judge calls require judge_model and judge_family")
        effective_call = call_fn
        effective_model = judge_model
        effective_family = judge_family.casefold()
        backend_name = judge_backend_name or "custom-callable"
        generation = judge_generation or {}

    source_manifest = resolved_run / "manifest.json"
    source = {
        "medcta_run_id": manifest.get("run_id") or resolved_run.name,
        "manifest_filename": source_manifest.name,
        "manifest_sha256": sha256_file(source_manifest),
        "candidate_model": candidate_model,
        "candidate_model_family": infer_model_family(candidate_model),
        "n_source_traces": len(traces),
        "n_answers_submitted_to_judge": len(items),
        "adapter_code_sha256": sha256_file(Path(__file__).resolve()),
    }
    destination = Path(out_dir) if out_dir else resolved_run / "judge_runs"
    report, results, judge_run_dir = run_judge_pipeline(
        items,
        rubric,
        effective_call,
        judge_model=effective_model,
        judge_family=effective_family,
        judge_backend=backend_name,
        judge_generation=generation,
        out_dir=destination,
        repeats=repeats,
        allow_same_family=allow_same_family,
        source_provenance=source,
        verbose=verbose,
    )
    sample_path = write_blinded_human_sample(
        results,
        judge_run_dir / "human_validation_sample.csv",
        n=human_sample_size,
        seed=human_sample_seed,
    )
    resolved_labels = {"correct", "partially_correct", "incorrect"}
    resolved = [
        result
        for result in results
        if result["aggregate"]["label"] in resolved_labels
    ]
    correct = sum(
        result["aggregate"]["label"] == "correct" for result in resolved
    )
    unresolved = len(results) - len(resolved)
    no_final_answer = len(evaluable_source) - len(items)
    report["medcta_semantic_final_answer_accuracy"] = {
        "status": "provisional_pending_human_validation",
        "rate": (
            correct / len(evaluable_source)
            if evaluable_source and unresolved == 0
            else None
        ),
        "numerator": correct,
        "denominator": len(evaluable_source),
        "no_final_answer_counted_incorrect": no_final_answer,
        "unresolved_judge_item_count": unresolved,
        "note": (
            "The denominator is every evaluable candidate case; cases without "
            "a final answer count as incorrect. The rate is withheld when any "
            "answer lacks a conclusive judge result."
        ),
    }
    report["human_validation"].update(
        {
            "sample_path": sample_path.name,
            "sample_size": min(human_sample_size, len(results)),
            "sample_seed": human_sample_seed,
            "blinded_fields": ["candidate_model", "judge_label", "judge_rationale"],
            "independent_human_raters_required": 2,
        }
    )
    write_json_atomic(judge_run_dir / "report.json", report)
    return report, results, judge_run_dir


def validate_completed_human_sample(judge_run_dir, human_csv) -> dict:
    run_dir = Path(judge_run_dir).resolve()
    results = [
        _load_json(path)
        for path in sorted(run_dir.glob("judgment_*.json"))
    ]
    if not results:
        raise FileNotFoundError(f"no judgment artifacts found in {run_dir}")
    validation = validate_human_sample(human_csv, results)
    write_json_atomic(run_dir / "human_validation.json", validation)
    report_path = run_dir / "report.json"
    report = _load_json(report_path)
    report["human_validation"].update(
        status="complete",
        completed_labels_filename=Path(human_csv).name,
        agreement=validation,
    )
    write_json_atomic(report_path, report)
    return validation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Judge saved MedCTA final answers without rerunning inference."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score", help="create a new judge run")
    score.add_argument("run_dir")
    score.add_argument("--judge-model")
    score.add_argument("--judge-family")
    score.add_argument("--rubric", default=str(DEFAULT_RUBRIC))
    score.add_argument("--repeats", type=int, default=3)
    score.add_argument("--human-sample-size", type=int, default=30)
    score.add_argument("--human-sample-seed", type=int, default=0)
    score.add_argument("--allow-same-family", action="store_true")
    score.add_argument("--out-dir")

    validate = subparsers.add_parser(
        "validate-human", help="compare completed human labels with judge labels"
    )
    validate.add_argument("judge_run_dir")
    validate.add_argument("human_csv")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "score":
        report, _results, run_dir = score_existing_run(
            args.run_dir,
            judge_model=args.judge_model,
            judge_family=args.judge_family,
            rubric_path=args.rubric,
            repeats=args.repeats,
            human_sample_size=args.human_sample_size,
            human_sample_seed=args.human_sample_seed,
            allow_same_family=args.allow_same_family,
            out_dir=args.out_dir,
        )
        print(f"judge_run_dir: {run_dir}")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        validation = validate_completed_human_sample(
            args.judge_run_dir,
            args.human_csv,
        )
        print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
