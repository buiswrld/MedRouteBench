"""Run, resume, and report the MedCTA reference-trajectory evaluation."""

import argparse
import datetime
import json
import sys
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple

from .config import (
    AZURE_DEPLOYMENT,
    MAX_TOKENS,
    PACKAGE_DIR,
    RUNS_DIR,
)
from shared.config import (
    MAX_RETRIES,
    MAX_RETRY_WAIT_SECONDS,
)
from shared.pipeline_utils import (
    callable_id as _callable_id,
    create_run_dir as _create_run_dir,
    sha256_file as _sha256_file,
    write_json_atomic as _write_json_atomic,
)
from .data import load_dataset, resolve_data_path
from .metrics import build_report
from .runner import run_case
from .schema import ACTIONS


RUN_SCHEMA_VERSION = 1


def _call_llm_json(
    system: str,
    user: str,
    image_url: str | None,
    *,
    model: str,
) -> str:
    from .llm import call_json

    return call_json(system, user, image_url, model=model)


def _select_backend(call_fn: Optional[Callable], model: Optional[str]):
    if call_fn is None:
        selected_model = model or AZURE_DEPLOYMENT
        if not selected_model:
            raise RuntimeError(
                "AZURE_DEPLOYMENT must be set. "
                "Add it to MedRouteBench/.env or export in the calling environment."
            )
        return (
            partial(_call_llm_json, model=selected_model),
            selected_model,
            "azure-vision",
            {
                "max_completion_tokens": MAX_TOKENS,
                "max_retries": MAX_RETRIES,
                "max_retry_wait_seconds": MAX_RETRY_WAIT_SECONDS,
            },
        )
    cid = _callable_id(call_fn)
    return call_fn, model or f"custom:{cid}", f"callable:{cid}", None


def _build_provenance(
    data_path: Path,
    dataset_metadata: dict,
    selected_case_ids: list[str],
    *,
    model: str,
    backend: str,
    generation: Optional[dict],
) -> dict:
    code_names = [
        "data.py",
        "metrics.py",
        "pipeline.py",
        "prompts.py",
        "runner.py",
        "schema.py",
    ]
    if backend == "azure-vision":
        code_names.append("llm.py")
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "backend": backend,
        "model": model,
        "generation": generation,
        "image_delivery": "pinned_huggingface_url_via_resolved_cdn",
        "selected_case_ids": selected_case_ids,
        "dataset": dataset_metadata,
        "adapted_data": {"path": str(data_path), "sha256": _sha256_file(data_path)},
        "code_sha256": {
            name: _sha256_file(PACKAGE_DIR / name) for name in code_names
        },
        "actions": list(ACTIONS),
    }


def _resume_signature(provenance: dict) -> dict:
    return {
        "schema_version": provenance.get("schema_version"),
        "backend": provenance.get("backend"),
        "model": provenance.get("model"),
        "generation": provenance.get("generation"),
        "image_delivery": provenance.get("image_delivery"),
        "selected_case_ids": provenance.get("selected_case_ids"),
        "dataset": provenance.get("dataset"),
        "adapted_data_sha256": (provenance.get("adapted_data") or {}).get("sha256"),
        "code_sha256": provenance.get("code_sha256"),
        "actions": provenance.get("actions"),
    }


def _validate_resume_provenance(saved: Optional[dict], current: dict) -> None:
    if not isinstance(saved, dict):
        raise ValueError("resume manifest is missing a valid provenance object")
    if _resume_signature(saved) != _resume_signature(current):
        raise ValueError("resume provenance does not match the current MedCTA run")


def _select_cases(payload: dict, n: int, case_ids: Optional[list[str]]) -> list[dict]:
    cases = payload["cases"]
    if case_ids is not None:
        by_id = {case["case_id"]: case for case in cases}
        missing = [case_id for case_id in case_ids if case_id not in by_id]
        if missing:
            raise ValueError(f"adapted subset does not contain case IDs {missing}")
        cases = [by_id[case_id] for case_id in case_ids]
    return cases[:n]


def _load_saved_traces(
    run_dir: Path,
    selected_case_ids: list[str],
    *,
    verbose: bool,
) -> dict[str, dict]:
    selected = set(selected_case_ids)
    existing_by_id: dict[str, dict] = {}
    for trace_path in run_dir.glob("trace_*.json"):
        try:
            with open(trace_path, encoding="utf-8") as handle:
                trace = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            if verbose:
                print(f"[WARN] ignoring unreadable trace {trace_path.name}: {exc}")
            continue
        case_id = str(trace.get("case_id"))
        if case_id in selected:
            existing_by_id[case_id] = trace
    return existing_by_id


def _build_progress_report(
    traces: list[dict],
    *,
    selected_case_ids: list[str],
    model: str,
    backend: str,
    run_id: str,
    provenance: dict,
) -> dict:
    report = build_report(
        traces,
        model=model,
        backend=backend,
        run_id=run_id,
        provenance=provenance,
    )
    traced_ids = {str(trace.get("case_id")) for trace in traces}
    missing_case_ids = [
        case_id for case_id in selected_case_ids if case_id not in traced_ids
    ]
    report.update(
        {
            "n_planned_cases": len(selected_case_ids),
            "run_complete": not missing_case_ids,
            "missing_case_ids": missing_case_ids,
        }
    )
    return report


def report_existing_run(run_dir, *, verbose: bool = True) -> Tuple[dict, list[dict]]:
    """Regenerate a report from saved traces without making inference calls."""
    resolved_run_dir = Path(run_dir).resolve()
    manifest_path = resolved_run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run directory has no manifest.json: {resolved_run_dir}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("run manifest is missing a valid provenance object")
    selected_case_ids_raw = provenance.get("selected_case_ids")
    if not isinstance(selected_case_ids_raw, list):
        raise ValueError("run manifest is missing selected_case_ids")
    selected_case_ids = [str(case_id) for case_id in selected_case_ids_raw]

    saved = _load_saved_traces(
        resolved_run_dir,
        selected_case_ids,
        verbose=verbose,
    )
    traces = [saved[case_id] for case_id in selected_case_ids if case_id in saved]
    report = _build_progress_report(
        traces,
        selected_case_ids=selected_case_ids,
        model=str(provenance.get("model") or "unknown"),
        backend=str(provenance.get("backend") or "unknown"),
        run_id=str(manifest.get("run_id") or resolved_run_dir.name),
        provenance=provenance,
    )
    report["report_source"] = "recomputed_from_saved_traces"
    report["metrics_code_sha256"] = _sha256_file(PACKAGE_DIR / "metrics.py")
    report_path = resolved_run_dir / (
        "report.json" if report["run_complete"] else "partial_report.json"
    )
    _write_json_atomic(report_path, report)
    if verbose:
        print(f"report_path: {report_path}")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return report, traces


def run_pipeline(
    n: int = 11,
    *,
    model: Optional[str] = None,
    out_dir=None,
    verbose: bool = True,
    call_fn: Optional[Callable] = None,
    resume_dir=None,
    cases_path=None,
    case_ids: Optional[list[str]] = None,
) -> Tuple[dict, list[dict]]:
    """Run the MedCTA starter subset with auditable inputs and artifacts."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if out_dir is not None and resume_dir is not None:
        raise ValueError("out_dir and resume_dir are mutually exclusive")

    effective_call, selected_model, backend, generation = _select_backend(call_fn, model)
    resolved_data = resolve_data_path(cases_path)
    payload = load_dataset(resolved_data)
    cases = _select_cases(payload, n, case_ids)
    selected_case_ids = [case["case_id"] for case in cases]
    model_preflight = None
    provenance = _build_provenance(
        resolved_data,
        payload["dataset"],
        selected_case_ids,
        model=selected_model,
        backend=backend,
        generation=generation,
    )

    if resume_dir is not None:
        run_dir = Path(resume_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {run_dir}")
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"resume directory has no manifest.json: {run_dir}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        _validate_resume_provenance(manifest.get("provenance"), provenance)
        run_id = str(manifest.get("run_id") or run_dir.name)
    else:
        runs_dir = Path(out_dir) if out_dir else RUNS_DIR
        run_dir, run_id = _create_run_dir(runs_dir)
        manifest = {
            "run_id": run_id,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_preflight": model_preflight,
            "provenance": provenance,
        }
        _write_json_atomic(run_dir / "manifest.json", manifest)

    existing_by_id: dict[str, dict] = {}
    if resume_dir is not None:
        existing_by_id = _load_saved_traces(
            run_dir,
            selected_case_ids,
            verbose=verbose,
        )

    if verbose:
        print(f"run_dir: {run_dir} | selected: {len(cases)}")
        print(f"backend: {backend} | model: {selected_model}")
        if resume_dir is not None:
            print(
                f"resume: {len(existing_by_id)} saved traces | "
                f"remaining: {len(cases) - len(existing_by_id)}"
            )

    traces: list[dict] = [
        existing_by_id[case_id]
        for case_id in selected_case_ids
        if case_id in existing_by_id
    ]
    partial_report = _build_progress_report(
        traces,
        selected_case_ids=selected_case_ids,
        model=selected_model,
        backend=backend,
        run_id=run_id,
        provenance=provenance,
    )
    _write_json_atomic(run_dir / "partial_report.json", partial_report)

    _case_id_order = {cid: i for i, cid in enumerate(selected_case_ids)}
    for index, case in enumerate(cases, 1):
        existing = existing_by_id.get(case["case_id"])
        if existing is not None:
            continue
        if verbose:
            print(
                f"[{index}/{len(cases)}] case_id={case['case_id']} | "
                f"reference_steps={len(case['reference_steps'])}"
            )
        trace = run_case(case, call_fn=effective_call)
        traces.append(trace)
        _write_json_atomic(run_dir / f"trace_{case['case_id']}.json", trace)
        partial_report = _build_progress_report(
            traces,
            selected_case_ids=selected_case_ids,
            model=selected_model,
            backend=backend,
            run_id=run_id,
            provenance=provenance,
        )
        _write_json_atomic(run_dir / "partial_report.json", partial_report)

    traces.sort(key=lambda t: _case_id_order.get(t["case_id"], len(selected_case_ids)))
    report = _build_progress_report(
        traces,
        selected_case_ids=selected_case_ids,
        model=selected_model,
        backend=backend,
        run_id=run_id,
        provenance=provenance,
    )
    _write_json_atomic(run_dir / "report.json", report)
    partial_path = run_dir / "partial_report.json"
    if partial_path.exists():
        partial_path.unlink()
    if verbose:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return report, traces


def inspect_trace(trace: dict) -> None:
    print(
        f"case_id={trace['case_id']} | status={trace['status']} | "
        f"exact={trace['trajectory_exact_match']} | "
        f"final_match={trace['final_answer_match']}"
    )
    for step in trace.get("steps") or []:
        output = step.get("model_output") or {}
        parsed = output.get("parsed")
        if parsed is None:
            problem = output.get("inference_error") or output.get("validation_error")
            print(f"  step={step['step_index']} INVALID: {problem}")
            continue
        print(
            f"  step={step['step_index']} expected={step['expected_action']}"
            f"/{step.get('expected_tool_name')} actual={parsed['action']}"
            f"/{parsed.get('tool_name')} match={step['action_match']} "
            f"repaired={output['repaired']}"
        )


def _parse_case_ids(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("--case-ids must be unique")
    return ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m medcta_eval.pipeline",
        description="MedCTA reference-trajectory routing evaluator",
    )
    parser.add_argument("--n", type=int, default=11, help="number of adapted cases")
    parser.add_argument("--model", default=None, help="model identifier")
    parser.add_argument("--out", default=None, help="artifact output directory")
    parser.add_argument("--cases", default=None, help="adapted MedCTA subset JSON")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--resume", default=None, help="resume a provenance-verified run directory"
    )
    run_mode.add_argument(
        "--report-existing",
        default=None,
        help="recompute a full or partial report from saved traces; no inference",
    )
    parser.add_argument(
        "--case-ids",
        default=None,
        help="optional comma-separated adapted case IDs in evaluation order",
    )
    parser.add_argument(
        "--inspect",
        type=int,
        default=None,
        metavar="I",
        help="print the I-th trace after the run (0-indexed)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.report_existing is not None:
        report, traces = report_existing_run(args.report_existing)
    else:
        report, traces = run_pipeline(
            n=args.n,
            model=args.model,
            out_dir=args.out,
            resume_dir=args.resume,
            cases_path=args.cases,
            case_ids=_parse_case_ids(args.case_ids),
        )
    if args.inspect is not None:
        if 0 <= args.inspect < len(traces):
            inspect_trace(traces[args.inspect])
        else:
            print(
                f"[WARN] --inspect {args.inspect} out of range (n={len(traces)})",
                file=sys.stderr,
            )
