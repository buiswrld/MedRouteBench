"""Run, resume, and report the MedCTA forced golden-path evaluation.

Unlike medcta_eval.pipeline, every case here is forced through the full
reference tool sequence regardless of what the model declares at each step
(see medcta_golden_eval.runner) -- this module only differs from
medcta_eval.pipeline in which runner/metrics module it wires in and in the
CLI description below.
"""

import argparse
import json
import sys
import threading
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple

from .config import (
    ANSWER_EQUIVALENCE_CONFIDENCE_THRESHOLD,
    FINAL_ACCURACY_CONFIDENCE_THRESHOLD,
    JUDGE_API_KEY,
    JUDGE_BASE_URL,
    JUDGE_MODEL,
    JUDGE_MODEL_FAMILY,
    JUDGE_PROVIDER,
    MAX_TOKENS,
    OPENROUTER_MODEL,
    PACKAGE_DIR,
    RUNS_DIR,
)
from shared.config import OPENROUTER_API_KEY
from shared import harness
from shared.config import (
    MAX_RETRIES,
    MAX_RETRY_WAIT_SECONDS,
)
from shared.pipeline_utils import (
    callable_id as _callable_id,
    sha256_file as _sha256_file,
    write_json_atomic as _write_json_atomic,
)
from shared.usage import RunUsageTracker, activate_usage_tracker
from .data import load_dataset, resolve_data_path
from .metrics import build_report
from .runner import run_case
from .schema import ACTIONS


RUN_SCHEMA_VERSION = 1

# No shared WORKERS default here: MedCTA runs stay sequential unless a caller
# opts in via --workers, since vision/judge calls have different rate-limit
# and cost characteristics than staged_eval's text-only backend.
DEFAULT_WORKERS = 1


def _call_llm_json(
    system: str,
    user: str,
    image_url: str | None,
    *,
    model: str,
    api_key: str | None = None,
) -> str:
    from .llm import call_json

    return call_json(system, user, image_url, model=model, api_key=api_key)


def _select_backend(
    call_fn: Optional[Callable],
    model: Optional[str],
    api_key: Optional[str] = None,
):
    if call_fn is None:
        selected_model = model or OPENROUTER_MODEL
        if not selected_model:
            raise RuntimeError(
                "OPENROUTER_MODEL must be set. "
                "Add it to MedRouteBench/.env or export in the calling environment."
            )
        selected_api_key = api_key or OPENROUTER_API_KEY
        if not selected_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY must be set. "
                "Add it to MedRouteBench/.env, export in the calling environment, "
                "or pass --api-key."
            )
        if not JUDGE_MODEL:
            raise RuntimeError(
                "MEDCTA_JUDGE_MODEL or MEDCTA_JUDGE_DEPLOYMENT must be set. "
                "Configure the fixed GPT-5.4 judge deployment."
            )
        if JUDGE_PROVIDER == "azure" and (not JUDGE_BASE_URL or not JUDGE_API_KEY):
            raise RuntimeError(
                "The Azure MedCTA judge requires a configured endpoint and API key."
            )
        return (
            partial(_call_llm_json, model=selected_model, api_key=selected_api_key),
            selected_model,
            "openrouter-vision",
            {
                "max_completion_tokens": MAX_TOKENS,
                "temperature": "provider_default",
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
    candidate_family = model.split("/", 1)[0].lower() if "/" in model else None
    code_names = [
        "data.py",
        "metrics.py",
        "pipeline.py",
        "prompts.py",
        "runner.py",
        "schema.py",
    ]
    if backend == "openrouter-vision":
        code_names.append("llm.py")
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "backend": backend,
        "model": model,
        "generation": generation,
        "judge": {
            "provider": JUDGE_PROVIDER,
            "model": JUDGE_MODEL,
            "model_family": JUDGE_MODEL_FAMILY,
            "candidate_model_family": candidate_family,
            "same_family_as_candidate": (
                candidate_family == JUDGE_MODEL_FAMILY
                if candidate_family is not None
                else None
            ),
            "answer_accuracy_threshold": FINAL_ACCURACY_CONFIDENCE_THRESHOLD,
            "answer_equivalence_threshold": ANSWER_EQUIVALENCE_CONFIDENCE_THRESHOLD,
            "prompt_file_sha256": _sha256_file(PACKAGE_DIR / "prompts.py"),
        },
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
        "judge": provenance.get("judge"),
        "image_delivery": provenance.get("image_delivery"),
        "selected_case_ids": provenance.get("selected_case_ids"),
        "dataset": provenance.get("dataset"),
        "adapted_data_sha256": (provenance.get("adapted_data") or {}).get("sha256"),
        "code_sha256": provenance.get("code_sha256"),
        "actions": provenance.get("actions"),
    }


def _select_cases(payload: dict, n: int, case_ids: Optional[list[str]]) -> list[dict]:
    cases = payload["cases"]
    if case_ids is not None:
        by_id = {case["case_id"]: case for case in cases}
        missing = [case_id for case_id in case_ids if case_id not in by_id]
        if missing:
            raise ValueError(f"adapted subset does not contain case IDs {missing}")
        cases = [by_id[case_id] for case_id in case_ids]
    return cases[:n]


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

    saved = harness.load_saved_traces(
        resolved_run_dir,
        "case_id",
        selected_case_ids,
        verbose=verbose,
    )
    traces = [saved[case_id] for case_id in selected_case_ids if case_id in saved]
    report_fn = partial(
        build_report,
        model=str(provenance.get("model") or "unknown"),
        backend=str(provenance.get("backend") or "unknown"),
        run_id=str(manifest.get("run_id") or resolved_run_dir.name),
        provenance=provenance,
    )
    report = harness.build_progress_report(
        traces,
        selected_ids=selected_case_ids,
        id_key="case_id",
        build_report_fn=report_fn,
    )
    report["report_source"] = "recomputed_from_saved_traces"
    report["metrics_code_sha256"] = _sha256_file(PACKAGE_DIR / "metrics.py")
    report["usage"] = RunUsageTracker(resolved_run_dir).summary()
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
    api_key: Optional[str] = None,
    out_dir=None,
    workers: Optional[int] = None,
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
    effective_workers = workers if workers is not None else DEFAULT_WORKERS
    if effective_workers < 1:
        raise ValueError("workers must be >= 1")

    effective_call, selected_model, backend, generation = _select_backend(
        call_fn, model, api_key
    )
    resolved_data = resolve_data_path(cases_path)
    payload = load_dataset(resolved_data)
    cases = _select_cases(payload, n, case_ids)
    selected_case_ids = [case["case_id"] for case in cases]
    provenance = _build_provenance(
        resolved_data,
        payload["dataset"],
        selected_case_ids,
        model=selected_model,
        backend=backend,
        generation=generation,
    )

    run_dir, run_id, _ = harness.resolve_run_dir(
        out_dir=out_dir,
        resume_dir=resume_dir,
        runs_dir=RUNS_DIR,
        provenance=provenance,
        resume_signature_fn=_resume_signature,
    )
    usage_tracker = RunUsageTracker(run_dir)

    existing_by_id: dict[str, dict] = {}
    if resume_dir is not None:
        existing_by_id = harness.load_saved_traces(
            run_dir,
            "case_id",
            selected_case_ids,
            verbose=verbose,
        )

    if verbose:
        print(f"run_dir: {run_dir} | selected: {len(cases)}")
        print(f"backend: {backend} | model: {selected_model} | workers: {effective_workers}")
        if resume_dir is not None:
            print(
                f"resume: {len(existing_by_id)} saved traces | "
                f"remaining: {len(cases) - len(existing_by_id)}"
            )

    report_fn = partial(
        build_report,
        model=selected_model,
        backend=backend,
        run_id=run_id,
        provenance=provenance,
    )

    traces_by_id: dict[str, dict] = dict(existing_by_id)

    def _write_partial() -> None:
        partial_report = harness.build_progress_report(
            [traces_by_id[cid] for cid in selected_case_ids if cid in traces_by_id],
            selected_ids=selected_case_ids,
            id_key="case_id",
            build_report_fn=report_fn,
        )
        partial_report["usage"] = usage_tracker.summary()
        _write_json_atomic(run_dir / "partial_report.json", partial_report)

    _write_partial()

    pending = [case for case in cases if case["case_id"] not in existing_by_id]

    # Concurrent workers share one lazily-built client; get_client() uses
    # double-checked locking (shared/llm.py) so the first-init race is safe
    # without eagerly constructing a client the offline/stub path never needs.
    print_lock = threading.Lock()

    def _run_one(index: int, case: dict) -> dict:
        if verbose:
            with print_lock:
                print(
                    f"[submit {index}/{len(pending)}] case_id={case['case_id']} | "
                    f"reference_steps={len(case['reference_steps'])}",
                    flush=True,
                )
        trace = run_case(case, call_fn=effective_call)
        # Independent path per case_id: safe to write from the worker thread.
        _write_json_atomic(run_dir / f"trace_{case['case_id']}.json", trace)
        return trace

    def _on_result(trace: dict) -> None:
        traces_by_id[trace["case_id"]] = trace
        _write_partial()

    with activate_usage_tracker(usage_tracker):
        harness.run_cases_concurrently(
            pending,
            run_one=_run_one,
            id_key="case_id",
            workers=effective_workers,
            on_result=_on_result,
            verbose=verbose,
        )

    # Deterministic order regardless of completion order or worker count.
    traces = [traces_by_id[cid] for cid in selected_case_ids if cid in traces_by_id]

    # Unlike staged_eval, MedCTA's final report.json (not just the partial one)
    # has always carried n_planned_cases/run_complete/missing_case_ids.
    report = harness.build_progress_report(
        traces,
        selected_ids=selected_case_ids,
        id_key="case_id",
        build_report_fn=report_fn,
    )
    report["usage"] = usage_tracker.write_summary()
    harness.finalize_report(run_dir, report)
    if verbose:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return report, traces


def inspect_trace(trace: dict) -> None:
    print(
        f"case_id={trace['case_id']} | status={trace['status']} | "
        f"attempted_exact={trace.get('attempted_trajectory_exact_match')} | "
        f"forced_path_complete={trace['trajectory_exact_match']} | "
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
        prog="python -m medcta_golden_eval.pipeline",
        description=(
            "MedCTA forced golden-path evaluator: replays every case through "
            "the full reference tool sequence regardless of the model's "
            "declared actions, measuring answer-quality evolution and "
            "attempted (never executed) early exits"
        ),
    )
    parser.add_argument("--n", type=int, default=11, help="number of adapted cases")
    parser.add_argument("--model", default=None, help="model identifier")
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API key override (defaults to OPENROUTER_API_KEY)",
    )
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
        "--workers",
        type=int,
        default=None,
        help="number of cases to run concurrently (default: sequential, i.e. 1)",
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
            api_key=args.api_key,
            out_dir=args.out,
            workers=args.workers,
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
