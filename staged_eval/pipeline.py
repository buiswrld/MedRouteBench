"""Load, split, sample, run, and report the two-stage PubMedQA experiment."""

import argparse
import datetime
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .config import MAX_TOKENS, PACKAGE_DIR, RUNS_DIR, AZURE_DEPLOYMENT, WORKERS
from shared.pipeline_utils import (
    callable_id as _callable_id,
    create_run_dir as _create_run_dir,
    sha256_file as _sha256_file,
    write_json_atomic as _write_json_atomic,
)
from .data import (
    FINAL_ANSWERS,
    eligible_cases,
    load_cases,
    load_ground_truth,
    resolve_cases_path,
    resolve_ground_truth_path,
    stratified_sample,
    warn_if_degenerate_labels,
)
from .metrics import build_report
from .runner import run_case
from .schema import ACTIONS


RUN_SCHEMA_VERSION = 1


def _call_llm_json(system: str, user: str, *, model: str) -> str:
    """Load the built-in provider adapter only when selected."""
    from .llm import call_json

    return call_json(system, user, model=model)


def _select_backend(call_fn: Optional[Callable], model: Optional[str]):
    if call_fn is None:
        selected_model = model or AZURE_DEPLOYMENT
        if not selected_model:
            raise RuntimeError(
                "AZURE_DEPLOYMENT must be set. "
                "Add it to MedRouteBench/.env or export in your shell."
            )
        return (
            partial(_call_llm_json, model=selected_model),
            selected_model,
            "azure",
            {"max_completion_tokens": MAX_TOKENS},
        )

    cid = _callable_id(call_fn)
    return call_fn, model or f"custom:{cid}", f"callable:{cid}", None


def _build_provenance(
    cases_path: Path,
    ground_truth_path: Path,
    *,
    model: str,
    backend: str,
    generation: Optional[dict],
    use_fixtures: bool,
) -> dict:
    code_names = [
        "data.py",
        "metrics.py",
        "pipeline.py",
        "prompts.py",
        "runner.py",
        "schema.py",
    ]
    if backend == "azure":
        code_names.append("llm.py")
    code_files = {name: PACKAGE_DIR / name for name in code_names}
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "backend": backend,
        "model": model,
        "generation": generation,
        "use_fixtures": use_fixtures,
        "cases": {
            "path": str(cases_path),
            "sha256": _sha256_file(cases_path),
        },
        "ground_truth": {
            "path": str(ground_truth_path),
            "sha256": _sha256_file(ground_truth_path),
        },
        "code_sha256": {
            name: _sha256_file(path) for name, path in code_files.items()
        },
        "actions": list(ACTIONS),
    }


def _resume_signature(provenance: dict) -> dict:
    return {
        "schema_version": provenance.get("schema_version"),
        "backend": provenance.get("backend"),
        "model": provenance.get("model"),
        "generation": provenance.get("generation"),
        "use_fixtures": provenance.get("use_fixtures"),
        "cases_sha256": (provenance.get("cases") or {}).get("sha256"),
        "ground_truth_sha256": (provenance.get("ground_truth") or {}).get("sha256"),
        "code_sha256": provenance.get("code_sha256"),
        "actions": provenance.get("actions"),
    }


def _validate_resume_provenance(saved: Optional[dict], current: dict) -> None:
    if not isinstance(saved, dict):
        raise ValueError("resume manifest is missing a valid provenance object")
    saved_signature = _resume_signature(saved)
    current_signature = _resume_signature(current)
    if saved_signature != current_signature:
        raise ValueError(
            "resume provenance does not match the current run configuration: "
            f"saved={saved_signature}, current={current_signature}"
        )


def _build_progress_report(
    traces: List[dict],
    *,
    selected_pmids: List[str],
    model: str,
    backend: str,
    run_id: str,
    provenance: dict,
    sample_counts: Optional[dict],
    dataset_counts: Optional[dict],
) -> dict:
    report = build_report(
        traces,
        model=model,
        backend=backend,
        run_id=run_id,
        provenance=provenance,
        sample_counts=sample_counts,
        dataset_counts=dataset_counts,
    )
    traced = {t["pmid"] for t in traces}
    missing = [pmid for pmid in selected_pmids if pmid not in traced]
    report.update(
        {
            "n_planned_cases": len(selected_pmids),
            "run_complete": not missing,
            "missing_pmids": missing,
        }
    )
    return report


def run_pipeline(
    n: int = 50,
    *,
    stratify: bool = True,
    model: Optional[str] = None,
    out_dir=None,
    limit: Optional[int] = None,
    workers: Optional[int] = None,
    verbose: bool = True,
    call_fn: Optional[Callable] = None,
    resume_dir=None,
    cases_path=None,
    ground_truth_path=None,
    use_fixtures: bool = False,
) -> Tuple[dict, List[dict]]:
    """Run the experiment with an explicit backend identity and auditable inputs."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if out_dir is not None and resume_dir is not None:
        raise ValueError("out_dir and resume_dir are mutually exclusive")
    effective_workers = workers if workers is not None else WORKERS
    if effective_workers < 1:
        raise ValueError("workers must be >= 1")

    effective_call, selected_model, backend, generation = _select_backend(
        call_fn,
        model,
    )
    resolved_cases = resolve_cases_path(cases_path, use_fixtures=use_fixtures)
    resolved_ground_truth = resolve_ground_truth_path(
        ground_truth_path,
        use_fixtures=use_fixtures,
    )
    provenance = _build_provenance(
        resolved_cases,
        resolved_ground_truth,
        model=selected_model,
        backend=backend,
        generation=generation,
        use_fixtures=use_fixtures,
    )

    all_cases = load_cases(path=resolved_cases, limit=limit)
    ground_truth = load_ground_truth(path=resolved_ground_truth)

    gold_cases = [
        case
        for case in all_cases
        if ground_truth.get(case["pmid"]) in FINAL_ANSWERS
    ]
    eligible = eligible_cases(gold_cases, ground_truth)
    if stratify and n < len(eligible):
        cases = stratified_sample(eligible, ground_truth, n)
    else:
        cases = eligible[:n]

    warn_if_degenerate_labels(cases, ground_truth)
    sample_counts = dict(Counter(ground_truth[case["pmid"]] for case in cases))
    dataset_counts = {
        "loaded": len(all_cases),
        "with_official_gold": len(gold_cases),
        "eligible_two_stage": len(eligible),
        "skipped_unsplittable": len(gold_cases) - len(eligible),
    }

    if resume_dir is not None:
        run_dir = Path(resume_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {run_dir}")
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "resume directory has no manifest.json and cannot be verified: "
                f"{run_dir}"
            )
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
            "provenance": provenance,
        }
        _write_json_atomic(run_dir / "manifest.json", manifest)

    selected_pmids = {case["pmid"] for case in cases}
    existing_by_pmid = {}
    if resume_dir is not None:
        for trace_path in run_dir.glob("trace_*.json"):
            try:
                with open(trace_path, encoding="utf-8") as handle:
                    trace = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                if verbose:
                    print(f"[WARN] ignoring unreadable trace {trace_path.name}: {exc}")
                continue
            pmid = str(trace.get("pmid"))
            if (
                pmid in selected_pmids
                and trace.get("pubmedqa_gold_label") == ground_truth.get(pmid)
            ):
                existing_by_pmid[pmid] = trace

    if verbose:
        if hasattr(sys.stdout, "reconfigure"):
            # keep progress visible when stdout is redirected to a file
            sys.stdout.reconfigure(line_buffering=True)
        print(
            f"run_dir: {run_dir} | selected: {len(cases)} | eligible: {len(eligible)}"
        )
        print(f"backend: {backend} | model: {selected_model} | workers: {effective_workers}")
        print(f"sample labels: {sample_counts}")
        if resume_dir is not None:
            print(
                f"resume: {len(existing_by_pmid)} saved traces found | "
                f"remaining: {len(cases) - len(existing_by_pmid)}"
            )

    selected_pmids = [case["pmid"] for case in cases]

    def _write_partial() -> None:
        partial_report = _build_progress_report(
            [traces_by_pmid[pmid] for pmid in selected_pmids if pmid in traces_by_pmid],
            selected_pmids=selected_pmids,
            model=selected_model,
            backend=backend,
            run_id=run_id,
            provenance=provenance,
            sample_counts=sample_counts,
            dataset_counts=dataset_counts,
        )
        _write_json_atomic(run_dir / "partial_report.json", partial_report)

    traces_by_pmid: dict = dict(existing_by_pmid)
    _write_partial()

    pending = [case for case in cases if case["pmid"] not in existing_by_pmid]

    # Concurrent workers share one lazily-built client; get_client() uses
    # double-checked locking (shared/llm.py) so the first-init race is safe
    # without eagerly constructing a client the offline/stub path never needs.
    print_lock = threading.Lock()

    def _run_one(index: int, case: dict) -> dict:
        if verbose:
            with print_lock:
                print(
                    f"[submit {index}/{len(pending)}] pmid={case['pmid']} | fixed stages=2",
                    flush=True,
                )
        started = time.monotonic()
        trace = run_case(case, ground_truth[case["pmid"]], call_fn=effective_call)
        trace["_elapsed_seconds"] = round(time.monotonic() - started, 3)
        # Independent path per pmid: safe to write from the worker thread.
        _write_json_atomic(run_dir / f"trace_{case['pmid']}.json", trace)
        return trace

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_case = {
            executor.submit(_run_one, index, case): case
            for index, case in enumerate(pending, 1)
        }
        # Collect results in the main thread so partial_report writes stay serialised.
        for future in as_completed(future_to_case):
            case = future_to_case[future]
            try:
                trace = future.result()
            except Exception as exc:  # keep one hard failure from killing the batch
                if verbose:
                    with print_lock:
                        print(
                            f"[WARN] pmid={case['pmid']} failed: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                continue  # untraced pmid surfaces via run_complete / missing_pmids
            traces_by_pmid[trace["pmid"]] = trace
            _write_partial()

    # Deterministic order regardless of completion order or worker count.
    traces = [traces_by_pmid[pmid] for pmid in selected_pmids if pmid in traces_by_pmid]

    report = build_report(
        traces,
        model=selected_model,
        backend=backend,
        run_id=run_id,
        provenance=provenance,
        sample_counts=sample_counts,
        dataset_counts=dataset_counts,
    )
    _write_json_atomic(run_dir / "report.json", report)
    partial_path = run_dir / "partial_report.json"
    if partial_path.exists():
        partial_path.unlink()
    if verbose:
        print(json.dumps(report, indent=2))
    return report, traces


def inspect_trace(trace: dict) -> None:
    """Print a compact two-stage summary."""
    print(
        f"pmid={trace['pmid']} | gold={trace['pubmedqa_gold_label']} | "
        f"status={trace['status']} | label={trace['label']}"
    )
    for stage in (1, 2):
        output = trace.get(f"stage{stage}_model_output")
        if output is None:
            print(f"  stage={stage} not run")
            continue
        parsed = output.get("parsed")
        if parsed is None:
            print(f"  stage={stage} INVALID: {output.get('validation_error')}")
            continue
        print(
            f"  stage={stage} action={parsed['action']:<14} "
            f"answer={str(parsed['answer']):<5} repaired={output['repaired']}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m staged_eval.pipeline",
        description="Fixed two-stage PubMedQA commit-then-revise experiment",
    )
    parser.add_argument("--n", type=int, default=50, help="target eligible sample size")
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="disable proportional yes/no/maybe sampling",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model identifier passed to the built-in backend and stored in provenance",
    )
    parser.add_argument("--out", default=None, help="artifact output directory")
    parser.add_argument(
        "--cases",
        default=None,
        help="explicit PubMedQA case JSON path",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="explicit PubMedQA ground-truth JSON path",
    )
    parser.add_argument(
        "--use-fixtures",
        action="store_true",
        help="explicitly use the bundled tiny fixture dataset",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="resume an interrupted run directory without repeating saved PMIDs",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap cases before filtering")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="number of cases to run concurrently (default from config.WORKERS)",
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
    report, traces = run_pipeline(
        n=args.n,
        stratify=not args.no_stratify,
        model=args.model,
        out_dir=args.out,
        limit=args.limit,
        workers=args.workers,
        resume_dir=args.resume,
        cases_path=args.cases,
        ground_truth_path=args.ground_truth,
        use_fixtures=args.use_fixtures,
    )
    if args.inspect is not None:
        if 0 <= args.inspect < len(traces):
            inspect_trace(traces[args.inspect])
        else:
            print(
                f"[WARN] --inspect {args.inspect} out of range (n={len(traces)})",
                file=sys.stderr,
            )
