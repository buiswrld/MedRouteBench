"""Load, split, sample, run, and report the two-stage PubMedQA experiment."""

import argparse
import json
import sys
import threading
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .config import (
    LLM_PROVIDER,
    MAX_TOKENS,
    PACKAGE_DIR,
    RUNS_DIR,
    SEED,
    WORKERS,
    default_model,
    normalize_provider,
)
from shared import harness
from shared.pipeline_utils import (
    callable_id as _callable_id,
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


def _call_llm_json(
    system: str,
    user: str,
    *,
    model: str,
    provider: str,
) -> str:
    """Load the built-in provider adapter only when selected."""
    from .llm import call_json

    return call_json(system, user, model=model, provider=provider)


def _select_backend(
    call_fn: Optional[Callable],
    model: Optional[str],
    provider: Optional[str],
):
    if call_fn is None:
        selected_provider = normalize_provider(provider or LLM_PROVIDER)
        selected_model = model or default_model(selected_provider)
        if not selected_model:
            raise RuntimeError(
                f"No model configured for {selected_provider}. Pass --model or set "
                f"{'OPENROUTER_MODEL' if selected_provider == 'openrouter' else 'AZURE_DEPLOYMENT'}."
            )
        return (
            partial(
                _call_llm_json,
                model=selected_model,
                provider=selected_provider,
            ),
            selected_model,
            selected_provider,
            {"max_completion_tokens": MAX_TOKENS, "seed": SEED},
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
    reversed_order: bool,
) -> dict:
    code_names = [
        "data.py",
        "metrics.py",
        "pipeline.py",
        "prompts.py",
        "runner.py",
        "schema.py",
    ]
    if backend in {"azure", "openrouter"}:
        code_names.append("llm.py")
    code_files = {name: PACKAGE_DIR / name for name in code_names}
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "backend": backend,
        "model": model,
        "generation": generation,
        "use_fixtures": use_fixtures,
        "reversed": reversed_order,
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
        "reversed": provenance.get("reversed"),
        "cases_sha256": (provenance.get("cases") or {}).get("sha256"),
        "ground_truth_sha256": (provenance.get("ground_truth") or {}).get("sha256"),
        "code_sha256": provenance.get("code_sha256"),
        "actions": provenance.get("actions"),
    }


def run_pipeline(
    n: int = 50,
    *,
    stratify: bool = True,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    out_dir=None,
    limit: Optional[int] = None,
    workers: Optional[int] = None,
    verbose: bool = True,
    call_fn: Optional[Callable] = None,
    resume_dir=None,
    cases_path=None,
    ground_truth_path=None,
    use_fixtures: bool = False,
    reversed_order: bool = False,
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
        call_fn, model, provider
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
        reversed_order=reversed_order,
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

    run_dir, run_id, _ = harness.resolve_run_dir(
        out_dir=out_dir,
        resume_dir=resume_dir,
        runs_dir=RUNS_DIR,
        provenance=provenance,
        resume_signature_fn=_resume_signature,
    )

    selected_pmids = [case["pmid"] for case in cases]
    existing_by_pmid = {}
    if resume_dir is not None:
        existing_by_pmid = harness.load_saved_traces(
            run_dir,
            "pmid",
            selected_pmids,
            extra_predicate=lambda trace: trace.get("pubmedqa_gold_label")
            == ground_truth.get(str(trace.get("pmid"))),
            verbose=verbose,
        )

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

    report_fn = partial(
        build_report,
        model=selected_model,
        backend=backend,
        run_id=run_id,
        provenance=provenance,
        sample_counts=sample_counts,
        dataset_counts=dataset_counts,
        reversed_order=reversed_order,
    )

    def _write_partial() -> None:
        partial_report = harness.build_progress_report(
            [traces_by_pmid[pmid] for pmid in selected_pmids if pmid in traces_by_pmid],
            selected_ids=selected_pmids,
            id_key="pmid",
            build_report_fn=report_fn,
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
        trace = run_case(
            case,
            ground_truth[case["pmid"]],
            call_fn=effective_call,
            reversed_order=reversed_order,
        )
        trace["_elapsed_seconds"] = round(time.monotonic() - started, 3)
        # Independent path per pmid: safe to write from the worker thread.
        _write_json_atomic(run_dir / f"trace_{case['pmid']}.json", trace)
        return trace

    def _on_result(trace: dict) -> None:
        traces_by_pmid[trace["pmid"]] = trace
        _write_partial()

    harness.run_cases_concurrently(
        pending,
        run_one=_run_one,
        id_key="pmid",
        workers=effective_workers,
        on_result=_on_result,
        verbose=verbose,
    )

    # Deterministic order regardless of completion order or worker count.
    traces = [traces_by_pmid[pmid] for pmid in selected_pmids if pmid in traces_by_pmid]

    report = report_fn(traces)
    harness.finalize_report(run_dir, report)
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
        "--provider",
        choices=("azure", "openrouter"),
        default=None,
        help="built-in provider (default: LLM_PROVIDER, otherwise azure)",
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
        "--reversed",
        action="store_true",
        help=(
            "show RESULTS-onward evidence at Stage 1 and everything-before-RESULTS "
            "at Stage 2 (flips the normal ordering)"
        ),
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
        provider=args.provider,
        out_dir=args.out,
        limit=args.limit,
        workers=args.workers,
        resume_dir=args.resume,
        cases_path=args.cases,
        ground_truth_path=args.ground_truth,
        use_fixtures=args.use_fixtures,
        reversed_order=args.reversed,
    )
    if args.inspect is not None:
        if 0 <= args.inspect < len(traces):
            inspect_trace(traces[args.inspect])
        else:
            print(
                f"[WARN] --inspect {args.inspect} out of range (n={len(traces)})",
                file=sys.stderr,
            )
