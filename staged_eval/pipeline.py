"""
Main pipeline: load → sample → run → report → write artifacts.

Usage
-----
python -m staged_eval.pipeline [--n 50] [--no-stratify] [--model ...] \
                                [--out ...] [--limit ...] [--inspect I]
"""
import argparse
import datetime
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from .config import GROQ_MODEL, RUNS_DIR, TEST_SET_PATH, GROUND_TRUTH_PATH
from .data import (
    load_cases,
    load_ground_truth,
    n_stages,
    stratified_sample,
    warn_if_degenerate_labels,
)
from .metrics import build_report
from .runner import run_case


def run_pipeline(
    n: int = 50,
    *,
    stratify: bool = True,
    model: Optional[str] = None,
    out_dir=None,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[dict, List[dict]]:
    """
    Load cases, sample, run staged evaluation, write artifacts, return
    (report_dict, traces_list).

    Parameters
    ----------
    n         : Target sample size.
    stratify  : Proportional yes/no/maybe stratification (default True).
    model     : Override GROQ_MODEL for the report header.
    out_dir   : Override RUNS_DIR for artifact output.
    limit     : Cap the dataset size before sampling (useful for testing).
    verbose   : Print progress to stdout.
    """
    _model    = model or GROQ_MODEL
    runs_dir  = Path(out_dir) if out_dir else RUNS_DIR

    all_cases = load_cases(limit=limit)
    gt        = load_ground_truth()

    if stratify and n < len(all_cases):
        cases = stratified_sample(all_cases, gt, n)
    else:
        cases = all_cases[:n]

    warn_if_degenerate_labels(cases, gt)
    sample_counts = dict(Counter(gt.get(c["pmid"]) for c in cases))

    ts      = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"run_dir: {run_dir}  |  n_cases: {len(cases)}")
        if stratify:
            print(f"stratified sample: {sample_counts}")

    traces: List[dict] = []
    for i, case in enumerate(cases, 1):
        if verbose:
            print(
                f"[{i}/{len(cases)}] pmid={case['pmid']}  "
                f"gt={gt.get(case['pmid'])}  n_stages={n_stages(case)}"
            )
        trace = run_case(case, gt.get(case["pmid"]))
        traces.append(trace)
        with open(run_dir / f"trace_{case['pmid']}.json", "w") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)

    report = build_report(traces, gt, model=_model, sample_counts=sample_counts)
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    if verbose:
        print(json.dumps(report, indent=2))

    return report, traces


def inspect_trace(trace: dict) -> None:
    """Print a compact per-stage summary of a trace."""
    print(f"pmid={trace['pmid']}  gt={trace['gt']}  n_stages={trace['n_stages']}")
    for s in trace["stages"]:
        p = s["parsed"]
        flag = "FINAL" if s["is_final"] else "     "
        print(
            f"  stage={s['stage']:>1} {flag} "
            f"action={p['action']:<18} "
            f"answer={str(p['answer']):<6} "
            f"conf={round(p['confidence'] or 0.0, 2):<4}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m staged_eval.pipeline",
        description="MedRouteBench staged evaluation runner",
    )
    parser.add_argument("--n",          type=int,   default=50,   help="target sample size")
    parser.add_argument("--no-stratify",action="store_true",      help="disable stratified sampling")
    parser.add_argument("--model",      default=None,             help="override GROQ_MODEL")
    parser.add_argument("--out",        default=None,             help="output directory for run")
    parser.add_argument("--limit",      type=int,   default=None, help="cap dataset before sampling")
    parser.add_argument(
        "--inspect", type=int, default=None, metavar="I",
        help="after the run, print the trace for the I-th case (0-indexed)",
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
    )
    if args.inspect is not None:
        if args.inspect < len(traces):
            inspect_trace(traces[args.inspect])
        else:
            print(f"[WARN] --inspect {args.inspect} out of range (n={len(traces)})", file=sys.stderr)
