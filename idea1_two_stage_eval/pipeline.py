"""
Main pipeline: load → sample → run → report → write artifacts.

Usage
-----
python -m staged_eval.pipeline [--n 50] [--no-stratify] [--model ...] \
                                [--out ...] [--limit ...] [--sleep-seconds S] [--inspect I]
"""
import argparse
import datetime
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from .config import GROQ_MODEL, RUNS_DIR, TEST_SET_PATH, GROUND_TRUTH_PATH
from .data import (
    load_cases,
    load_ground_truth,
    split_revision_evidence,
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
    sleep_seconds: float = 0.0,
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
    sleep_seconds : Pause this many seconds between case runs.
    verbose   : Print progress to stdout.
    """
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds must be >= 0")

    _model = model or GROQ_MODEL
    runs_dir = Path(out_dir) if out_dir else RUNS_DIR

    all_cases = load_cases(limit=limit)
    gt = load_ground_truth()
    prepared_all = [
        prepared
        for prepared in (split_revision_evidence(case) for case in all_cases)
        if prepared is not None
    ]
    n_skipped_ineligible = len(all_cases) - len(prepared_all)

    if stratify and n < len(prepared_all):
        cases = stratified_sample(prepared_all, gt, n)
    else:
        cases = prepared_all[:n]

    warn_if_degenerate_labels(cases, gt)
    sample_counts = dict(Counter(gt.get(c["pmid"]) for c in cases))
    split_counts = dict(Counter(c["split_strategy"] for c in cases))

    ts      = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"run_dir: {run_dir}  |  n_cases: {len(cases)}")
        print(f"eligible cases: {len(prepared_all)}  |  skipped ineligible: {n_skipped_ineligible}")
        if stratify:
            print(f"stratified sample: {sample_counts}")
        print(f"split strategies: {split_counts}")
        if sleep_seconds:
            print(f"sleep between cases: {sleep_seconds}s")

    traces: List[dict] = []
    for i, case in enumerate(cases, 1):
        if verbose:
            print(
                f"[{i}/{len(cases)}] pmid={case['pmid']}  "
                f"gt={gt.get(case['pmid'])}  split={case['split_strategy']}  "
                f"stage1_chunks={len(case['stage1_evidence'])}  "
                f"stage2_chunks={len(case['stage2_added_evidence'])}"
            )
        trace = run_case(case, gt.get(case["pmid"]))
        traces.append(trace)
        with open(run_dir / f"trace_{case['pmid']}.json", "w") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
        if sleep_seconds and i < len(cases):
            time.sleep(sleep_seconds)

    report = build_report(
        traces,
        model=_model,
        sample_counts=sample_counts,
        split_counts=split_counts,
        n_skipped_ineligible=n_skipped_ineligible,
    )
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    if verbose:
        print(json.dumps(report, indent=2))

    return report, traces


def inspect_trace(trace: dict) -> None:
    """Print a compact summary of a two-stage trace."""
    print(
        f"pmid={trace['pmid']}  gold={trace['gold_label']}  completed={trace['completed']}  "
        f"label={trace['label']}"
    )
    if trace["stage1_model_output"]:
        s1 = trace["stage1_model_output"]
        print(
            f"  stage1: action={s1['action']:<14} answer={str(s1['answer']):<6} "
            f"conf={round(s1['confidence'] or 0.0, 2):<4}"
        )
    if trace["stage2_model_output"]:
        s2 = trace["stage2_model_output"]
        print(
            f"  stage2: action={s2['action']:<14} answer={str(s2['answer']):<6} "
            f"conf={round(s2['confidence'] or 0.0, 2):<4}"
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
        "--sleep-seconds", type=float, default=0.0,
        help="pause this many seconds between case runs",
    )
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
        sleep_seconds=args.sleep_seconds,
    )
    if args.inspect is not None:
        if args.inspect < len(traces):
            inspect_trace(traces[args.inspect])
        else:
            print(f"[WARN] --inspect {args.inspect} out of range (n={len(traces)})", file=sys.stderr)
