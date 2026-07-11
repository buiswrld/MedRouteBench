"""Load, split, sample, run, and report the two-stage PubMedQA experiment."""

import argparse
import datetime
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .config import GROQ_MODEL, RUNS_DIR
from .data import (
    FINAL_ANSWERS,
    eligible_cases,
    load_cases,
    load_ground_truth,
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
    call_fn: Optional[Callable] = None,
    resume_dir=None,
) -> Tuple[dict, List[dict]]:
    """Run the experiment and write one JSON trace per selected eligible case."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if out_dir is not None and resume_dir is not None:
        raise ValueError("out_dir and resume_dir are mutually exclusive")

    selected_model = model or GROQ_MODEL
    runs_dir = Path(out_dir) if out_dir else RUNS_DIR
    all_cases = load_cases(limit=limit)
    ground_truth = load_ground_truth()

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
        run_dir = Path(resume_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {run_dir}")
    else:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = runs_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

    selected_pmids = {case["pmid"] for case in cases}
    existing_by_pmid = {}
    if resume_dir is not None:
        for trace_path in run_dir.glob("trace_*.json"):
            with open(trace_path, encoding="utf-8") as handle:
                trace = json.load(handle)
            pmid = str(trace.get("pmid"))
            if pmid in selected_pmids:
                existing_by_pmid[pmid] = trace

    if verbose:
        print(f"run_dir: {run_dir} | selected: {len(cases)} | eligible: {len(eligible)}")
        print(f"sample labels: {sample_counts}")
        if resume_dir is not None:
            print(
                f"resume: {len(existing_by_pmid)} complete traces found | "
                f"remaining: {len(cases) - len(existing_by_pmid)}"
            )

    traces: List[dict] = []
    for index, case in enumerate(cases, 1):
        existing = existing_by_pmid.get(case["pmid"])
        if existing is not None:
            traces.append(existing)
            continue
        if verbose:
            print(f"[{index}/{len(cases)}] pmid={case['pmid']} | fixed stages=2")
        kwargs = {"call_fn": call_fn} if call_fn is not None else {}
        trace = run_case(case, ground_truth[case["pmid"]], **kwargs)
        traces.append(trace)
        with open(
            run_dir / f"trace_{case['pmid']}.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(trace, handle, indent=2, ensure_ascii=False)

    report = build_report(
        traces,
        model=selected_model,
        sample_counts=sample_counts,
        dataset_counts=dataset_counts,
    )
    with open(run_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
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
    parser.add_argument("--model", default=None, help="model name stored in the report")
    parser.add_argument("--out", default=None, help="artifact output directory")
    parser.add_argument(
        "--resume",
        default=None,
        help="resume an interrupted run directory without repeating saved PMIDs",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap cases before filtering")
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
        resume_dir=args.resume,
    )
    if args.inspect is not None:
        if 0 <= args.inspect < len(traces):
            inspect_trace(traces[args.inspect])
        else:
            print(
                f"[WARN] --inspect {args.inspect} out of range (n={len(traces)})",
                file=sys.stderr,
            )
