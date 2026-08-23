"""
Cross-run consistency analysis across repeated pipeline runs on the same cases.

For each pmid common to every selected run, computes:
  (1) mean correctness (final answer == gold label), averaged across runs and
      then across cases
  (2) proportion of cases where the stage-2 action (KEEP_ANSWER / REVISE_ANSWER /
      ABSTAIN) was identical across all selected runs
  (3) proportion of cases where the final answer was identical across all
      selected runs

Usage
-----
python -m staged_eval.cross_run_consistency [--runs DIR1 DIR2 ...] [--n-latest N] [--out DIR]
"""
import argparse
import datetime
import json
from pathlib import Path
from statistics import mean
from typing import List, Optional

from shared.pipeline_utils import write_json_atomic as _write_json_atomic
from .config import RUNS_DIR
from .metrics import _stage2_answer_action


def load_run(run_dir: Path) -> dict:
    """Load every ``trace_*.json`` in a run directory as ``{pmid: trace}``."""
    data = {}
    for path in run_dir.glob("trace_*.json"):
        with open(path, encoding="utf-8") as handle:
            trace = json.load(handle)
        data[str(trace["pmid"])] = trace
    return data


def _final_answer(trace: dict) -> Optional[str]:
    """Final (stage-2) answer, or None when the case did not complete validly."""
    return _stage2_answer_action(trace)[0]


def _stage2_action(trace: dict) -> Optional[str]:
    """Stage-2 action, or None when the case did not complete validly."""
    return _stage2_answer_action(trace)[1]


def resolve_run_dirs(runs: Optional[List[str]], n_latest: int) -> List[Path]:
    """Resolve explicit run names under RUNS_DIR, or the N most recent run dirs."""
    if runs:
        return [RUNS_DIR / name for name in runs]
    candidates = sorted(path for path in RUNS_DIR.iterdir() if path.is_dir())
    if len(candidates) < n_latest:
        raise ValueError(
            f"only {len(candidates)} run dirs found under {RUNS_DIR}, need {n_latest}"
        )
    return candidates[-n_latest:]


def analyze(run_dirs: List[Path], verbose: bool = True) -> dict:
    """Compute cross-run consistency metrics over cases common to all runs."""
    runs = [load_run(run_dir) for run_dir in run_dirs]
    if verbose:
        for run_dir, loaded in zip(run_dirs, runs):
            print(f"loaded {len(loaded)} traces from {run_dir.name}")

    common_pmids = set(runs[0]) if runs else set()
    for loaded in runs[1:]:
        common_pmids &= set(loaded)
    common_pmids = sorted(common_pmids)
    if verbose:
        print(f"cases common to all {len(runs)} runs: {len(common_pmids)}")

    per_case_mean_correct: List[float] = []
    same_action_flags: List[int] = []
    same_final_answer_flags: List[int] = []

    for pmid in common_pmids:
        traces = [loaded[pmid] for loaded in runs]

        corrects = [
            1 if _final_answer(t) == t.get("pubmedqa_gold_label") else 0
            for t in traces
        ]
        per_case_mean_correct.append(mean(corrects))

        actions = {_stage2_action(t) for t in traces}
        same_action_flags.append(1 if len(actions) == 1 else 0)

        final_answers = {_final_answer(t) for t in traces}
        same_final_answer_flags.append(1 if len(final_answers) == 1 else 0)

    result = {
        "run_dirs": [run_dir.name for run_dir in run_dirs],
        "n_cases": len(common_pmids),
        "mean_per_case_correctness": mean(per_case_mean_correct) if common_pmids else None,
        "proportion_same_stage2_action": mean(same_action_flags) if common_pmids else None,
        "proportion_same_final_answer": mean(same_final_answer_flags) if common_pmids else None,
    }
    if verbose:
        print()
        print(
            f"(1) mean per-case correctness over {result['n_cases']} cases: "
            f"{_fmt(result['mean_per_case_correctness'])}"
        )
        print(
            f"(2) proportion identical stage-2 action across all runs: "
            f"{_fmt(result['proportion_same_stage2_action'])}"
        )
        print(
            f"(3) proportion identical final answer across all runs: "
            f"{_fmt(result['proportion_same_final_answer'])}"
        )
    return result


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m staged_eval.cross_run_consistency",
        description="Cross-run consistency metrics across repeated pipeline runs",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="run directory names under runs/ to compare (default: the N latest)",
    )
    parser.add_argument(
        "--n-latest",
        type=int,
        default=3,
        help="if --runs is not given, compare the N most recent run dirs (default 3)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="directory to write consistency_report.json (default: RUNS_DIR)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dirs = resolve_run_dirs(args.runs, args.n_latest)
    report = analyze(dirs)
    out_dir = Path(args.out) if args.out else RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out_path = out_dir / f"consistency_report_{stamp}.json"
    _write_json_atomic(out_path, report)
    print(f"\nwrote {out_path}")
