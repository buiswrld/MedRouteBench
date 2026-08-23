"""Shared run/resume/report orchestration for MedRouteBench pipelines.

Both evaluator packages (medcta_eval, staged_eval) run the same shape of
experiment: resolve a run directory (fresh or resumed-and-provenance-
verified), execute cases (sequentially or concurrently), checkpoint a
partial report after each case, then write the final report. This module
owns that shape; each package supplies its own case loading, per-case
inference (`run_case`), and metrics (`build_report`).
"""

import datetime
import importlib.metadata
import json
import platform
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from .pipeline_utils import create_run_dir, write_json_atomic


def _runtime_versions() -> dict:
    packages = {}
    for distribution in ("openai", "tenacity", "python-dotenv", "Pillow"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
    }


def resolve_run_dir(
    *,
    out_dir,
    resume_dir,
    runs_dir: Path,
    provenance: dict,
    resume_signature_fn: Callable[[dict], dict],
    extra_manifest_fields: Optional[dict] = None,
) -> Tuple[Path, str, dict]:
    """Create a fresh run directory, or resolve and provenance-verify one to resume."""
    if resume_dir is not None:
        run_dir = Path(resume_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {run_dir}")
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"resume directory has no manifest.json and cannot be verified: {run_dir}"
            )
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        saved_provenance = manifest.get("provenance")
        if not isinstance(saved_provenance, dict):
            raise ValueError("resume manifest is missing a valid provenance object")
        saved_signature = resume_signature_fn(saved_provenance)
        current_signature = resume_signature_fn(provenance)
        if saved_signature != current_signature:
            raise ValueError(
                "resume provenance does not match the current run configuration: "
                f"saved={saved_signature}, current={current_signature}"
            )
        run_id = str(manifest.get("run_id") or run_dir.name)
    else:
        resolved_runs_dir = Path(out_dir) if out_dir else runs_dir
        run_dir, run_id = create_run_dir(resolved_runs_dir)
        manifest = {
            "run_id": run_id,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "runtime": _runtime_versions(),
            "provenance": provenance,
            **(extra_manifest_fields or {}),
        }
        write_json_atomic(run_dir / "manifest.json", manifest)
    return run_dir, run_id, manifest


def load_saved_traces(
    run_dir: Path,
    id_key: str,
    selected_ids: Iterable[str],
    *,
    extra_predicate: Optional[Callable[[dict], bool]] = None,
    verbose: bool = True,
) -> dict:
    """Load trace_*.json files belonging to the selected ids, skipping unreadable ones."""
    selected = set(selected_ids)
    existing: dict = {}
    for trace_path in run_dir.glob("trace_*.json"):
        try:
            with open(trace_path, encoding="utf-8") as handle:
                trace = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            if verbose:
                print(f"[WARN] ignoring unreadable trace {trace_path.name}: {exc}")
            continue
        item_id = str(trace.get(id_key))
        if item_id in selected and (extra_predicate is None or extra_predicate(trace)):
            existing[item_id] = trace
    return existing


def build_progress_report(
    traces: List[dict],
    *,
    selected_ids: List[str],
    id_key: str,
    build_report_fn: Callable[[List[dict]], dict],
) -> dict:
    """Build a report plus completeness bookkeeping (missing ids, run_complete)."""
    report = build_report_fn(traces)
    traced = {str(t.get(id_key)) for t in traces}
    missing = [item_id for item_id in selected_ids if item_id not in traced]
    report.update(
        {
            "n_planned_cases": len(selected_ids),
            "run_complete": not missing,
            f"missing_{id_key}s": missing,
        }
    )
    return report


def run_cases_concurrently(
    pending: List[dict],
    *,
    run_one: Callable[[int, dict], dict],
    id_key: str,
    workers: int,
    on_result: Optional[Callable[[dict], None]] = None,
    verbose: bool = True,
) -> dict:
    """Run `run_one(index, case)` over `pending` with up to `workers` threads.

    A single hard per-case failure is logged and skipped rather than killing
    the batch; the untraced id then surfaces via the caller's missing-ids
    bookkeeping. `on_result` runs on the main thread after each success, so a
    caller can checkpoint a partial report without needing its own locking.
    """
    results: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_case = {
            executor.submit(run_one, index, case): case
            for index, case in enumerate(pending, 1)
        }
        for future in as_completed(future_to_case):
            case = future_to_case[future]
            try:
                trace = future.result()
            except Exception as exc:  # keep one hard failure from killing the batch
                if verbose:
                    print(
                        f"[WARN] {id_key}={case.get(id_key)} failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            results[trace[id_key]] = trace
            if on_result is not None:
                on_result(trace)
    return results


def finalize_report(run_dir: Path, report: dict) -> None:
    """Write the final report.json and remove any partial_report.json."""
    write_json_atomic(run_dir / "report.json", report)
    partial_path = run_dir / "partial_report.json"
    if partial_path.exists():
        partial_path.unlink()
