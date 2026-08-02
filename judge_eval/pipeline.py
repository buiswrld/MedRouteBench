"""Auditable batch execution for reference-based LLM judging."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from .evaluator import JudgeItem, evaluate_item, infer_model_family
from .prompts import PROMPT_TEMPLATE_VERSION, build_system_prompt
from .rubric import Rubric


JUDGE_RUN_SCHEMA_VERSION = 1


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_item_filename(item_id: str) -> str:
    digest = hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:12]
    return f"judgment_{digest}.json"


def _portable_path(path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return resolved.name


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict:
    if total == 0:
        return {"confidence_level": 0.95, "lower": None, "upper": None}
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return {
        "confidence_level": 0.95,
        "lower": max(0.0, (centre - margin) / denominator),
        "upper": min(1.0, (centre + margin) / denominator),
        "method": "Wilson score interval",
    }


def _make_run_dir(parent: Path) -> tuple[Path, str]:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    run_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir = parent / run_id
    run_dir.mkdir(exist_ok=False)
    return run_dir, run_id


def _family_guard(
    *,
    judge_model: str,
    judge_family: str,
    candidate_model: Optional[str],
    allow_same_family: bool,
) -> dict:
    candidate_family = infer_model_family(candidate_model)
    same_model = bool(
        candidate_model and candidate_model.casefold() == judge_model.casefold()
    )
    same_family = bool(
        candidate_family and candidate_family == judge_family.casefold()
    )
    if (same_model or same_family) and not allow_same_family:
        raise ValueError(
            "judge and candidate appear to use the same model family. "
            "Use a different family, or pass allow_same_family=True only with "
            "an explicit methodological justification."
        )
    return {
        "candidate_model": candidate_model,
        "candidate_model_family": candidate_family,
        "judge_model": judge_model,
        "judge_model_family": judge_family.casefold(),
        "same_model": same_model,
        "same_family": same_family,
        "same_family_override": bool(allow_same_family and (same_model or same_family)),
    }


def build_report(results: list[dict], *, run_id: str, provenance: dict) -> dict:
    aggregates = [result["aggregate"] for result in results]
    labels = Counter(aggregate["label"] for aggregate in aggregates)
    scored = [
        aggregate
        for aggregate in aggregates
        if aggregate["label"] in {"correct", "partially_correct", "incorrect"}
    ]
    pass_count = sum(aggregate["passed"] is True for aggregate in scored)
    mean_scores = [
        aggregate["mean_score"]
        for aggregate in scored
        if aggregate["mean_score"] is not None
    ]
    repetitions = [repeat for result in results for repeat in result["repetitions"]]
    return {
        "schema_version": JUDGE_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "provenance": provenance,
        "n_items": len(results),
        "n_scored_items": len(scored),
        "n_not_scorable": labels.get("not_scorable", 0),
        "n_inconclusive": labels.get("inconclusive", 0),
        "n_items_with_no_valid_judgment": sum(
            aggregate["valid_repetitions"] == 0 for aggregate in aggregates
        ),
        "label_counts": dict(sorted(labels.items())),
        "pass_rate": {
            "rate": pass_count / len(scored) if scored else None,
            "numerator": pass_count,
            "denominator": len(scored),
            "confidence_interval": _wilson_interval(pass_count, len(scored)),
        },
        "mean_score": sum(mean_scores) / len(mean_scores) if mean_scores else None,
        "unanimous_rate": {
            "rate": sum(aggregate["unanimous"] for aggregate in aggregates) / len(aggregates)
            if aggregates
            else None,
            "numerator": sum(aggregate["unanimous"] for aggregate in aggregates),
            "denominator": len(aggregates),
        },
        "needs_human_review_count": sum(
            aggregate["needs_human_review"] for aggregate in aggregates
        ),
        "judge_call_count": len(repetitions),
        "invalid_or_failed_call_count": sum(not repeat["valid"] for repeat in repetitions),
        "repair_count": sum(repeat["repaired"] for repeat in repetitions),
        "human_validation": {
            "status": "pending",
            "requirement": (
                "Complete blinded independent human labels and report human-human "
                "and judge-human agreement before using the judge metric as a headline result."
            ),
        },
    }


def run_judge_pipeline(
    items: list[JudgeItem],
    rubric: Rubric,
    call_fn: Callable[[str, str], str],
    *,
    judge_model: str,
    judge_family: str,
    judge_backend: str,
    judge_generation: dict,
    out_dir,
    repeats: int = 3,
    allow_same_family: bool = False,
    source_provenance: Optional[dict] = None,
    verbose: bool = True,
) -> tuple[dict, list[dict], Path]:
    if not items:
        raise ValueError("judge pipeline requires at least one item")
    item_ids = [item.item_id for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("judge item IDs must be unique")
    candidate_models = {item.candidate_model for item in items}
    if len(candidate_models) > 1:
        raise ValueError("one judge run may evaluate only one candidate model")
    candidate_model = next(iter(candidate_models))
    family_check = _family_guard(
        judge_model=judge_model,
        judge_family=judge_family,
        candidate_model=candidate_model,
        allow_same_family=allow_same_family,
    )
    run_dir, run_id = _make_run_dir(Path(out_dir))
    provenance = {
        "judge_backend": judge_backend,
        "judge_generation": judge_generation,
        "model_family_check": family_check,
        "rubric": {
            "rubric_id": rubric.rubric_id,
            "version": rubric.version,
            "sha256": rubric.sha256,
            "source_path": _portable_path(rubric.source_path),
        },
        "prompt": {
            "template_version": PROMPT_TEMPLATE_VERSION,
            "system_prompt_sha256": sha256_text(build_system_prompt(rubric)),
        },
        "repeats_per_item": repeats,
        "repair_invalid_response_once": True,
        "item_ids": item_ids,
        "candidate_identity_in_prompt": False,
        "reference_order_policy": "deterministic_rotate_and_reverse_across_repeats",
        "source": source_provenance,
        "code_sha256": {
            name: sha256_file(Path(__file__).resolve().parent / name)
            for name in (
                "agreement.py",
                "client.py",
                "evaluator.py",
                "human.py",
                "pipeline.py",
                "prompts.py",
                "rubric.py",
                "schema.py",
            )
        },
    }
    manifest = {
        "schema_version": JUDGE_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "provenance": provenance,
    }
    write_json_atomic(run_dir / "manifest.json", manifest)

    results = []
    for index, item in enumerate(items, 1):
        if verbose:
            print(f"[{index}/{len(items)}] judging item_id={item.item_id}")
        result = evaluate_item(item, rubric, call_fn, repeats=repeats)
        results.append(result)
        write_json_atomic(run_dir / _safe_item_filename(item.item_id), result)
        partial = build_report(results, run_id=run_id, provenance=provenance)
        partial.update(
            run_complete=False,
            n_planned_items=len(items),
            missing_item_ids=item_ids[index:],
        )
        write_json_atomic(run_dir / "partial_report.json", partial)

    report = build_report(results, run_id=run_id, provenance=provenance)
    report.update(
        run_complete=True,
        n_planned_items=len(items),
        missing_item_ids=[],
    )
    write_json_atomic(run_dir / "report.json", report)
    partial_path = run_dir / "partial_report.json"
    if partial_path.exists():
        partial_path.unlink()
    return report, results, run_dir
