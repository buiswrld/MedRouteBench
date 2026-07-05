"""
PubMedQA data loading, stage helpers, and sampling utilities.
"""
import json
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from .config import TEST_SET_PATH, GROUND_TRUTH_PATH


# ── loaders ──────────────────────────────────────────────────────────────────
def load_cases(path=None, limit: Optional[int] = None) -> List[dict]:
    """Load test_set.json → list of case dicts (each has a 'pmid' key injected)."""
    p = Path(path) if path else TEST_SET_PATH
    with open(p) as f:
        raw = json.load(f)
    items = [{"pmid": pmid, **case} for pmid, case in raw.items()]
    return items[:limit] if limit else items


def load_ground_truth(path=None) -> dict:
    """Load test_ground_truth.json → {pmid: label} dict."""
    p = Path(path) if path else GROUND_TRUTH_PATH
    with open(p) as f:
        return json.load(f)


# ── stage helpers ─────────────────────────────────────────────────────────────
def _normalized_labels(case: dict) -> List[str]:
    """
    Return a label list of length == len(CONTEXTS).
    Blank or duplicate labels get SECTION_<i> suffixes.
    """
    ctx = case["CONTEXTS"]
    raw_lbls = list(case.get("LABELS") or [])
    out, seen = [], set()
    for i in range(len(ctx)):
        raw_lbl = raw_lbls[i] if i < len(raw_lbls) else ""
        lbl = (raw_lbl or "").strip().upper().replace(" ", "_") or f"SECTION_{i}"
        if lbl in seen:
            lbl = f"{lbl}_{i}"
        seen.add(lbl)
        out.append(lbl)
    return out


def n_stages(case: dict) -> int:
    """Total stages for a case: stage 0 = question only, stages 1..N = one label each."""
    return len(case["CONTEXTS"]) + 1


def revealed_pairs(case: dict, stage: int) -> List[Tuple[str, str]]:
    """Return (label, context) pairs visible at `stage`. stage 0 → []."""
    if stage <= 0:
        return []
    labels = _normalized_labels(case)
    ctx = case["CONTEXTS"]
    k = min(stage, len(ctx))
    return list(zip(labels[:k], ctx[:k]))


def is_final_stage(case: dict, stage: int) -> bool:
    return stage == len(case["CONTEXTS"])


def warn_if_degenerate_labels(cases: List[dict], gt: dict) -> None:
    """Print a warning if all loaded cases share the same ground-truth label."""
    labels = {gt[c["pmid"]] for c in cases if c["pmid"] in gt}
    if len(labels) <= 1:
        print(
            f"[WARN] All {len(cases)} loaded cases share label(s) {labels}. "
            f"Metrics uninformative; sample is only for wiring."
        )


# ── sampling ─────────────────────────────────────────────────────────────────
def stratified_sample(all_cases: List[dict], gt: dict, n: int) -> List[dict]:
    """
    Proportional stratified sample (floor + largest-remainder), deterministic
    (file order within each bucket).  Returns exactly `n` cases.
    """
    counts = Counter(gt.values())
    total_pop = sum(counts.values())
    raw_q = {lbl: n * counts[lbl] / total_pop for lbl in counts}
    quotas = {lbl: int(v) for lbl, v in raw_q.items()}
    rem = n - sum(quotas.values())
    for lbl, _ in sorted(raw_q.items(), key=lambda kv: -(kv[1] - int(kv[1])))[:rem]:
        quotas[lbl] += 1
    picked, taken = [], {lbl: 0 for lbl in quotas}
    for c in all_cases:
        lbl = gt.get(c["pmid"])
        if lbl in quotas and taken[lbl] < quotas[lbl]:
            picked.append(c)
            taken[lbl] += 1
        if len(picked) == n:
            break
    return picked
