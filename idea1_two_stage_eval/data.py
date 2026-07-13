"""
PubMedQA data loading, evidence splitting, and sampling utilities.
"""
import json
import re
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


def normalize_label(raw_label: str, index: int) -> str:
    """Create a display-friendly section label from PubMedQA metadata."""
    text = " ".join(str(raw_label or "").replace("_", " ").replace("-", " ").split())
    if not text:
        return f"Section {index + 1}"
    return text.title()


def normalized_labels(case: dict) -> List[str]:
    """Return deduplicated, display-friendly labels aligned to CONTEXTS."""
    contexts = case["CONTEXTS"]
    raw_labels = list(case.get("LABELS") or [])
    labels, seen = [], set()
    for i in range(len(contexts)):
        label = normalize_label(raw_labels[i] if i < len(raw_labels) else "", i)
        base = label
        suffix = 2
        while label in seen:
            label = f"{base} ({suffix})"
            suffix += 1
        seen.add(label)
        labels.append(label)
    return labels


def _label_key(label: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", label.upper()).strip("_")


def labeled_contexts(case: dict) -> List[Tuple[str, str]]:
    """Zip normalized labels with PubMedQA context chunks."""
    return list(zip(normalized_labels(case), case["CONTEXTS"]))


def split_revision_evidence(case: dict) -> Optional[dict]:
    """
    Split one PubMedQA case into two stages for the commit-then-revise setup.

    Only keep cases with a non-first RESULTS section:
      stage 1 = everything before the first RESULTS section
      stage 2 = the RESULTS section and everything after it
    """
    pairs = labeled_contexts(case)
    if len(pairs) < 2:
        return None

    split_idx = next(
        (i for i, (label, _) in enumerate(pairs) if "RESULT" in _label_key(label)),
        None,
    )

    if split_idx is None or split_idx <= 0 or split_idx >= len(pairs):
        return None

    stage1_pairs = pairs[:split_idx]
    stage2_pairs = pairs[split_idx:]
    strategy = "results_boundary"

    if not stage1_pairs or not stage2_pairs:
        return None

    return {
        **case,
        "normalized_labels": [label for label, _ in pairs],
        "stage1_evidence": stage1_pairs,
        "stage2_added_evidence": stage2_pairs,
        "stage2_full_context": pairs,
        "split_strategy": strategy,
    }


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
