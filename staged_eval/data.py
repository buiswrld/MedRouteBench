"""PubMedQA loading, evidence splitting, and sampling utilities."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

from .config import (
    TEST_SET_PATH,
    ORI_PQAL_PATH,
    GROUND_TRUTH_PATH,
    FIXTURE_TEST_SET_PATH,
    FIXTURE_GROUND_TRUTH_PATH,
)


FINAL_ANSWERS = {"yes", "no", "maybe"}


def _resolve_path(
    path,
    candidate_paths,
    fixture_path: Path,
    *,
    use_fixtures: bool,
    description: str,
) -> Path:
    """Resolve one input path without silently changing datasets."""
    if path is not None and use_fixtures:
        raise ValueError(
            f"Cannot combine an explicit {description} path with use_fixtures=True"
        )

    if path is not None:
        resolved = Path(path)
    elif use_fixtures:
        resolved = fixture_path
    else:
        resolved = next(
            (candidate for candidate in candidate_paths if candidate.is_file()),
            None,
        )
        if resolved is None:
            searched = ", ".join(str(candidate) for candidate in candidate_paths)
            raise FileNotFoundError(
                f"No production {description} file found. Searched: {searched}. "
                "Pass use_fixtures=True (or --use-fixtures) only for an explicit "
                "fixture run."
            )

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{description.capitalize()} file does not exist: {resolved}"
        )
    return resolved.resolve()


def resolve_cases_path(path=None, *, use_fixtures: bool = False) -> Path:
    """Resolve the production case file, or the fixture only when requested."""
    return _resolve_path(
        path,
        [TEST_SET_PATH, ORI_PQAL_PATH],
        FIXTURE_TEST_SET_PATH,
        use_fixtures=use_fixtures,
        description="PubMedQA cases",
    )


def resolve_ground_truth_path(path=None, *, use_fixtures: bool = False) -> Path:
    """Resolve the production gold file, or the fixture only when requested."""
    return _resolve_path(
        path,
        [GROUND_TRUTH_PATH],
        FIXTURE_GROUND_TRUTH_PATH,
        use_fixtures=use_fixtures,
        description="PubMedQA ground truth",
    )


def load_cases(
    path=None,
    limit: Optional[int] = None,
    *,
    use_fixtures: bool = False,
) -> List[dict]:
    """Load PubMedQA cases and inject each PMID into its case dictionary."""
    resolved = resolve_cases_path(path, use_fixtures=use_fixtures)
    with open(resolved, encoding="utf-8") as handle:
        raw = json.load(handle)
    items = [{"pmid": pmid, **case} for pmid, case in raw.items()]
    return items[:limit] if limit is not None else items


def load_ground_truth(path=None, *, use_fixtures: bool = False) -> dict:
    """Load the official PubMedQA test labels as ``{pmid: yes|no|maybe}``."""
    resolved = resolve_ground_truth_path(path, use_fixtures=use_fixtures)
    with open(resolved, encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        str(pmid): str(label).strip().lower()
        for pmid, label in raw.items()
        if str(label).strip().lower() in FINAL_ANSWERS
    }


def normalized_labels(case: dict) -> List[str]:
    """Return readable, unique section labels aligned one-to-one with CONTEXTS."""
    contexts = list(case.get("CONTEXTS") or [])
    raw_labels = list(case.get("LABELS") or [])
    labels: List[str] = []
    seen = Counter()

    for index in range(len(contexts)):
        raw = raw_labels[index] if index < len(raw_labels) else ""
        label = re.sub(r"[^A-Z0-9]+", " ", str(raw or "").strip().upper()).strip()
        label = re.sub(r"\s+", " ", label) or f"SECTION {index + 1}"
        seen[label] += 1
        if seen[label] > 1:
            label = f"{label} {seen[label]}"
        labels.append(label)
    return labels


def split_evidence(case: dict) -> Optional[dict]:
    """
    Split a PubMedQA case into fixed preliminary and added evidence.

    Uses the first normalized label containing ``RESULT`` as the boundary.
    Returns ``None`` when no usable RESULT section is found or either side has no text.
    Cases where the RESULT section is first (no preliminary evidence) are also excluded.
    """
    contexts = [str(context or "").strip() for context in (case.get("CONTEXTS") or [])]
    labels = normalized_labels(case)
    n_contexts = len(contexts)
    if n_contexts < 2:
        return None

    result_index = next(
        (index for index, label in enumerate(labels) if "RESULT" in label),
        None,
    )
    if result_index is None or not (0 < result_index < n_contexts):
        return None

    preliminary = [
        {"label": labels[index], "context": contexts[index]}
        for index in range(result_index)
    ]
    added = [
        {"label": labels[index], "context": contexts[index]}
        for index in range(result_index, n_contexts)
    ]
    if not any(item["context"] for item in preliminary):
        return None
    if not any(item["context"] for item in added):
        return None
    return {
        "strategy": "first_results_section",
        "split_index": result_index,
        "stage1_evidence": preliminary,
        "stage2_added_evidence": added,
        "full_context": preliminary + added,
    }


def eligible_cases(cases: List[dict], gt: dict) -> List[dict]:
    """Keep only gold-labelled cases that can form two non-empty evidence stages."""
    return [
        case
        for case in cases
        if gt.get(str(case.get("pmid"))) in FINAL_ANSWERS and split_evidence(case) is not None
    ]


def warn_if_degenerate_labels(cases: List[dict], gt: dict) -> None:
    """Print a warning if all selected cases share one ground-truth label."""
    if not cases:
        return
    labels = {gt[c["pmid"]] for c in cases if c["pmid"] in gt}
    if len(labels) <= 1:
        print(
            f"[WARN] All {len(cases)} selected cases share label(s) {labels}. "
            "Metrics may be uninformative."
        )


def stratified_sample(all_cases: List[dict], gt: dict, n: int) -> List[dict]:
    """Return a deterministic proportional yes/no/maybe sample of up to ``n`` cases."""
    if n <= 0:
        return []

    buckets = {
        label: [case for case in all_cases if gt.get(case["pmid"]) == label]
        for label in sorted(FINAL_ANSWERS)
    }
    counts = {label: len(cases) for label, cases in buckets.items() if cases}
    total_population = sum(counts.values())
    if not total_population:
        return []

    target = min(n, total_population)
    raw_quotas = {
        label: target * count / total_population for label, count in counts.items()
    }
    quotas = {label: int(quota) for label, quota in raw_quotas.items()}
    remainder = target - sum(quotas.values())
    order = sorted(
        raw_quotas,
        key=lambda label: (-(raw_quotas[label] - quotas[label]), label),
    )
    for label in order[:remainder]:
        quotas[label] += 1

    selected: List[dict] = []
    for label in sorted(quotas):
        selected.extend(buckets[label][: quotas[label]])

    # Restore source-file order so runs remain easy to compare.
    selected_ids = {case["pmid"] for case in selected}
    return [case for case in all_cases if case["pmid"] in selected_ids]
