"""Strict parsing for judge responses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional

from .rubric import Rubric


REQUIRED_KEYS = {
    "label",
    "matched_reference_index",
    "material_contradiction",
    "unsupported_clinical_claim",
    "rationale",
}


@dataclass(frozen=True)
class Judgment:
    label: str
    score: Optional[float]
    passed: Optional[bool]
    matched_reference_index: Optional[int]
    material_contradiction: bool
    unsupported_clinical_claim: bool
    rationale: str

    def as_dict(self) -> dict:
        return asdict(self)


def safe_json_object(raw: str) -> tuple[Optional[dict], Optional[str]]:
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        return None, f"json_parse: {exc}"
    if not isinstance(parsed, dict):
        return None, "judge response root is not an object"
    return parsed, None


def validate_judgment(
    raw,
    rubric: Rubric,
    *,
    reference_count: int,
) -> tuple[Optional[Judgment], Optional[str]]:
    if not isinstance(raw, dict):
        return None, "judge response root is not an object"
    if set(raw) != REQUIRED_KEYS:
        return None, (
            "judge response keys mismatch; "
            f"missing={sorted(REQUIRED_KEYS - set(raw))}, "
            f"extra={sorted(set(raw) - REQUIRED_KEYS)}"
        )

    label = raw["label"]
    if label not in rubric.labels:
        return None, f"unknown judge label {label!r}"
    matched = raw["matched_reference_index"]
    if matched is not None:
        if isinstance(matched, bool) or not isinstance(matched, int):
            return None, "matched_reference_index must be an integer or null"
        if not 0 <= matched < reference_count:
            return None, "matched_reference_index is outside the supplied references"
    if label == "correct" and matched is None:
        return None, "a correct judgment must identify a matched reference"

    contradiction = raw["material_contradiction"]
    unsupported = raw["unsupported_clinical_claim"]
    if not isinstance(contradiction, bool):
        return None, "material_contradiction must be boolean"
    if not isinstance(unsupported, bool):
        return None, "unsupported_clinical_claim must be boolean"
    if label == "correct" and (contradiction or unsupported):
        return None, "a correct judgment cannot contain a material defect"

    rationale = raw["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        return None, "rationale must be a non-empty string"
    rationale = rationale.strip()
    if len(rationale) > 800:
        return None, "rationale exceeds 800 characters"

    score = rubric.score_for(label)
    return Judgment(
        label=label,
        score=score,
        passed=label in rubric.pass_labels if score is not None else None,
        matched_reference_index=matched,
        material_contradiction=contradiction,
        unsupported_clinical_claim=unsupported,
        rationale=rationale,
    ), None
