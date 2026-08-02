"""Transparent deterministic diagnostics for MedCTA final answers."""

from __future__ import annotations

import re
import unicodedata


def normalize_answer(value: str | None) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def strict_answer_match(
    candidate: str | None,
    accepted_answers: list[str],
) -> bool:
    """Normalized equality against every accepted answer.

    This is intentionally strict and remains a transparent diagnostic. Semantic
    clinical correctness is evaluated later by the separate judge pipeline.
    """
    normalized_candidate = normalize_answer(candidate)
    return bool(normalized_candidate) and any(
        normalized_candidate == normalize_answer(reference)
        for reference in accepted_answers
    )
