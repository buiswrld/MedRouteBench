"""Versioned rubric loading and hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Rubric:
    rubric_id: str
    version: str
    title: str
    construct: str
    criteria: tuple[str, ...]
    labels: dict[str, str]
    score_by_label: dict[str, Optional[float]]
    pass_labels: tuple[str, ...]
    sha256: str
    source_path: str

    def score_for(self, label: str) -> Optional[float]:
        return self.score_by_label[label]


def _canonical_bytes(payload: dict) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def load_rubric(path) -> Rubric:
    resolved = Path(path).resolve()
    with open(resolved, encoding="utf-8") as handle:
        payload = json.load(handle)

    required = {
        "schema_version",
        "rubric_id",
        "version",
        "title",
        "construct",
        "criteria",
        "labels",
        "score_by_label",
        "pass_labels",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(
            "rubric must contain exactly the versioned rubric fields; "
            f"missing={sorted(required - set(payload or {}))}, "
            f"extra={sorted(set(payload or {}) - required)}"
        )
    if payload["schema_version"] != 1:
        raise ValueError("unsupported rubric schema_version")
    labels = payload["labels"]
    scores = payload["score_by_label"]
    pass_labels = payload["pass_labels"]
    if not isinstance(labels, dict) or not labels:
        raise ValueError("rubric labels must be a non-empty object")
    if set(scores) != set(labels):
        raise ValueError("score_by_label must define every rubric label exactly once")
    if not set(pass_labels).issubset(labels):
        raise ValueError("pass_labels must be rubric labels")
    if any(
        score is not None
        and (not isinstance(score, (int, float)) or not 0 <= score <= 1)
        for score in scores.values()
    ):
        raise ValueError("rubric scores must be null or numbers in [0, 1]")

    return Rubric(
        rubric_id=str(payload["rubric_id"]),
        version=str(payload["version"]),
        title=str(payload["title"]),
        construct=str(payload["construct"]),
        criteria=tuple(str(item) for item in payload["criteria"]),
        labels={str(key): str(value) for key, value in labels.items()},
        score_by_label={
            str(key): None if value is None else float(value)
            for key, value in scores.items()
        },
        pass_labels=tuple(str(item) for item in pass_labels),
        sha256=hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
        source_path=str(resolved),
    )
