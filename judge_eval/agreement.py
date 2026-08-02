"""Human/judge agreement metrics with deterministic confidence intervals."""

from __future__ import annotations

import random
from collections import Counter


def percent_agreement(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("agreement requires two non-empty equal-length label lists")
    return sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Unweighted Cohen's kappa for nominal labels."""
    observed = percent_agreement(labels_a, labels_b)
    n = len(labels_a)
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    labels = set(counts_a) | set(counts_b)
    expected = sum(counts_a[label] * counts_b[label] for label in labels) / (n * n)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def quadratic_weighted_kappa(
    labels_a: list[str],
    labels_b: list[str],
    *,
    order: tuple[str, ...] = ("incorrect", "partially_correct", "correct"),
) -> dict:
    """Quadratic weighted kappa on ordinal labels.

    Pairs containing labels outside ``order`` (notably ``not_scorable``) are
    excluded and reported rather than assigned an arbitrary ordinal value.
    """
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("weighted kappa requires equal non-empty label lists")
    indexes = {label: index for index, label in enumerate(order)}
    pairs = [
        (indexes[a], indexes[b])
        for a, b in zip(labels_a, labels_b)
        if a in indexes and b in indexes
    ]
    excluded = len(labels_a) - len(pairs)
    if not pairs:
        return {"value": None, "n": 0, "excluded": excluded, "order": list(order)}
    n = len(pairs)
    counts_a = Counter(a for a, _ in pairs)
    counts_b = Counter(b for _, b in pairs)
    scale = max(1, len(order) - 1)
    observed_disagreement = sum(((a - b) / scale) ** 2 for a, b in pairs) / n
    expected_disagreement = sum(
        counts_a[i] * counts_b[j] * ((i - j) / scale) ** 2
        for i in range(len(order))
        for j in range(len(order))
    ) / (n * n)
    value = (
        1.0
        if expected_disagreement == 0 and observed_disagreement == 0
        else 1 - observed_disagreement / expected_disagreement
        if expected_disagreement
        else None
    )
    return {"value": value, "n": n, "excluded": excluded, "order": list(order)}


def bootstrap_agreement_interval(
    labels_a: list[str],
    labels_b: list[str],
    *,
    seed: int = 0,
    samples: int = 2000,
    alpha: float = 0.05,
) -> dict:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("bootstrap requires equal non-empty label lists")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    rng = random.Random(seed)
    n = len(labels_a)
    values = []
    for _ in range(samples):
        indexes = [rng.randrange(n) for _ in range(n)]
        values.append(
            sum(labels_a[index] == labels_b[index] for index in indexes) / n
        )
    values.sort()
    lower_index = max(0, int((alpha / 2) * samples))
    upper_index = min(samples - 1, int((1 - alpha / 2) * samples) - 1)
    return {
        "confidence_level": 1 - alpha,
        "lower": values[lower_index],
        "upper": values[upper_index],
        "bootstrap_samples": samples,
        "seed": seed,
    }


def agreement_report(
    labels_a: list[str],
    labels_b: list[str],
    *,
    rater_a: str,
    rater_b: str,
    seed: int = 0,
    bootstrap_samples: int = 2000,
) -> dict:
    labels = sorted(set(labels_a) | set(labels_b))
    confusion = {
        actual: {
            predicted: sum(
                a == actual and b == predicted
                for a, b in zip(labels_a, labels_b)
            )
            for predicted in labels
        }
        for actual in labels
    }
    return {
        "n": len(labels_a),
        "rater_a": rater_a,
        "rater_b": rater_b,
        "percent_agreement": percent_agreement(labels_a, labels_b),
        "cohen_kappa": cohen_kappa(labels_a, labels_b),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(labels_a, labels_b),
        "agreement_confidence_interval": bootstrap_agreement_interval(
            labels_a,
            labels_b,
            seed=seed,
            samples=bootstrap_samples,
        ),
        "confusion_matrix": confusion,
        "label_counts": {
            rater_a: dict(sorted(Counter(labels_a).items())),
            rater_b: dict(sorted(Counter(labels_b).items())),
        },
    }
