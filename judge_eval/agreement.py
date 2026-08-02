"""Agreement metrics for the blinded human audit."""

from collections import Counter


def percent_agreement(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("agreement requires two non-empty equal-length label lists")
    return sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    observed = percent_agreement(labels_a, labels_b)
    n = len(labels_a)
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    labels = set(counts_a) | set(counts_b)
    expected = sum(counts_a[x] * counts_b[x] for x in labels) / (n * n)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def agreement_report(
    labels_a: list[str],
    labels_b: list[str],
    *,
    rater_a: str,
    rater_b: str,
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
        "confusion_matrix": confusion,
        "label_counts": {
            rater_a: dict(sorted(Counter(labels_a).items())),
            rater_b: dict(sorted(Counter(labels_b).items())),
        },
    }
