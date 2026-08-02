"""Blinded human-audit sampling and completed-label validation."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .agreement import agreement_report


HUMAN_LABELS = {"correct", "partially_correct", "incorrect", "not_scorable"}


def _round_robin_stratified(results: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for result in results:
        buckets[result["aggregate"]["label"]].append(result)
    for values in buckets.values():
        rng.shuffle(values)
    selected = []
    labels = sorted(buckets)
    while len(selected) < min(n, len(results)):
        progressed = False
        for label in labels:
            if buckets[label] and len(selected) < n:
                selected.append(buckets[label].pop())
                progressed = True
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def write_blinded_human_sample(
    results: list[dict],
    path,
    *,
    n: int = 30,
    seed: int = 0,
) -> Path:
    selected = _round_robin_stratified(results, n, seed)
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "item_id",
                "task",
                "reference_answers_json",
                "candidate_answer",
                "human_label_1",
                "human_label_2",
                "adjudicated_label",
                "notes",
            ],
        )
        writer.writeheader()
        for result in selected:
            item = result["item"]
            writer.writerow(
                {
                    "item_id": item["item_id"],
                    "task": item["task"],
                    "reference_answers_json": json.dumps(
                        item["reference_answers"], ensure_ascii=False
                    ),
                    "candidate_answer": item["candidate_answer"],
                    "human_label_1": "",
                    "human_label_2": "",
                    "adjudicated_label": "",
                    "notes": "",
                }
            )
    return resolved


def validate_human_sample(path, results: list[dict]) -> dict:
    by_id = {result["item"]["item_id"]: result for result in results}
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    human_1, human_2, consensus, judge = [], [], [], []
    errors = []
    for row_number, row in enumerate(rows, 2):
        item_id = row.get("item_id", "")
        first = row.get("human_label_1", "").strip()
        second = row.get("human_label_2", "").strip()
        adjudicated = row.get("adjudicated_label", "").strip()
        if item_id not in by_id:
            errors.append(f"row {row_number}: unknown item_id {item_id!r}")
            continue
        if first not in HUMAN_LABELS or second not in HUMAN_LABELS:
            errors.append(f"row {row_number}: both independent labels are required")
            continue
        if first != second and adjudicated not in HUMAN_LABELS:
            errors.append(f"row {row_number}: disagreement requires adjudicated_label")
            continue
        if adjudicated and adjudicated not in HUMAN_LABELS:
            errors.append(f"row {row_number}: invalid adjudicated_label")
            continue
        judge_label = by_id[item_id]["aggregate"]["label"]
        if judge_label not in HUMAN_LABELS:
            errors.append(f"row {row_number}: judge result is {judge_label!r}")
            continue
        human_1.append(first)
        human_2.append(second)
        consensus.append(adjudicated or first)
        judge.append(judge_label)
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "human_human": agreement_report(
            human_1,
            human_2,
            rater_a="human_1",
            rater_b="human_2",
        ),
        "judge_human": agreement_report(
            consensus,
            judge,
            rater_a="human_consensus",
            rater_b="llm_judge",
        ),
    }
