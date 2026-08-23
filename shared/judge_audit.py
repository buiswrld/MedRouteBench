"""Export a seeded, model-blinded sample of saved MedCTA judge decisions."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from .pipeline_utils import write_json_atomic


ANNOTATION_FIELDS = (
    "item_id",
    "question",
    "accepted_answers_json",
    "candidate_answer",
    "human_label",
    "notes",
)


def collect_final_answers(run_dirs: list[Path]) -> list[dict]:
    """Collect unique scored final answers without exposing model identity."""
    collected = []
    seen = set()
    for run_dir in sorted(Path(path).resolve() for path in run_dirs):
        manifest_path = run_dir / "manifest.json"
        manifest = {}
        if manifest_path.is_file():
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        provenance = manifest.get("provenance") or {}
        for trace_path in sorted(run_dir.glob("trace_*.json")):
            with open(trace_path, encoding="utf-8") as handle:
                trace = json.load(handle)
            question = trace.get("question")
            answer = trace.get("final_answer")
            golds = trace.get("accepted_ground_truth_answers")
            score = trace.get("final_answer_score")
            if not question or not answer or not isinstance(golds, list):
                continue
            duplicate_key = (str(question), tuple(map(str, golds)), str(answer))
            if duplicate_key in seen:
                continue
            seen.add(duplicate_key)
            collected.append(
                {
                    "question": str(question),
                    "accepted_answers": list(map(str, golds)),
                    "candidate_answer": str(answer),
                    "source_run": str(run_dir),
                    "source_trace": trace_path.name,
                    "case_id": str(trace.get("case_id")),
                    "candidate_model": provenance.get("model"),
                    "judge": provenance.get("judge"),
                    "judge_score": (
                        float(score) if isinstance(score, (int, float)) else None
                    ),
                    "judge_correct": bool(trace.get("final_answer_match")),
                }
            )
    return collected


def export_blinded_sample(
    run_dirs: list[Path],
    *,
    out_dir: Path,
    sample_size: int = 30,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Write a blinded annotation CSV and a separate unblinding key."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    items = collect_final_answers(run_dirs)
    if not items:
        raise ValueError("no scored MedCTA final answers found")
    rng = random.Random(seed)
    selected = rng.sample(items, min(sample_size, len(items)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = out_dir / "judge_human_audit.csv"
    key_path = out_dir / "judge_human_audit_key.json"

    key_items = []
    with open(annotation_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        for index, item in enumerate(selected, start=1):
            item_id = f"audit_{index:03d}"
            writer.writerow(
                {
                    "item_id": item_id,
                    "question": item["question"],
                    "accepted_answers_json": json.dumps(
                        item["accepted_answers"], ensure_ascii=False
                    ),
                    "candidate_answer": item["candidate_answer"],
                    "human_label": "",
                    "notes": "",
                }
            )
            key_items.append({"item_id": item_id, **item})

    write_json_atomic(
        key_path,
        {
            "schema_version": 1,
            "seed": seed,
            "requested_sample_size": sample_size,
            "available_unique_items": len(items),
            "selected_items": key_items,
            "allowed_human_labels": ["correct", "incorrect", "not_scorable"],
        },
    )
    return annotation_path, key_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a blinded human-audit sample from saved MedCTA runs."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    annotation_path, key_path = export_blinded_sample(
        args.run_dirs,
        out_dir=args.out_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"annotation_csv: {annotation_path}")
    print(f"unblinding_key: {key_path}")


if __name__ == "__main__":
    main()
