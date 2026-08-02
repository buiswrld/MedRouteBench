"""Pure per-item judge evaluation and repeat aggregation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
from typing import Callable, Optional

from .prompts import build_repair_prompt, build_system_prompt, build_user_prompt
from .rubric import Rubric
from .schema import safe_json_object, validate_judgment


JudgeCall = Callable[[str, str], str]


@dataclass(frozen=True)
class JudgeItem:
    item_id: str
    task: str
    candidate_answer: str
    reference_answers: tuple[str, ...]
    candidate_model: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.item_id:
            raise ValueError("judge item requires item_id")
        if not self.task.strip():
            raise ValueError(f"judge item {self.item_id} requires a task")
        if not self.candidate_answer.strip():
            raise ValueError(f"judge item {self.item_id} requires a candidate answer")
        if not self.reference_answers or not all(
            isinstance(answer, str) and answer.strip()
            for answer in self.reference_answers
        ):
            raise ValueError(f"judge item {self.item_id} requires reference answers")


def infer_model_family(model: Optional[str]) -> Optional[str]:
    """Return a coarse provider family for self-judge safeguards."""
    if not model:
        return None
    value = model.casefold()
    aliases = {
        "openai": ("gpt-", "o1", "o3", "o4"),
        "anthropic": ("claude",),
        "google": ("gemini", "gemma"),
        "moonshot": ("kimi",),
        "deepseek": ("deepseek",),
        "zhipu": ("glm",),
        "alibaba": ("qwen",),
        "meta": ("llama",),
        "mistral": ("mistral", "mixtral"),
    }
    for family, markers in aliases.items():
        if any(marker in value for marker in markers):
            return family
    return None


def _reference_order(count: int, repeat_index: int) -> list[int]:
    """Deterministically rotate and reverse references across repeated calls."""
    order = list(range(count))
    if count <= 1:
        return order
    if repeat_index % 2:
        order.reverse()
    else:
        shift = (repeat_index // 2) % count
        order = order[shift:] + order[:shift]
    return order


def _run_once(
    item: JudgeItem,
    rubric: Rubric,
    call_fn: JudgeCall,
    *,
    repeat_index: int,
    repair_invalid: bool,
) -> dict:
    order = _reference_order(len(item.reference_answers), repeat_index)
    references = [item.reference_answers[index] for index in order]
    system_prompt = build_system_prompt(rubric)
    user_prompt = build_user_prompt(
        task=item.task,
        candidate_answer=item.candidate_answer,
        reference_answers=references,
    )
    try:
        initial_raw = call_fn(system_prompt, user_prompt)
    except Exception as exc:
        return {
            "repeat_index": repeat_index,
            "reference_order": order,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "initial_raw": None,
            "raw": None,
            "parsed": None,
            "valid": False,
            "repaired": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    parsed_raw, parse_error = safe_json_object(initial_raw)
    parsed, validation_error = validate_judgment(
        parsed_raw,
        rubric,
        reference_count=len(references),
    )
    error = parse_error or validation_error
    repaired = False
    raw = initial_raw

    if error and repair_invalid:
        repair_prompt = build_repair_prompt(error, initial_raw)
        try:
            raw = call_fn(system_prompt, repair_prompt)
            repaired = True
        except Exception as exc:
            return {
                "repeat_index": repeat_index,
                "reference_order": order,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "initial_raw": initial_raw,
                "raw": None,
                "parsed": None,
                "valid": False,
                "repaired": True,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        parsed_raw, parse_error = safe_json_object(raw)
        parsed, validation_error = validate_judgment(
            parsed_raw,
            rubric,
            reference_count=len(references),
        )
        error = parse_error or validation_error

    parsed_dict = parsed.as_dict() if parsed is not None else None
    if parsed_dict is not None and parsed_dict["matched_reference_index"] is not None:
        displayed_index = parsed_dict["matched_reference_index"]
        parsed_dict["matched_reference_index"] = order[displayed_index]

    return {
        "repeat_index": repeat_index,
        "reference_order": order,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "initial_raw": initial_raw,
        "raw": raw,
        "parsed": parsed_dict,
        "valid": error is None,
        "repaired": repaired,
        "error": None if error is None else {"type": "ValidationError", "message": error},
    }


def _aggregate(repetitions: list[dict], rubric: Rubric) -> dict:
    valid = [repeat["parsed"] for repeat in repetitions if repeat["valid"]]
    labels = [item["label"] for item in valid]
    counts = Counter(labels)
    valid_count = len(valid)
    max_count = max(counts.values(), default=0)
    leaders = sorted(label for label, count in counts.items() if count == max_count)
    has_valid_majority = valid_count > len(repetitions) / 2
    label = (
        leaders[0]
        if has_valid_majority and len(leaders) == 1
        else "inconclusive"
    )
    scored_values = [item["score"] for item in valid if item["score"] is not None]
    consistency = max_count / valid_count if valid_count else None
    passed = label in rubric.pass_labels if label in rubric.labels else None
    needs_review = (
        label in {"inconclusive", "partially_correct", "not_scorable"}
        or valid_count != len(repetitions)
        or len(counts) != 1
        or any(
            item["material_contradiction"]
            or item["unsupported_clinical_claim"]
            for item in valid
        )
    )
    return {
        "label": label,
        "passed": passed,
        "mean_score": sum(scored_values) / len(scored_values) if scored_values else None,
        "score_std": (
            math.sqrt(
                sum(
                    (value - (sum(scored_values) / len(scored_values))) ** 2
                    for value in scored_values
                )
                / len(scored_values)
            )
            if scored_values
            else None
        ),
        "label_counts": dict(sorted(counts.items())),
        "valid_repetitions": valid_count,
        "failed_repetitions": len(repetitions) - valid_count,
        "consistency_rate": consistency,
        "unanimous": valid_count == len(repetitions) and len(counts) == 1,
        "needs_human_review": needs_review,
    }


def evaluate_item(
    item: JudgeItem,
    rubric: Rubric,
    call_fn: JudgeCall,
    *,
    repeats: int = 3,
    repair_invalid: bool = True,
) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    repetitions = [
        _run_once(
            item,
            rubric,
            call_fn,
            repeat_index=index,
            repair_invalid=repair_invalid,
        )
        for index in range(repeats)
    ]
    return {
        "item": asdict(item),
        "rubric": {
            "rubric_id": rubric.rubric_id,
            "version": rubric.version,
            "sha256": rubric.sha256,
        },
        "repetitions": repetitions,
        "aggregate": _aggregate(repetitions, rubric),
    }
