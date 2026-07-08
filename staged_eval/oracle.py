"""
Automatic routing oracle for staged PubMedQA traces.
"""
from typing import Optional


FINAL_ANSWERS = {"yes", "no", "maybe"}


def expected_action(
    *,
    is_final: bool,
    gt_label: Optional[str],
    prior_answer: Optional[str],
) -> str:
    """
    Return the expected next action for a stage.

    The basic PubMedQA runner treats non-final stages as evidence-gathering
    stages. On the final stage, a wrong prior answer should be revised; otherwise
    the agent should answer. If no gold label is available, abstention is the
    only automatically scorable final action.
    """
    if not is_final:
        return "RETRIEVE_EVIDENCE"
    if gt_label not in FINAL_ANSWERS:
        return "ABSTAIN"
    if prior_answer in FINAL_ANSWERS and prior_answer != gt_label:
        return "REVISE_ANSWER"
    return "ANSWER"


def expected_action_for_trace(trace: dict, stage_index: int, gt: Optional[dict] = None) -> str:
    """Compute the expected action for an existing trace stage."""
    stage = trace["stages"][stage_index]
    if "expected_action" in stage:
        return stage["expected_action"]
    gt_label = trace.get("gt")
    if gt is not None:
        gt_label = gt.get(trace["pmid"], gt_label)
    prior_answer = None
    if stage_index > 0:
        prior_answer = trace["stages"][stage_index - 1]["parsed"].get("answer")
    return expected_action(
        is_final=stage["is_final"],
        gt_label=gt_label,
        prior_answer=prior_answer,
    )
