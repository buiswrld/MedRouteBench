"""
Per-case staged runner.

`run_case` accepts an injectable `call_fn` so it can be driven by a stub in
tests without touching the network.
"""
from dataclasses import asdict
from typing import Callable, Optional

from .schema import AgentOutput, safe_json_loads, validate
from .data import n_stages, revealed_pairs
from .prompts import (
    SYSTEM_PROMPT as _DEFAULT_SYSTEM_PROMPT,
    REPAIR_TEMPLATE as _DEFAULT_REPAIR_TEMPLATE,
    build_user_prompt,
)
from .llm import call_json as _default_call_json


def run_case(
    case: dict,
    gt_label: Optional[str],
    *,
    call_fn: Optional[Callable] = None,
    system_prompt: Optional[str] = None,
    repair_template: Optional[str] = None,
) -> dict:
    """
    Run one case through all its stages and return a trace dict.

    Parameters
    ----------
    case         : PubMedQA case dict (must have 'pmid', 'QUESTION', 'CONTEXTS').
    gt_label     : Ground-truth label ("yes" / "no" / "maybe" / None).
    call_fn      : LLM call function ``(system: str, user: str) -> str``.
                   Defaults to ``llm.call_json``.
    system_prompt: Override the default SYSTEM_PROMPT.
    repair_template: Override the default REPAIR_TEMPLATE.
    """
    _call   = call_fn       if call_fn       is not None else _default_call_json
    _system = system_prompt if system_prompt is not None else _DEFAULT_SYSTEM_PROMPT
    _repair = repair_template if repair_template is not None else _DEFAULT_REPAIR_TEMPLATE

    total = n_stages(case)
    trace = {
        "pmid": case["pmid"],
        "gt": gt_label,
        "n_stages": total,
        "n_contexts": len(case["CONTEXTS"]),
        "stages": [],
    }
    prior_output: Optional[dict] = None

    for stage in range(total):
        user = build_user_prompt(case, stage, total, prior_output)
        raw = _call(_system, user)
        parsed_raw, _ = safe_json_loads(raw)
        parsed, err = validate(parsed_raw)
        parse_error = False

        if err:  # one repair attempt
            raw2 = _call(_system, _repair.format(ERROR=err, ORIGINAL=user))
            parsed_raw2, _ = safe_json_loads(raw2)
            parsed, err2 = validate(parsed_raw2)
            raw = raw2
            if err2:
                parsed = AgentOutput(
                    "ABSTAIN", None, 0.0,
                    f"parse_error: {err2}", None,
                )
                parse_error = True

        trace["stages"].append({
            "stage": stage,
            "is_final": stage == total - 1,
            "revealed_labels": [lbl for lbl, _ in revealed_pairs(case, stage)],
            "raw": raw,
            "parsed": asdict(parsed),
            "parse_error": parse_error,
        })
        prior_output = asdict(parsed)

    return trace
