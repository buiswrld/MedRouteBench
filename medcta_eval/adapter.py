"""Reproducible adapter from the pinned MedCTA release to the v1 subset."""

import argparse
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Sequence

from .config import (
    DATA_PATH,
    SOURCE_DATASET,
    SOURCE_IMAGE_BASE_URL,
    SOURCE_PAGE_URL,
    SOURCE_RAW_URL,
    SOURCE_REVISION,
    STARTER_CASE_IDS,
)


ALLOWED_TOOLS = {
    "OCR",
    "ImageDescription",
    "RegionAttributeDescription",
    "GoogleSearch",
    "Calculator",
}


def fetch_raw_dataset(source: str = SOURCE_RAW_URL) -> dict:
    """Load the raw MedCTA JSON from a URL or local path."""
    candidate = Path(source)
    if candidate.is_file():
        with open(candidate, encoding="utf-8") as handle:
            return json.load(handle)
    request = urllib.request.Request(
        source,
        headers={"User-Agent": "MedRouteBench-MedCTA-adapter/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _flatten_strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_strings(item))
        return flattened
    return []


def _observation_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict) and isinstance(content.get("content"), str):
        return content["content"]
    raise ValueError("tool observation does not contain text content")


def _adapt_case(case_id: str, raw_case: dict, revision: str) -> dict:
    tools = raw_case.get("tools")
    files = raw_case.get("files")
    dialogs = raw_case.get("dialogs")
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"case {case_id}: missing tools")
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError(f"case {case_id}: expected exactly one image")
    if not isinstance(dialogs, list) or len(dialogs) < 4:
        raise ValueError(f"case {case_id}: malformed dialogs")

    tool_names = [tool.get("name") for tool in tools]
    if len(tool_names) != len(set(tool_names)):
        raise ValueError(f"case {case_id}: duplicate available tool names")
    if any(name not in ALLOWED_TOOLS for name in tool_names):
        raise ValueError(f"case {case_id}: unsupported tool in {tool_names}")

    first = dialogs[0]
    if first.get("role") != "user" or not isinstance(first.get("content"), str):
        raise ValueError(f"case {case_id}: first dialog must be a user question")

    image_path = files[0].get("path")
    if files[0].get("type") != "image" or not isinstance(image_path, str):
        raise ValueError(f"case {case_id}: invalid image reference")

    reference_steps = []
    cursor = 1
    while cursor < len(dialogs):
        assistant = dialogs[cursor]
        if assistant.get("role") != "assistant":
            raise ValueError(f"case {case_id}: expected assistant at dialog {cursor}")
        tool_calls = assistant.get("tool_calls")
        if not tool_calls:
            if cursor != len(dialogs) - 1:
                raise ValueError(f"case {case_id}: final answer is not terminal")
            final_answer = assistant.get("content")
            if not isinstance(final_answer, str) or not final_answer.strip():
                raise ValueError(f"case {case_id}: missing final answer")
            reference_steps.append(
                {
                    "step_index": len(reference_steps),
                    "action": "FINAL_ANSWER",
                    "tool_name": None,
                    "reference_arguments": None,
                    "reference_observation": None,
                    "reference_answer": final_answer.strip(),
                }
            )
            cursor += 1
            break

        if not isinstance(tool_calls, list) or not tool_calls:
            raise ValueError(f"case {case_id}: malformed tool calls")
        if cursor + len(tool_calls) >= len(dialogs):
            raise ValueError(f"case {case_id}: tool call lacks an observation")

        # The evaluator requests one action per model turn. MedCTA case 19 has
        # two calls in one assistant turn, so replay them deterministically in
        # source order and reveal each matching saved observation in sequence.
        for parallel_index, tool_call in enumerate(tool_calls):
            call = tool_call.get("function") or {}
            tool_name = call.get("name")
            arguments = call.get("arguments")
            observation = dialogs[cursor + 1 + parallel_index]
            if tool_name not in tool_names:
                raise ValueError(
                    f"case {case_id}: reference uses unavailable {tool_name}"
                )
            if not isinstance(arguments, dict):
                raise ValueError(f"case {case_id}: tool arguments must be an object")
            if (
                observation.get("role") != "tool"
                or observation.get("name") != tool_name
            ):
                raise ValueError(f"case {case_id}: call/observation tool mismatch")
            reference_step = {
                "step_index": len(reference_steps),
                "action": "CALL_TOOL",
                "tool_name": tool_name,
                "reference_arguments": arguments,
                "reference_observation": _observation_text(
                    observation.get("content")
                ),
                "reference_answer": None,
            }
            if len(tool_calls) > 1:
                reference_step["source_parallel_call"] = {
                    "assistant_dialog_index": cursor,
                    "call_index": parallel_index,
                    "call_count": len(tool_calls),
                    "policy": "sequentialized_in_source_order",
                }
            reference_steps.append(reference_step)
        cursor += 1 + len(tool_calls)

    if cursor != len(dialogs) or reference_steps[-1]["action"] != "FINAL_ANSWER":
        raise ValueError(f"case {case_id}: trajectory lacks one terminal answer")

    gt = raw_case.get("gt_answer") or {}
    accepted_answers = _flatten_strings(gt.get("whitelist"))
    if not accepted_answers:
        raise ValueError(f"case {case_id}: missing ground-truth whitelist")

    return {
        "case_id": case_id,
        "question": first["content"].strip(),
        "image_path": image_path,
        "image_reference": f"{SOURCE_PAGE_URL}/resolve/{revision}/{image_path}",
        "available_tools": [
            {
                "name": tool["name"],
                "description": str(tool.get("description") or "").strip(),
                "inputs": tool.get("inputs") or [],
            }
            for tool in tools
        ],
        "reference_steps": reference_steps,
        "ground_truth": {
            "accepted_answers": accepted_answers,
            "blacklist": gt.get("blacklist"),
        },
    }


def adapt_raw_dataset(
    raw: dict,
    case_ids: Sequence[str] = STARTER_CASE_IDS,
    *,
    revision: str = SOURCE_REVISION,
    source_url: str = SOURCE_RAW_URL,
) -> dict:
    """Adapt a deterministic subset and validate each trajectory."""
    if not isinstance(raw, dict):
        raise ValueError("raw MedCTA dataset must be an object keyed by case ID")
    normalized_ids = [str(case_id) for case_id in case_ids]
    if not normalized_ids or len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("case IDs must be nonempty and unique")
    missing = [case_id for case_id in normalized_ids if case_id not in raw]
    if missing:
        raise ValueError(f"raw MedCTA dataset is missing case IDs {missing}")

    return {
        "schema_version": 1,
        "dataset": {
            "name": SOURCE_DATASET,
            "license": "apache-2.0",
            "source_page": SOURCE_PAGE_URL,
            "source_url": source_url,
            "source_revision": revision,
            "image_base_url": SOURCE_IMAGE_BASE_URL,
            "parallel_call_policy": "sequentialized_in_source_order",
        },
        "selected_case_ids": normalized_ids,
        "cases": [_adapt_case(case_id, raw[case_id], revision) for case_id in normalized_ids],
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _parse_case_ids(value: str) -> list[str]:
    ids: list[str] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(part.strip()) for part in token.split("-", 1))
            if end < start:
                raise ValueError(f"invalid descending ID range: {token}")
            ids.extend(str(index) for index in range(start, end + 1))
        else:
            ids.append(str(int(token)))
    return ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m medcta_eval.adapter",
        description="Build or verify the pinned MedCTA starter subset",
    )
    parser.add_argument("--source", default=SOURCE_RAW_URL, help="raw JSON URL or path")
    parser.add_argument("--output", default=str(DATA_PATH), help="adapted JSON path")
    parser.add_argument("--ids", default="0-10", help="comma-separated IDs/ranges")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare with the committed subset without writing",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    case_ids = _parse_case_ids(args.ids)
    adapted = adapt_raw_dataset(
        fetch_raw_dataset(args.source),
        case_ids,
        source_url=args.source,
    )
    output = Path(args.output)
    if args.check:
        if not output.is_file():
            print(f"missing adapted subset: {output}")
            return 1
        with open(output, encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved != adapted:
            print(f"adapted subset differs from {output}")
            return 1
        print(f"verified {len(case_ids)} MedCTA cases in {output}")
        return 0

    _atomic_write_json(output, adapted)
    print(f"wrote {len(case_ids)} MedCTA cases to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
