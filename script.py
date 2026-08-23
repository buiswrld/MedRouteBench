#!/usr/bin/env python3
"""Batch-run MedRouteBench pipelines from a declarative list of experiments.

Runs everything in-process, sequentially, by calling each pipeline's
run_pipeline() directly (no subprocesses). Must be run from inside
MedRouteBench (`python script.py`) so the medcta_eval / medcta_golden_eval /
staged_eval packages are importable.

Edit EXPERIMENTS below, then:

    python script.py            # run everything
    python script.py --dry-run  # print what would run, without running it

Each entry is a dict:

    {"experiment": "medcta_golden", "runs": 3, "model": "openai/gpt-5.4", "api_key": "sk-or-..."}

- "experiment": one of "medcta", "medcta_golden", "staged"
- "runs": how many independent replicate runs to launch (each becomes its
  own timestamped run directory under <package>/runs/ automatically)
- "api_key" is optional -- omit it (or leave it empty) to fall back to
  OPENROUTER_API_KEY in MedRouteBench/.env. Prefer omitting it and using
  .env/shell env vars so keys don't end up sitting in this file.
- Any other field is passed straight through to run_pipeline(), e.g.
  "n": 107, "workers": 4, "out": "some/dir", "cases": "path/to/cases.json".
  staged_eval also accepts "ground_truth", "reversed", "use_fixtures",
  "no_stratify". See PACKAGE_FIELDS below for the exact set per experiment.
"""
import argparse
import sys

import medcta_eval.pipeline as medcta_pipeline
import medcta_golden_eval.pipeline as medcta_golden_pipeline
import staged_eval.pipeline as staged_pipeline

EXPERIMENTS = [
    {"experiment": "medcta_golden", "runs": 3, "model": "anthropic/claude-opus-4.6", "api_key": ""}
]

EXPERIMENT_MODULES = {
    "medcta": medcta_pipeline,
    "medcta_eval": medcta_pipeline,
    "medcta_golden": medcta_golden_pipeline,
    "medcta_golden_eval": medcta_golden_pipeline,
    "staged": staged_pipeline,
    "staged_eval": staged_pipeline,
}

RESERVED_KEYS = {"experiment", "runs"}

# entry field -> run_pipeline() kwarg, shared by every pipeline
_COMMON_FIELDS = {
    "model": "model",
    "api_key": "api_key",
    "n": "n",
    "workers": "workers",
    "out": "out_dir",
}

# medcta_eval / medcta_golden_eval accept these on top of the common fields
_MEDCTA_FIELDS = {**_COMMON_FIELDS, "cases": "cases_path", "case_ids": "case_ids"}

# staged_eval accepts these on top of the common fields ("reversed" and
# "no_stratify" need value transforms, handled separately below)
_STAGED_FIELDS = {
    **_COMMON_FIELDS,
    "cases": "cases_path",
    "ground_truth": "ground_truth_path",
    "use_fixtures": "use_fixtures",
}

PACKAGE_FIELDS = {
    medcta_pipeline: _MEDCTA_FIELDS,
    medcta_golden_pipeline: _MEDCTA_FIELDS,
    staged_pipeline: _STAGED_FIELDS,
}


def build_kwargs(entry: dict, module) -> dict:
    fields = PACKAGE_FIELDS[module]
    kwargs = {"verbose": True}
    for key, value in entry.items():
        if key in RESERVED_KEYS:
            continue
        if key in fields:
            kwargs[fields[key]] = value
        elif module is staged_pipeline and key == "reversed":
            kwargs["reversed_order"] = value
        elif module is staged_pipeline and key == "no_stratify":
            kwargs["stratify"] = not value
        else:
            raise ValueError(
                f"field {key!r} is not supported for experiment "
                f"{entry.get('experiment')!r} in entry {entry}"
            )
    return kwargs


def redact(kwargs: dict) -> dict:
    """Copy safe to print: never echo a real API key to the terminal."""
    if "api_key" in kwargs and kwargs["api_key"]:
        return {**kwargs, "api_key": "***"}
    return kwargs


def validate(experiments: list[dict]) -> list[str]:
    problems = []
    for i, entry in enumerate(experiments):
        exp = entry.get("experiment")
        module = EXPERIMENT_MODULES.get(exp) if isinstance(exp, str) else None
        if module is None:
            problems.append(
                f"entry {i}: unknown experiment {exp!r} "
                f"(known: {sorted(set(EXPERIMENT_MODULES))})"
            )
            continue
        runs = entry.get("runs", 1)
        if not isinstance(runs, int) or runs < 1:
            problems.append(f"entry {i}: 'runs' must be a positive integer, got {runs!r}")
        try:
            build_kwargs(entry, module)
        except ValueError as exc:
            problems.append(f"entry {i}: {exc}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run MedRouteBench pipelines")
    parser.add_argument(
        "--dry-run", action="store_true", help="print what would run, without running it"
    )
    args = parser.parse_args()

    problems = validate(EXPERIMENTS)
    if problems:
        print("Fix these before running:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        sys.exit(1)

    total_runs = sum(entry.get("runs", 1) for entry in EXPERIMENTS)
    print(f"{len(EXPERIMENTS)} experiment entr{'y' if len(EXPERIMENTS) == 1 else 'ies'}, "
          f"{total_runs} total run(s), sequential{' (dry run)' if args.dry_run else ''}")

    results = []
    run_number = 0
    for entry in EXPERIMENTS:
        module = EXPERIMENT_MODULES[entry["experiment"]]
        runs = entry.get("runs", 1)
        kwargs = build_kwargs(entry, module)
        for replicate in range(1, runs + 1):
            run_number += 1
            label = (
                f"[{run_number}/{total_runs}] {entry['experiment']} "
                f"model={entry.get('model', '<default>')} replicate {replicate}/{runs}"
            )
            print(f"\n=== {label} ===")
            print(f"  {module.__name__}.run_pipeline(**{redact(kwargs)!r})")
            if args.dry_run:
                results.append((label, "dry-run"))
                continue
            try:
                module.run_pipeline(**kwargs)
                status = "ok"
            except Exception as exc:  # noqa: BLE001 - keep the batch going on any failure
                status = f"FAILED: {exc}"
                print(f"  -> {status}, continuing with remaining runs", file=sys.stderr)
            results.append((label, status))

    print("\n=== Summary ===")
    for label, status in results:
        print(f"  {status:22s} {label}")


if __name__ == "__main__":
    main()
