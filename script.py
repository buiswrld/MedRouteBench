#!/usr/bin/env python3
"""Batch-run MedRouteBench pipelines from a declarative list of experiments.

Edit EXPERIMENTS below, then:

    python script.py            # run everything
    python script.py --dry-run  # print the commands without running them

Each entry is a dict:

    {"experiment": "medcta_golden", "runs": 3, "model": "openai/gpt-5.4", "api_key": "sk-or-..."}

- "experiment": one of "medcta", "medcta_golden", "staged"
- "runs": how many independent replicate runs to launch (each becomes its
  own timestamped run directory under <package>/runs/ automatically)
- "api_key" is optional -- omit it to fall back to OPENROUTER_API_KEY in
  MedRouteBench/.env. Prefer omitting it and using .env/shell env vars so
  keys don't end up sitting in this file.
- Any other field is passed straight through as a CLI flag, e.g. "n": 107,
  "workers": 4, "out": "some/dir", "reversed": True, "use_fixtures": True,
  "no_stratify": True (see VALUE_FLAGS / BOOLEAN_FLAGS below for the full
  set of recognized fields).
"""
import argparse
import subprocess
import sys
from pathlib import Path

MEDROUTEBENCH_DIR = Path(__file__).resolve().parent

EXPERIMENTS = [
    {"experiment": "medcta_golden", "runs": 3, "model": "anthropic/claude-opus-4.6", "api_key": ""}
]

EXPERIMENT_PACKAGES = {
    "medcta": "medcta_eval",
    "medcta_eval": "medcta_eval",
    "medcta_golden": "medcta_golden_eval",
    "medcta_golden_eval": "medcta_golden_eval",
    "staged": "staged_eval",
    "staged_eval": "staged_eval",
}

RESERVED_KEYS = {"experiment", "runs"}

# entry field -> CLI flag that takes a value
VALUE_FLAGS = {
    "model": "--model",
    "api_key": "--api-key",
    "n": "--n",
    "workers": "--workers",
    "out": "--out",
    "cases": "--cases",
    "ground_truth": "--ground-truth",
    "case_ids": "--case-ids",
}

# entry field -> CLI flag that takes no value (present only if truthy)
BOOLEAN_FLAGS = {
    "reversed": "--reversed",
    "use_fixtures": "--use-fixtures",
    "no_stratify": "--no-stratify",
}


def build_command(entry: dict, package: str) -> list[str]:
    cmd = [sys.executable, "-m", f"{package}.pipeline"]
    for key, value in entry.items():
        if key in RESERVED_KEYS:
            continue
        if key in VALUE_FLAGS:
            cmd += [VALUE_FLAGS[key], str(value)]
        elif key in BOOLEAN_FLAGS:
            if value:
                cmd.append(BOOLEAN_FLAGS[key])
        else:
            raise ValueError(f"unknown experiment field {key!r} in entry {entry}")
    return cmd


def redact(cmd: list[str]) -> str:
    """Command string safe to print: never echo a real API key to the terminal."""
    printable = list(cmd)
    for i, token in enumerate(printable):
        if token == "--api-key" and i + 1 < len(printable):
            printable[i + 1] = "***"
    return " ".join(printable)


def validate(experiments: list[dict]) -> list[str]:
    problems = []
    for i, entry in enumerate(experiments):
        exp = entry.get("experiment")
        package = EXPERIMENT_PACKAGES.get(exp) if isinstance(exp, str) else None
        if package is None:
            problems.append(
                f"entry {i}: unknown experiment {exp!r} "
                f"(known: {sorted(set(EXPERIMENT_PACKAGES))})"
            )
            continue
        runs = entry.get("runs", 1)
        if not isinstance(runs, int) or runs < 1:
            problems.append(f"entry {i}: 'runs' must be a positive integer, got {runs!r}")
        try:
            build_command(entry, package)
        except ValueError as exc:
            problems.append(f"entry {i}: {exc}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run MedRouteBench pipelines")
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without running them"
    )
    args = parser.parse_args()

    if not MEDROUTEBENCH_DIR.is_dir():
        print(f"MedRouteBench directory not found at {MEDROUTEBENCH_DIR}", file=sys.stderr)
        sys.exit(1)

    problems = validate(EXPERIMENTS)
    if problems:
        print("Fix these before running:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        sys.exit(1)

    total_runs = sum(entry.get("runs", 1) for entry in EXPERIMENTS)
    print(f"{len(EXPERIMENTS)} experiment entr{'y' if len(EXPERIMENTS) == 1 else 'ies'}, "
          f"{total_runs} total run(s){' (dry run)' if args.dry_run else ''}")

    results = []
    run_number = 0
    for entry in EXPERIMENTS:
        package = EXPERIMENT_PACKAGES[entry["experiment"]]
        runs = entry.get("runs", 1)
        cmd = build_command(entry, package)
        for replicate in range(1, runs + 1):
            run_number += 1
            label = (
                f"[{run_number}/{total_runs}] {entry['experiment']} "
                f"model={entry.get('model', '<default>')} replicate {replicate}/{runs}"
            )
            print(f"\n=== {label} ===")
            print(f"  $ {redact(cmd)}")
            if args.dry_run:
                results.append((label, "dry-run"))
                continue
            proc = subprocess.run(cmd, cwd=MEDROUTEBENCH_DIR)
            status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
            results.append((label, status))
            if proc.returncode != 0:
                print(f"  -> {status}, continuing with remaining runs", file=sys.stderr)

    print("\n=== Summary ===")
    for label, status in results:
        print(f"  {status:22s} {label}")


if __name__ == "__main__":
    main()
