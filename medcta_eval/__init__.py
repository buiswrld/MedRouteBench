"""MedCTA reference-trajectory evaluator for MedRouteBench Idea #3."""

from .config import DATA_PATH, GROQ_MODEL
from .schema import ACTIONS, AgentOutput


def __getattr__(name):
    if name in {"run_pipeline", "inspect_trace"}:
        from .pipeline import inspect_trace, run_pipeline

        return {"run_pipeline": run_pipeline, "inspect_trace": inspect_trace}[name]
    if name == "run_case":
        from .runner import run_case

        return run_case
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "run_case",
    "run_pipeline",
    "inspect_trace",
    "ACTIONS",
    "AgentOutput",
    "DATA_PATH",
    "GROQ_MODEL",
]
