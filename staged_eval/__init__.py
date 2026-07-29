"""MedRouteBench fixed two-stage PubMedQA revision experiment."""

from .config import AZURE_OPENAI_DEPLOYMENT, GROUND_TRUTH_PATH, TEST_SET_PATH
from .schema import ACTIONS, STAGE1_ALLOWED, STAGE2_ALLOWED, AgentOutput


def __getattr__(name):
    """Load pipeline entry points lazily so module CLI execution stays clean."""
    if name in {"run_pipeline", "inspect_trace"}:
        from .pipeline import inspect_trace, run_pipeline

        return {"run_pipeline": run_pipeline, "inspect_trace": inspect_trace}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "run_pipeline",
    "inspect_trace",
    "ACTIONS",
    "STAGE1_ALLOWED",
    "STAGE2_ALLOWED",
    "AgentOutput",
    "TEST_SET_PATH",
    "GROUND_TRUTH_PATH",
    "AZURE_OPENAI_DEPLOYMENT",
]
