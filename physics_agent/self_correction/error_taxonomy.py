"""
Error Taxonomy (Stage 4) -- the deterministic "self-correction mapping"
table from the design doc: detected check failure -> error_type ->
corrective strategy.

This is deliberately rule-based, not LLM-based. By the time this runs, the
Stage 3 checks have already produced structured signal (`checks_failed`,
`check_details`) -- classifying *that* structured signal doesn't need
another model call, and a deterministic classifier is exactly what makes
this table something Stage 7 (meta-learning) can later inspect, extend, or
retune based on which rows actually correlate with successful revisions.

Priority order matters: if multiple checks failed simultaneously, the
earliest-matching rule below is used, on the reasoning that some faults are
usually upstream causes of others (e.g. an algebra error can itself cause
what looks like a physics disagreement or a confused write-up, so fixing
the algebra first is the more useful first move).
"""
from __future__ import annotations

from typing import Dict, Tuple

from ..trace import Trace

# error_type: a short, stable label -- this is what would feed a knowledge
# graph / meta-learning stage later, so it's kept as a fixed vocabulary
# rather than free text.
#
# strategy: which RevisionPlanner strategy handles this error_type.
ErrorClassification = Tuple[str, str, str]  # (error_type, strategy, rationale)


def classify_error(trace: Trace) -> ErrorClassification:
    """
    Returns (error_type, strategy, rationale) based on trace.checks_failed
    and trace.check_details. Only meaningful to call when
    trace.checks_failed is non-empty.
    """
    failed = set(trace.checks_failed)
    details_by_check: Dict[str, str] = {d["check"]: d["details"] for d in trace.check_details}

    if "math" in failed:
        return (
            "algebra_error",
            "rederive_math",
            f"Math check failed (re-substitution did not satisfy the equation): "
            f"{details_by_check.get('math', '')}",
        )

    if "physics" in failed:
        physics_details = details_by_check.get("physics", "")
        if "disagrees" in physics_details:
            return (
                "cross_method_disagreement",
                "rederive_physics_setup",
                f"Symbolic math and simulation gave different answers: {physics_details}",
            )
        return (
            "physics_conceptual_error",
            "rederive_physics_setup",
            f"Physics check failed (dimensional/conservation/validity issue): {physics_details}",
        )

    if "logic" in failed:
        return (
            "reasoning_inconsistency",
            "resynthesize",
            f"Logic check failed (internal inconsistency in the write-up): "
            f"{details_by_check.get('logic', '')}",
        )

    if failed == {"confidence"}:
        return (
            "low_confidence_no_specific_fault",
            "escalate_verification",
            f"Confidence below threshold despite no specific check failing: "
            f"{details_by_check.get('confidence', '')}",
        )

    # Defensive fallback: with today's four checks (logic, physics, math,
    # confidence) every non-empty checks_failed combination is covered by
    # one of the branches above, since math/physics/logic take priority
    # over a confidence-only failure. This branch exists so a future
    # custom check that isn't one of those four still gets *some*
    # classification rather than crashing the pipeline.
    return (
        "multiple_faults",
        "full_replan",
        f"Multiple/unrecognized checks failed together: {sorted(failed)}",
    )
