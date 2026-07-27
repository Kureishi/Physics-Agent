"""
Self-Correction Engine (Stage 4).

Ties together: Stage 3's self-evaluation output -> error_taxonomy's Error
Detector -> RevisionPlanner's corrective action -> Stage 3 re-run, looping
until the candidate solution passes every check or `max_revisions` is
reached (the safety rail from the design doc -- never loop indefinitely
chasing self-consistency).

Before each revision, the current round's state (tool calls, solution,
check results) is archived into trace.revision_history, since
trace.tool_calls / trace.checks_failed / trace.check_details themselves are
overwritten each round to represent only the *current* candidate -- Stage 3's
checks are written to operate on "the current attempt," and mixing in stale
tool calls from a since-corrected earlier round would make e.g. MathCheck
fail forever on an old mistake that's no longer part of the answer. The
history list is what preserves the full story for later inspection /
meta-learning without breaking that invariant.
"""
from __future__ import annotations

import time
from dataclasses import asdict

from .error_taxonomy import classify_error
from .revision_planner import RevisionPlanner
from ..orchestrator import ToolOrchestrator
from ..self_eval.pipeline import SelfEvaluationPipeline
from ..trace import Trace


def _build_feedback(trace: Trace) -> str:
    lines = [
        f"- {d['check']} check failed: {d['details']}" for d in trace.check_details if not d["passed"]
    ]
    return "The previous attempt failed verification:\n" + "\n".join(lines)


class SelfCorrectionEngine:
    def __init__(
        self,
        orchestrator: ToolOrchestrator,
        self_eval_pipeline: SelfEvaluationPipeline,
        max_revisions: int = 3,
    ):
        self.orchestrator = orchestrator
        self.self_eval = self_eval_pipeline
        self.revision_planner = RevisionPlanner(orchestrator)
        self.max_revisions = max_revisions

    def run(self, trace: Trace) -> Trace:
        while True:
            if not trace.checks_failed:
                break  # current candidate already passes everything

            error_type, strategy, rationale = classify_error(trace)
            trace.error_type = error_type

            if trace.revision_count >= self.max_revisions:
                break  # safety rail: stop trying, ship best-effort as unresolved

            trace.revision_history.append(
                {
                    "round": trace.revision_count,
                    "error_type": error_type,
                    "rationale": rationale,
                    "tool_calls": [asdict(tc) for tc in trace.tool_calls],
                    "initial_solution": trace.initial_solution,
                    "checks_failed": list(trace.checks_failed),
                    "check_details": list(trace.check_details),
                }
            )

            trace.revision_count += 1
            feedback = _build_feedback(trace)
            self.revision_planner.apply(strategy, trace, feedback)

            # The checks above described the round that's now archived;
            # re-run Stage 3 fresh against the updated candidate.
            trace.checks_run = []
            trace.checks_failed = []
            trace.check_details = []
            self.self_eval.run(trace)

        trace.final_answer = trace.initial_solution
        trace.time_to_solve_ms = (time.time() - trace.timestamp) * 1000

        if trace.revision_count == 0:
            trace.resolution_status = "passed_initial"
        elif not trace.checks_failed:
            trace.resolution_status = "resolved_after_revision"
        else:
            trace.resolution_status = "unresolved_max_revisions"

        return trace
