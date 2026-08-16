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

Each archived round also records `strategy` (which correction was applied)
and `resolved` (did the specific check(s) that triggered this round's
correction actually stop failing afterward) -- Stage 5's memory
consolidation is what actually reads these two fields, so they're recorded
here, at the only point where both are known.

Escalation path (Safety Rails): `max_revisions` stops a runaway loop, but
"stop and mark unresolved" and "stop and ask a person" are different
outcomes, and until now this engine only had the first. The
`escalate_verification` strategy (error_taxonomy's row for "confidence low,
no specific check failed") already pulls in one independent check
(literature search) as its one, specific corrective action -- if
classify_error lands on escalate_verification again on a LATER round for
the same trace, that means the independent check didn't move confidence
either. At that point further revisions would just be guessing with no new
signal to act on, so the loop stops early with resolution_status
"escalated_for_human_review" instead of burning through the remaining
revision budget repeating a check that already didn't help. Pattern-level
escalation across many traces (e.g. one domain hitting
unresolved_max_revisions repeatedly) is a separate, outer-loop concern --
see self_correction/escalation.py.

Strategy override (Stage 7, closing the loop left open on purpose): when a
StrategyOverridePolicy is supplied, classify_error's hardcoded strategy for
this (domain, error_type) pair can be replaced with whatever procedural
memory has actually found to work best, once there's enough real data to
trust it (see meta_learning/strategy_override.py for the exact bars). This
happens BEFORE the escalation check above, using the strategy that will
actually be applied and archived -- so if procedural memory has learned a
better fix than "escalate and eventually give up" for a given confidence
issue, that fix is what runs, and the escalation path only fires when
there either isn't one yet or the taxonomy default is still winning.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Optional

from .error_taxonomy import classify_error
from .revision_planner import RevisionPlanner
from ..meta_learning.strategy_override import StrategyOverridePolicy
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
        strategy_override_policy: Optional[StrategyOverridePolicy] = None,
    ):
        self.orchestrator = orchestrator
        self.self_eval = self_eval_pipeline
        self.revision_planner = RevisionPlanner(orchestrator)
        self.max_revisions = max_revisions
        self.strategy_override_policy = strategy_override_policy

    def run(self, trace: Trace) -> Trace:
        while True:
            if not trace.checks_failed:
                break  # current candidate already passes everything

            error_type, strategy, rationale = classify_error(trace)
            trace.error_type = error_type

            if self.strategy_override_policy is not None:
                overridden_strategy, override_reason = self.strategy_override_policy.override(
                    trace.domain_tags, error_type, strategy
                )
                if override_reason:
                    strategy = overridden_strategy
                    rationale = f"{rationale} | {override_reason}"

            already_escalated = any(
                r["strategy"] == "escalate_verification" for r in trace.revision_history
            )
            if strategy == "escalate_verification" and already_escalated:
                # The one specific corrective action for this error_type
                # (an independent literature-search check) was already
                # tried on an earlier round and confidence is still low --
                # repeating it again wouldn't add new signal. Stop and
                # flag for a person instead of spending more of the
                # revision budget guessing.
                trace.resolution_status = "escalated_for_human_review"
                break

            if trace.revision_count >= self.max_revisions:
                break  # safety rail: stop trying, ship best-effort as unresolved

            checks_failed_before = list(trace.checks_failed)
            trace.revision_history.append(
                {
                    "round": trace.revision_count,
                    "error_type": error_type,
                    "strategy": strategy,
                    "rationale": rationale,
                    "tool_calls": [asdict(tc) for tc in trace.tool_calls],
                    "initial_solution": trace.initial_solution,
                    "checks_failed": checks_failed_before,
                    "check_details": list(trace.check_details),
                    "resolved": None,  # filled in below, once we know the outcome
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

            # "Resolved" means the specific check(s) that triggered this
            # round's correction are no longer failing -- not that every
            # check in the whole trace passes (a different check could
            # still be failing and get its own round next iteration).
            still_failing = set(checks_failed_before) & set(trace.checks_failed)
            trace.revision_history[-1]["resolved"] = len(still_failing) == 0

        trace.final_answer = trace.initial_solution
        trace.time_to_solve_ms = (time.time() - trace.timestamp) * 1000

        if trace.resolution_status == "escalated_for_human_review":
            pass  # already set inside the loop -- don't overwrite it below
        elif trace.revision_count == 0:
            trace.resolution_status = "passed_initial"
        elif not trace.checks_failed:
            trace.resolution_status = "resolved_after_revision"
        else:
            trace.resolution_status = "unresolved_max_revisions"

        return trace
