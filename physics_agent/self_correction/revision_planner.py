"""
Revision Planner (Stage 4) -- the "Revision Planner" box in the
architecture diagram.

Takes a correction `strategy` (from error_taxonomy.classify_error) and
carries it out via the ToolOrchestrator, producing an updated candidate
solution (trace.initial_solution is overwritten in place -- this is the
"Updated Candidate Solution" box in the diagram).

Each strategy is deliberately a different amount of work, matched to what
actually needs fixing:
  - rederive_math / rederive_physics_setup / full_replan: redo tool
    selection + execution + synthesis, with feedback about what went wrong
    fed back into both prompts.
  - resynthesize: the tools and physics were fine; only the write-up's
    reasoning was inconsistent, so just redo the synthesis step.
  - escalate_verification: nothing specific failed, confidence is just
    low; pull in one more independent signal (literature search) rather
    than repeating work that already passed every check.
"""
from __future__ import annotations

from ..orchestrator import ToolOrchestrator
from ..trace import Trace

_STRATEGY_EMPHASIS = {
    "rederive_math": (
        "Focus especially on re-deriving the algebra correctly -- double-check "
        "every substitution and the equation itself before solving."
    ),
    "rederive_physics_setup": (
        "Focus especially on re-checking the physical setup: which formula "
        "applies, whether its conditions of validity are met, and whether "
        "conservation laws are respected given the numbers involved."
    ),
    "full_replan": "Reconsider the approach from scratch.",
}


class RevisionPlanner:
    def __init__(self, orchestrator: ToolOrchestrator):
        self.orchestrator = orchestrator

    def apply(self, strategy: str, trace: Trace, feedback: str) -> Trace:
        if strategy in _STRATEGY_EMPHASIS:
            full_feedback = f"{feedback}\n{_STRATEGY_EMPHASIS[strategy]}"
            self.orchestrator.run(trace, feedback=full_feedback)
        elif strategy == "resynthesize":
            self.orchestrator.resynthesize(trace, feedback=feedback)
        elif strategy == "escalate_verification":
            self.orchestrator.escalate_with_literature_search(trace, feedback=feedback)
        else:
            raise ValueError(f"Unknown correction strategy: {strategy!r}")
        return trace
