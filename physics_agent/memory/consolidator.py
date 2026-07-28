"""
Memory Consolidator (Stage 5) -- the "Memory + Knowledge Update" box in the
architecture diagram.

This is the actual hinge between "solving" (Stages 1-4, which produce one
answer to one problem) and "learning" (this stage onward, which is
supposed to make the *next* problem easier). Called once, after Stage 4
finishes with a trace, it writes to all four memory types:

  - Episodic:   the full trace, appended (already just EpisodicMemory.write,
                wrapped here for a single call site).
  - Semantic:   nudges the confidence of every retrieved_knowledge fact
                based on whether the final candidate passed verification.
  - Procedural: for every revision round, records whether the strategy
                tried actually resolved what it was meant to fix.
  - Error:      for every revision round, records the failure signature,
                root cause, fix applied, and bumps its recurrence frequency.

Note what this deliberately does NOT do yet: it doesn't change
error_taxonomy's fixed strategy choices, doesn't adjust tool-selection
policy, and doesn't build a knowledge graph. It only writes the data those
later (meta-learning) decisions would need. Consolidation and adaptation
are different steps for the same reason solving and learning are --
mixing "record what happened" with "change future behavior based on it" in
one component makes both harder to get right and harder to audit.
"""
from __future__ import annotations

from .error_memory import ErrorMemory
from .procedural import ProceduralMemory
from ..retrieval import SemanticStore
from ..trace import EpisodicMemory, Trace


class MemoryConsolidator:
    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticStore,
        procedural: ProceduralMemory,
        error: ErrorMemory,
    ):
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural
        self.error = error

    def consolidate(self, trace: Trace) -> None:
        self._update_semantic(trace)
        self._update_procedural_and_error(trace)
        self.episodic.write(trace)

    def _update_semantic(self, trace: Trace) -> None:
        """
        A candidate that ultimately passed every check is (weak, but real)
        evidence the retrieved facts it leaned on were applied correctly;
        one that finished unresolved is the opposite signal. Deliberately
        coarse: it doesn't try to attribute credit/blame among several
        retrieved facts individually, since we have no way to know which
        one the final solution actually depended on most.
        """
        success = not trace.checks_failed
        for fact in trace.retrieved_knowledge:
            fact_id = fact.get("id")
            if fact_id:
                self.semantic.record_outcome(fact_id, success=success)

    def _update_procedural_and_error(self, trace: Trace) -> None:
        for round_record in trace.revision_history:
            self.procedural.record_outcome(
                domain_tags=trace.domain_tags,
                error_type=round_record["error_type"],
                strategy=round_record["strategy"],
                resolved=bool(round_record["resolved"]),
            )
            self.error.record(
                error_type=round_record["error_type"],
                domain_tags=trace.domain_tags,
                root_cause=round_record["rationale"],
                fix_applied=round_record["strategy"],
                resolved=bool(round_record["resolved"]),
            )
