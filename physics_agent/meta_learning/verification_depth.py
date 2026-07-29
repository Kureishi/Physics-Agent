"""
Verification Depth Policy (Stage 7).

Tracks, per domain tag, whether the confidence this system reports on its
own solutions is well-calibrated against ITS OWN eventual verification
outcome -- not ground-truth correctness, which this system deliberately
never claims to know (see trace.py's note on final_correct). "Calibrated"
here specifically means: does high self-reported confidence correlate with
the trace resolving cleanly, and low confidence with it ending up
unresolved? That's a proxy for correctness, not correctness itself.

If a domain shows a pattern of high average confidence alongside a
substantial rate of "unresolved_max_revisions" endings (overconfidence),
this recommends RAISING that domain's ConfidenceCheck threshold --
forcing it to fail more readily there, which pushes more attempts through
Stage 4's escalation/correction path before being accepted. This is the
"increase mandatory verification depth" adjustment from the design doc.

This deliberately only ever recommends raising the threshold above the
system default, never lowering it below it. Lowering would trade safety
for speed based on a proxy signal; raising trades speed for caution based
on the same proxy. Only the second trade is one this system makes
automatically -- the first would need a stronger justification than "this
domain's proxy signal looks fine so far."
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from ..trace import EpisodicMemory

MIN_TRACES_BEFORE_ACTING = 5
OVERCONFIDENCE_GAP_THRESHOLD = 0.15
MAX_THRESHOLD = 0.9
RAISE_STEP = 0.1


class VerificationDepthPolicy:
    def __init__(self, episodic_memory: EpisodicMemory):
        self.episodic_memory = episodic_memory
        self._stats = self._compute_stats()

    def _compute_stats(self) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"n": 0, "n_unresolved": 0, "confidence_sum": 0.0}
        )
        for trace in self.episodic_memory.read_all():
            if trace.resolution_status is None or trace.final_confidence is None:
                continue
            for tag in trace.domain_tags:
                s = stats[tag]
                s["n"] += 1
                s["confidence_sum"] += trace.final_confidence
                if trace.resolution_status == "unresolved_max_revisions":
                    s["n_unresolved"] += 1
        return stats

    def overconfidence_gap(self, domain_tag: str) -> Optional[float]:
        """
        avg_confidence - trust_warranted, where trust_warranted is the
        fraction of traces in this domain that did NOT end up unresolved.
        Positive means the domain reports more confidence than its own
        outcomes justify. None if there isn't enough data yet.
        """
        s = self._stats.get(domain_tag)
        if s is None or s["n"] < MIN_TRACES_BEFORE_ACTING:
            return None
        avg_confidence = s["confidence_sum"] / s["n"]
        trust_warranted = 1 - (s["n_unresolved"] / s["n"])
        return avg_confidence - trust_warranted

    def recommended_confidence_threshold(self, domain_tags: List[str], default_threshold: float) -> float:
        gaps = [
            g for g in (self.overconfidence_gap(tag) for tag in domain_tags) if g is not None
        ]
        if not gaps:
            return default_threshold

        max_gap = max(gaps)
        if max_gap <= OVERCONFIDENCE_GAP_THRESHOLD:
            return default_threshold

        raise_amount = RAISE_STEP * (max_gap / OVERCONFIDENCE_GAP_THRESHOLD)
        return min(MAX_THRESHOLD, default_threshold + raise_amount)
