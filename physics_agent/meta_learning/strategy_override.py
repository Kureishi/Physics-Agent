"""
Strategy-Override Policy (Stage 7) -- closes the loop the design doc
explicitly called out as left undone: "ProceduralMemory.best_strategy_for
and ErrorMemory.most_frequent already exist and are tested -- but nothing
reads them to change behavior. error_taxonomy.py's strategy mapping...
is still a fixed lookup table. Once best_strategy_for has enough samples
showing a different strategy resolves a given (domain, error_type) pair
more often, the taxonomy should be able to use that instead of the
hardcoded default."

Unlike check_value.py, pruning.py, curriculum_signals.py, and anomaly.py
-- all deliberately report-only, "flag it, let a person decide" -- this is
the one meta-learning component built to change behavior live during
solving, joining ToolSelectionPolicy and VerificationDepthPolicy (see
meta_learning/report.py's docstring for why those three stay out of the
passive review report).

Consumed by SelfCorrectionEngine, not error_taxonomy.classify_error
itself: classify_error stays a pure, deterministic function of trace
state alone (useful on its own -- e.g. anomaly.py and tests reproduce its
output without needing procedural memory in scope). Deciding whether to
act on a *different* strategy is the engine's call, made once it also has
the domain context procedural memory needs.

Safety posture, deliberately more conservative than
ProceduralMemory.best_strategy_for's own floor -- consistent with
error_taxonomy's own "priority order matters" framing (which corrective
action runs isn't arbitrary):
  - the alternative strategy needs MIN_USES_TO_OVERRIDE uses for this
    exact (domain, error_type) pair, stricter than best_strategy_for's
    floor of 3 uses -- "trusted enough to report in a review" and
    "trusted enough to act on automatically, unreviewed" aren't the same
    bar.
  - if the default strategy also has real data for this exact
    combination (n_uses >= 3), the alternative must beat it by at least
    MIN_IMPROVEMENT_MARGIN, not just edge it out -- a bar for switching
    away from a strategy already known to work at all here.
  - if the default strategy has no (or too little) data here, there is
    nothing concrete to beat, so the alternative is judged on its own
    track record instead: it must clear MIN_ACCEPTABLE_SUCCESS_RATE in
    absolute terms, not merely be "the best option tried so far" (which,
    with few samples, could just mean "the only option tried so far").
  - never touches which error_type/check gets addressed first when
    several checks fail at once -- that ordering stays error_taxonomy's
    job; this only ever replaces which corrective action is used, once
    an error_type has already been picked.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from ..memory.procedural import ProceduralMemory

MIN_USES_TO_OVERRIDE = 5
MIN_IMPROVEMENT_MARGIN = 0.15
MIN_ACCEPTABLE_SUCCESS_RATE = 0.5


class StrategyOverridePolicy:
    def __init__(self, procedural_memory: ProceduralMemory):
        self.procedural_memory = procedural_memory

    def override(
        self, domain_tags: List[str], error_type: str, default_strategy: str
    ) -> Tuple[str, Optional[str]]:
        """
        Returns (strategy_to_use, override_reason). override_reason is
        None whenever default_strategy is kept -- whether because no
        alternative exists, the best alternative IS the default, or
        nothing yet clears the bars above -- so callers can check
        `if override_reason:` rather than compare strategy strings to
        detect whether anything changed.
        """
        best = self.procedural_memory.best_strategy_for(domain_tags, error_type)
        if best is None or best["strategy"] == default_strategy:
            return default_strategy, None
        if best["n_uses"] < MIN_USES_TO_OVERRIDE:
            return default_strategy, None

        default_entry = self.procedural_memory.get(domain_tags, error_type, default_strategy)

        if default_entry is not None and default_entry["n_uses"] >= 3:
            if best["success_rate"] - default_entry["success_rate"] < MIN_IMPROVEMENT_MARGIN:
                return default_strategy, None
            comparison = (
                f"vs. default '{default_strategy}' at {default_entry['success_rate']:.0%} "
                f"over {default_entry['n_uses']} uses"
            )
        else:
            if best["success_rate"] < MIN_ACCEPTABLE_SUCCESS_RATE:
                return default_strategy, None
            comparison = f"default '{default_strategy}' untried for this combination"

        reason = (
            f"procedural memory: '{best['strategy']}' resolved '{error_type}' in "
            f"{'+'.join(best['domain_tags'])} {best['success_rate']:.0%} of the time over "
            f"{best['n_uses']} uses ({comparison})"
        )
        return best["strategy"], reason
