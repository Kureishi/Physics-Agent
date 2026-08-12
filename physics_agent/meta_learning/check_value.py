"""
Check-Value Report (Stage 7).

For each Stage 3 check, computes how often it ever caught something across
a whole solve (failed in the final round or in any archived revision
round) versus how often it ran without ever finding an issue. This is the
signal the design doc's "verification check value" bullet calls for.

Deciding what to DO with a low catch rate (e.g. skipping a rarely-useful
check to save time) is deliberately left to a human or a future iteration.
With only four checks total, this system automatically disabling one on
its own say-so would trade away a safety guarantee for a speed gain that
hasn't been shown to matter yet -- especially since a low catch rate might
just mean "this domain hasn't hit that failure mode yet," not "this check
is useless." This module produces the report; it does not act on it.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..trace import EpisodicMemory, Trace

# Fixed vocabulary: the four checks that exist today. Included even if a
# freshly-created episodic memory has zero traces yet, so the report shape
# is stable regardless of how much data exists.
KNOWN_CHECKS = {"logic", "physics", "math", "confidence"}


def compute_check_value_for_traces(traces: List[Trace]) -> Dict[str, Dict[str, Any]]:
    """
    Same computation as compute_check_value_report, but over an
    already-selected list of traces rather than a whole EpisodicMemory.
    Factored out so other modules -- specifically anomaly.py, which needs
    this computed separately over a "recent" and a "baseline" slice of the
    same memory -- reuse the exact failure-counting logic rather than
    re-implementing it and risking the two drifting apart.
    """
    n_traces = len(traces)

    check_names = set(KNOWN_CHECKS)
    for t in traces:
        check_names.update(t.checks_run)
        for round_record in t.revision_history:
            check_names.update(round_record["checks_failed"])

    ever_failed_counts = {name: 0 for name in check_names}
    for t in traces:
        ever_failed = set(t.checks_failed)
        for round_record in t.revision_history:
            ever_failed.update(round_record["checks_failed"])
        for name in ever_failed:
            ever_failed_counts[name] += 1

    return {
        name: {
            "n_traces": n_traces,
            "n_ever_failed": ever_failed_counts[name],
            "catch_rate": (ever_failed_counts[name] / n_traces) if n_traces else 0.0,
        }
        for name in sorted(check_names)
    }


def traces_with_checks_run(episodic_memory: EpisodicMemory) -> List[Trace]:
    """The same filter compute_check_value_report applies before counting
    -- exposed so anomaly.py slices the identical population instead of
    re-deriving the filter itself."""
    return [t for t in episodic_memory.read_all() if t.checks_run]


def compute_check_value_report(episodic_memory: EpisodicMemory) -> Dict[str, Dict[str, Any]]:
    return compute_check_value_for_traces(traces_with_checks_run(episodic_memory))
