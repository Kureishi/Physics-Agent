"""
Curriculum Benchmark (Stage 8).

Summarizes CurriculumLog entries across many rounds: for each signal
source, how many rounds showed the targeted metric improve, regress, or
stay unchanged after a practice round. "Improve" is source-specific,
since a higher or lower number means something different depending on
what's being measured:
  - error_memory / episodic_memory: fewer (or equal) recurrences/unresolved
    cases is better.
  - knowledge_graph: higher average confidence is better.

This is a genuinely measured summary, not a guaranteed-positive one --
metric_after can be worse than metric_before, and this reports that
honestly (as "regressed") rather than only ever showing improvement.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional


def _improved(source: str, before: Optional[float], after: Optional[float]) -> Optional[bool]:
    if before is None or after is None:
        return None
    if source in ("error_memory", "episodic_memory"):
        return after <= before
    if source == "knowledge_graph":
        return after >= before
    return None


def summarize(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    by_source: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"n_rounds": 0, "n_improved": 0, "n_regressed": 0, "n_unchanged": 0, "n_unmeasurable": 0}
    )

    for entry in entries:
        source = entry["targeted_signal"]["source"]
        stats = by_source[source]
        stats["n_rounds"] += 1

        before = entry["metric_before"]
        after = entry["metric_after"]
        if before is None or after is None:
            stats["n_unmeasurable"] += 1
            continue
        if before == after:
            stats["n_unchanged"] += 1
            continue
        if _improved(source, before, after):
            stats["n_improved"] += 1
        else:
            stats["n_regressed"] += 1

    return dict(by_source)
