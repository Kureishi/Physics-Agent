"""
Tool-Selection Policy (Stage 7).

Reads episodic memory to learn, per domain tag, which tools' presence in a
problem's very first attempt correlated with not needing any correction at
all (resolution_status == "passed_initial") -- the cleanest signal
available for "this tool choice worked," since this system deliberately
never claims to know ground-truth correctness (see trace.py's note on
final_correct); what it does know is whether its own verification pipeline
was satisfied without any revision.

This is read-only analysis over completed traces, consumed by
ToolOrchestrator (via `filter_tools`) only to de-prioritize -- never to
add -- tools with a well-established poor track record for a domain, once
there's enough data to trust the number. Filtering never empties the
offered list: if the learned policy would distrust every candidate, the
original (unfiltered) list is kept instead, so a policy trained on limited
or skewed data can narrow choices but can never leave the orchestrator with
nothing to offer.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from ..trace import EpisodicMemory, Trace

MIN_USES_BEFORE_ACTING = 5
POOR_SUCCESS_RATE_THRESHOLD = 0.2


def _initial_round_tools(trace: Trace) -> List[str]:
    """
    The tools used in the very first attempt, before any correction --
    trace.tool_calls itself only reflects the current/last round (Stage 2
    overwrites it each revision), so if any revision ever happened, round 0
    is recovered from the archived snapshot instead.
    """
    if trace.revision_history:
        return sorted({tc["tool"] for tc in trace.revision_history[0]["tool_calls"]})
    return sorted({tc.tool for tc in trace.tool_calls})


class ToolSelectionPolicy:
    def __init__(self, episodic_memory: EpisodicMemory):
        self.episodic_memory = episodic_memory
        self._stats = self._compute_stats()

    def _compute_stats(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """{domain_tag: {tool_name: {"n_uses": int, "n_clean": int}}}"""
        stats: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"n_uses": 0, "n_clean": 0})
        )
        for trace in self.episodic_memory.read_all():
            if trace.resolution_status is None:
                continue  # incomplete trace (Stage 4 never ran) -- no signal
            initial_tools = _initial_round_tools(trace)
            clean = trace.resolution_status == "passed_initial"
            for domain_tag in trace.domain_tags:
                for tool_name in initial_tools:
                    entry = stats[domain_tag][tool_name]
                    entry["n_uses"] += 1
                    if clean:
                        entry["n_clean"] += 1
        return stats

    def success_rate(self, domain_tag: str, tool_name: str) -> Optional[float]:
        """None if there isn't enough data yet to trust a rate for this
        (domain_tag, tool_name) pair."""
        entry = self._stats.get(domain_tag, {}).get(tool_name)
        if entry is None or entry["n_uses"] < MIN_USES_BEFORE_ACTING:
            return None
        return entry["n_clean"] / entry["n_uses"]

    def filter_tools(self, domain_tags: List[str], available_tools: List[str]) -> List[str]:
        """
        Drops tools with a well-established poor track record across the
        given domain tags (averaged where a tool has data for more than
        one), never dropping below at least one tool. Remaining tools are
        ordered with better-performing (or not-yet-judged) tools first.
        """
        scored = []
        for tool in available_tools:
            rates = [
                r
                for r in (self.success_rate(tag, tool) for tag in domain_tags)
                if r is not None
            ]
            rate = sum(rates) / len(rates) if rates else None
            scored.append((tool, rate))

        kept = [(tool, rate) for tool, rate in scored if rate is None or rate >= POOR_SUCCESS_RATE_THRESHOLD]
        if not kept:
            # Safety: never leave the orchestrator with zero tools just
            # because the learned policy distrusts all of them.
            kept = scored

        # None (no data yet) is treated as neutral -- neither penalized nor
        # specially favored -- so it sorts as if it had the median possible
        # rate, rather than always-first or always-last.
        kept.sort(key=lambda pair: pair[1] if pair[1] is not None else 0.5, reverse=True)
        return [tool for tool, _ in kept]
