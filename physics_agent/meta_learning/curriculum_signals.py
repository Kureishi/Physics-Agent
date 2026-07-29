"""
Curriculum Signals (Stage 7 -> feeds a future Stage 8).

Combines the three "what's weak" sources this system already tracks into
one ranked list a later autonomous-curriculum stage could consume to
generate targeted practice problems:
  - ErrorMemory: recurring failure signatures, ranked by frequency.
  - EpisodicMemory: traces that never resolved (unresolved_max_revisions),
    grouped by domain tag.
  - KnowledgeGraph: connected clusters of low-confidence facts.

This module only ranks and summarizes; it does not generate problems --
that's explicitly out of scope here per the design doc (problem
generation is Stage 8's job, not Stage 7's).
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..knowledge_graph.graph import KnowledgeGraph
from ..memory.error_memory import ErrorMemory
from ..trace import EpisodicMemory


def weak_areas(
    episodic_memory: EpisodicMemory,
    error_memory: ErrorMemory,
    knowledge_graph: KnowledgeGraph,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []

    for entry in error_memory.most_frequent(limit=limit):
        signals.append(
            {
                "source": "error_memory",
                "domain_tags": entry["domain_tags"],
                "reason": f"error type '{entry['error_type']}' recurred {entry['frequency']}x",
                "weight": entry["frequency"],
            }
        )

    unresolved = episodic_memory.query_by_resolution_status("unresolved_max_revisions")
    domain_unresolved_counts: Dict[str, int] = {}
    for trace in unresolved:
        for tag in trace.domain_tags:
            domain_unresolved_counts[tag] = domain_unresolved_counts.get(tag, 0) + 1
    for tag, count in sorted(domain_unresolved_counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]:
        signals.append(
            {
                "source": "episodic_memory",
                "domain_tags": [tag],
                "reason": f"{count} problem(s) never resolved within the revision limit",
                "weight": count,
            }
        )

    for cluster in knowledge_graph.find_low_confidence_clusters():
        tags = set()
        for node_id in cluster:
            node = knowledge_graph.get_node(node_id)
            if node:
                tags.update(node.get("tags", []))
        signals.append(
            {
                "source": "knowledge_graph",
                "domain_tags": sorted(tags),
                "reason": f"low-confidence fact cluster: {cluster}",
                "weight": len(cluster),
            }
        )

    signals.sort(key=lambda s: s["weight"], reverse=True)
    return signals[:limit]
