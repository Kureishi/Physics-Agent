"""
Meta-learning report CLI (Stage 7).

Prints a review of accumulated memory: check-value rates, procedural
strategies flagged for declining success, and ranked weak areas. This is
deliberately separate from physics_agent.cli (which solves one problem) --
meta-learning is an outer loop over *many* solves, not a step in solving
any one of them.

Usage:
    python -m physics_agent.meta_report
"""
from __future__ import annotations

import json

from .config import Config
from .knowledge_graph.graph import KnowledgeGraph
from .memory.error_memory import ErrorMemory
from .memory.procedural import ProceduralMemory
from .meta_learning.report import build_report
from .retrieval import SemanticStore
from .trace import EpisodicMemory


def main() -> None:
    config = Config()
    store = SemanticStore(config.semantic_store_path)
    knowledge_graph = KnowledgeGraph(config.knowledge_graph_path, store)
    episodic = EpisodicMemory(config.episodic_memory_path)
    procedural = ProceduralMemory(config.procedural_memory_path)
    error_memory = ErrorMemory(config.error_memory_path)

    report = build_report(episodic, procedural, error_memory, knowledge_graph)

    print(f"Traces reviewed: {report['n_traces']}\n")

    print("Check value (catch rate = fraction of traces where this check ever failed):")
    for check_name, stats in sorted(report["check_value"].items()):
        print(f"  {check_name:10s}  n_traces={stats['n_traces']:4d}  catch_rate={stats['catch_rate']:.2%}")

    print("\nProcedural strategies flagged for declining success rate:")
    if not report["declining_strategies"]:
        print("  (none)")
    for entry in report["declining_strategies"]:
        print(
            f"  {entry['id']}: success_rate={entry['success_rate']:.2%} "
            f"over {entry['n_uses']} uses"
        )

    print("\nWeak areas (ranked, for a future curriculum stage to target):")
    if not report["weak_areas"]:
        print("  (none identified yet)")
    for signal in report["weak_areas"]:
        print(f"  [{signal['source']}] {signal['reason']} (domains: {signal['domain_tags']})")


if __name__ == "__main__":
    main()
