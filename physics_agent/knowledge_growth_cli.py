"""
Knowledge Growth CLI -- "Autonomous knowledge growth, not a hand-curated
15-fact seed file." See physics_agent/meta_learning/knowledge_growth.py
for the full design note.

Usage:
    # Show what WOULD be proposed, without adding anything.
    python -m physics_agent.knowledge_growth_cli --dry-run-proposals

    # Actually propose and add qualifying candidates to semantic memory.
    python -m physics_agent.knowledge_growth_cli

    # Lower the bar for a small/test dataset (default is 5 observations).
    python -m physics_agent.knowledge_growth_cli --min-observations 2

Also runs periodically as part of the scheduler's "grow" decision (see
physics_agent/scheduler/scheduler.py) -- this CLI is for running it
standalone, on demand, or inspecting what it would do.
"""
from __future__ import annotations

import argparse

from .config import Config
from .meta_learning.knowledge_growth import (
    MIN_OBSERVATIONS_TO_PROPOSE,
    ProposedFactsRegistry,
    find_candidate_facts,
    propose_and_add,
)
from .retrieval import SemanticStore
from .trace import EpisodicMemory


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Knowledge Growth")
    parser.add_argument(
        "--dry-run-proposals", action="store_true",
        help="List candidates without adding anything to semantic memory",
    )
    parser.add_argument(
        "--min-observations", type=int, default=MIN_OBSERVATIONS_TO_PROPOSE,
        help=f"Minimum distinct cleanly-resolved traces before a candidate is trusted (default: {MIN_OBSERVATIONS_TO_PROPOSE})",
    )
    args = parser.parse_args()

    config = Config()
    episodic = EpisodicMemory(config.episodic_memory_path)

    candidates = find_candidate_facts(episodic, min_observations=args.min_observations)

    if not candidates:
        print(f"No candidates found with at least {args.min_observations} observation(s).")
        return

    print(f"Found {len(candidates)} candidate(s):\n")
    for c in candidates:
        print(f"  expression:     {c['expression']}")
        if c["solve_for"]:
            print(f"  solve_for:      {c['solve_for']}")
        print(f"  domain_tags:    {c['domain_tags']}")
        print(f"  n_observations: {c['n_observations']}")
        print()

    if args.dry_run_proposals:
        print("(--dry-run-proposals: nothing added)")
        return

    semantic = SemanticStore(config.semantic_store_path)
    registry = ProposedFactsRegistry(config.proposed_facts_registry_path)
    added = propose_and_add(semantic, registry, candidates)

    if not added:
        print("All qualifying candidates were already proposed in an earlier run -- nothing new added.")
        return

    print(f"Added {len(added)} new self-derived fact(s) to semantic memory:")
    for entry in added:
        print(f"  [{entry['entry_id']}] {entry['statement']} (confidence starts low; refined via future use)")


if __name__ == "__main__":
    main()
