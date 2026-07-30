"""
Problem Set Runner (batch harness).

Not a new pipeline stage -- a convenience harness for running the Stage
1-7 solving pipeline (physics_agent.cli.run) against many problems in one
command. Two reasons this exists rather than just calling cli.run() in a
shell loop:

  1. Stage 7's policies and Stage 8's curriculum both need real
     accumulated data to do anything meaningful (their minimum-sample-size
     gates mean a handful of problems isn't enough) -- this is the fastest
     way to build that up.
  2. It prints a batch summary (resolution-status breakdown, average
     revisions, average confidence, and how often the planner's own domain
     classification agreed with this problem set's human-assigned
     `domain_hint`) that's useful for a first real sanity check against an
     actual LM Studio model.

Usage:
    python -m physics_agent.problem_set_cli
    python -m physics_agent.problem_set_cli --dry-run
    python -m physics_agent.problem_set_cli --limit 5
    python -m physics_agent.problem_set_cli path/to/other_set.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import cli
from .config import Config

DEFAULT_PROBLEM_SET_PATH = "data/problem_sets/intro_physics_set.json"


def load_problem_set(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_problem_set(
    problems: List[Dict[str, Any]],
    dry_run: bool = False,
    config: Optional[Config] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    config = config or Config()
    if limit:
        problems = problems[:limit]

    results = []
    for i, problem in enumerate(problems, 1):
        preview = problem["problem_text"][:70]
        print(f"[{i}/{len(problems)}] {problem['id']}: {preview}...")
        start = time.time()
        try:
            trace = cli.run(problem["problem_text"], dry_run=dry_run, config=config)
            elapsed = time.time() - start
            print(
                f"    -> domain_tags={trace.domain_tags}  resolution={trace.resolution_status}  "
                f"revisions={trace.revision_count}  confidence={trace.final_confidence}  ({elapsed:.1f}s)"
            )
            results.append({"id": problem["id"], "domain_hint": problem.get("domain_hint"), "trace": trace})
        except Exception as e:
            # A single problem crashing outright (e.g. LM Studio unreachable)
            # shouldn't lose the rest of the batch's results.
            print(f"    -> FAILED to solve: {e}")
            results.append({"id": problem["id"], "domain_hint": problem.get("domain_hint"), "error": str(e)})

    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    n = len(results)
    n_errors = sum(1 for r in results if "error" in r)
    solved = [r for r in results if "trace" in r]

    print("\n" + "=" * 60)
    print(f"Problem set run complete: {n} problems, {n_errors} crashed outright")
    if not solved:
        return

    status_counts = Counter(r["trace"].resolution_status for r in solved)
    print("\nResolution status breakdown:")
    for status, count in status_counts.most_common():
        print(f"  {status}: {count}")

    revisions = [r["trace"].revision_count for r in solved]
    print(f"\nAverage revisions needed: {sum(revisions) / len(revisions):.2f}")

    confidences = [r["trace"].final_confidence for r in solved if r["trace"].final_confidence is not None]
    if confidences:
        print(f"Average final confidence: {sum(confidences) / len(confidences):.2f}")

    checked = [r for r in solved if r.get("domain_hint")]
    if checked:
        matched = sum(1 for r in checked if r["domain_hint"] in r["trace"].domain_tags)
        print(f"\nPlanner's domain classification matched this set's domain_hint: {matched}/{len(checked)}")

    error_types = Counter(r["trace"].error_type for r in solved if r["trace"].error_type)
    if error_types:
        print("\nError types detected (at least one revision round triggered):")
        for error_type, count in error_types.most_common():
            print(f"  {error_type}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of physics problems through the full pipeline")
    parser.add_argument(
        "problem_set_path",
        nargs="?",
        default=DEFAULT_PROBLEM_SET_PATH,
        help=f"Path to a JSON problem set (default: {DEFAULT_PROBLEM_SET_PATH})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use a mock LLM instead of calling LM Studio")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N problems in the set")
    args = parser.parse_args()

    problems = load_problem_set(args.problem_set_path)
    results = run_problem_set(problems, dry_run=args.dry_run, limit=args.limit)
    print_summary(results)


if __name__ == "__main__":
    main()
