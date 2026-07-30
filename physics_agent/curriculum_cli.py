"""
Curriculum CLI (Stage 8).

Separate from physics_agent.cli (solves one user-submitted problem) and
physics_agent.meta_report (reviews accumulated memory without changing
anything) -- this one actively generates and solves new practice problems,
so it's the one entry point in this project that writes new episodic
traces without a person supplying the problem text.

Usage:
    # Generate + solve N practice problems targeting the current top weak areas
    python -m physics_agent.curriculum_cli --dry-run --n 2

    # Summarize past curriculum rounds instead of running a new one
    python -m physics_agent.curriculum_cli --report
"""
from __future__ import annotations

import argparse

from .config import Config
from .curriculum.benchmark import summarize
from .curriculum.curriculum_runner import CurriculumLog, CurriculumRunner


def _print_report(config: Config) -> None:
    log = CurriculumLog(config.curriculum_log_path)
    entries = log.read_all()
    print(f"Curriculum rounds logged: {len(entries)}\n")

    if not entries:
        print("(no curriculum rounds run yet)")
        return

    report = summarize(entries)
    for source, stats in sorted(report.items()):
        print(
            f"[{source}] rounds={stats['n_rounds']}  improved={stats['n_improved']}  "
            f"regressed={stats['n_regressed']}  unchanged={stats['n_unchanged']}  "
            f"unmeasurable={stats['n_unmeasurable']}"
        )


def _print_round_results(results) -> None:
    for r in results:
        print(f"\nTargeted: [{r.targeted_signal['source']}] {r.targeted_signal['reason']}")
        print(f"Generated problem: {r.generated_problem_text}")
        print(f"Target concepts: {r.target_concepts}")
        if r.literature_context:
            print(f"Literature context used: {r.literature_context}")
        print(f"Resolution: {r.resolution_status}  (confidence: {r.final_confidence})")
        print(f"Metric ({r.metric_description}):")
        print(f"  before = {r.metric_before}")
        print(f"  after  = {r.metric_after}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 8: Autonomous Curriculum")
    parser.add_argument(
        "--dry-run", action="store_true", help="Use a mock LLM instead of calling LM Studio"
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of practice problems to generate this round"
    )
    parser.add_argument(
        "--report", action="store_true", help="Summarize past curriculum rounds instead of running a new one"
    )
    args = parser.parse_args()

    config = Config()

    if args.report:
        _print_report(config)
        return

    runner = CurriculumRunner(config, dry_run=args.dry_run)
    results = runner.run_round(n_problems=args.n)

    if not results:
        print(
            "No weak-area signals available yet -- solve more problems first "
            "(or problem generation failed for every candidate signal this round)."
        )
        return

    _print_round_results(results)


if __name__ == "__main__":
    main()
