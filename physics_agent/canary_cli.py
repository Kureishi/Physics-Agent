"""
Canary CLI -- Safety Rails, "Ground-truth canary problems".

Separate from physics_agent.cli (solves one user-submitted problem),
physics_agent.meta_report (reviews accumulated memory), and
physics_agent.curriculum_cli (generates + solves new practice problems).
This one runs a small, FIXED, human-verified problem set through the same
pipeline and grades the pipeline's own checks against known-correct
answers -- see physics_agent/canary/runner.py's docstring for why that's a
meaningfully different signal from anything else in this project.

Usage:
    # Run the canary suite now (against a running LM Studio server)
    python -m physics_agent.canary_cli

    # Structural smoke test only -- offline, mock LLM, not a real grading
    # run (see the --dry-run warning printed below)
    python -m physics_agent.canary_cli --dry-run

    # Summarize the accumulated canary run history instead of running now
    python -m physics_agent.canary_cli --report

Suggested use: run this after any change to a self-eval check, a memory
consolidation rule, or a prompt -- and periodically (e.g. before each
autonomous curriculum cycle, once that scheduling layer exists) so a
regression like the MathCheck bug shows up as a canary failure instead of
silently accumulating into memory.
"""
from __future__ import annotations

import argparse
import sys

from .canary.runner import CanaryLog, CanaryRunner, latest_result_per_canary, summarize
from .config import Config

# Verdicts that mean the self-eval checks disagreed with ground truth --
# these are the two the design doc's canary bullet exists to catch, and
# the ones worth a nonzero exit code / loud report.
_CONCERNING_VERDICTS = ("correct_but_flagged", "incorrect_but_passed")


def _print_run(results, previous_by_id) -> bool:
    """Prints each canary's outcome plus a summary. Returns True if any
    concerning verdict was found this run."""
    any_concerning = False

    for r in results:
        marker = "OK" if r.verdict in ("correct_and_passed", "incorrect_and_flagged") else "!!"
        print(f"[{marker}] {r.canary_id} ({r.domain_hint}): {r.verdict}")
        print(
            f"      expected={r.expected_value} {r.units}   "
            f"matched={r.matched_value} (from {r.extraction_source}, "
            f"{r.n_candidates} candidate(s))"
        )
        if r.checks_failed:
            print(f"      checks_failed={r.checks_failed}")
        print(f"      resolution_status={r.resolution_status}  confidence={r.final_confidence}")

        if r.verdict in _CONCERNING_VERDICTS:
            any_concerning = True
            prev = previous_by_id.get(r.canary_id)
            if prev is not None and prev.verdict != r.verdict:
                print(f"      ** changed from '{prev.verdict}' on the previous run **")

    counts = summarize(results)
    print("\nSummary:")
    for verdict, count in counts.items():
        print(f"  {verdict:22s} {count}")

    if any_concerning:
        print(
            "\nAt least one canary shows the pipeline's checks disagreeing with a "
            "known-correct answer (correct_but_flagged or incorrect_but_passed). "
            "This is exactly the signal the MathCheck bug should have tripped -- "
            "worth investigating before trusting recent runs' statistics."
        )
    else:
        print("\nAll graded canaries agree with ground truth.")

    n_unmeasurable = counts.get("unmeasurable", 0)
    if n_unmeasurable:
        print(
            f"\nNote: {n_unmeasurable} canary(ies) were unmeasurable (no numeric answer "
            "could be extracted from the trace) -- a grading-pipeline gap, not a "
            "verdict on the solve itself."
        )

    return any_concerning


def _print_report(config: Config) -> None:
    log = CanaryLog(config.canary_log_path)
    entries = log.read_all()
    print(f"Canary runs logged: {len(entries)}\n")

    if not entries:
        print("(no canary runs logged yet -- run python -m physics_agent.canary_cli first)")
        return

    latest = latest_result_per_canary(entries)
    counts = summarize(list(latest.values()))

    print("Most recent verdict per canary:")
    for canary_id, r in sorted(latest.items()):
        print(f"  {canary_id:20s} {r.verdict:22s} (as of last run)")

    print("\nSummary (latest run per canary):")
    for verdict, count in counts.items():
        print(f"  {verdict:22s} {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Safety Rails: Ground-truth canary problems")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use a mock LLM instead of calling LM Studio (structural smoke test only -- "
             "the mock LLM ignores problem-specific numbers, so most canaries will read "
             "as unmeasurable/incorrect; this exercises the pipeline plumbing, not grading)",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Summarize past canary runs instead of running the suite now",
    )
    args = parser.parse_args()

    config = Config()

    if args.report:
        _print_report(config)
        return

    if args.dry_run:
        print(
            "Note: --dry-run uses a mock LLM that ignores each canary's specific "
            "numbers. This checks the pipeline runs end-to-end without errors; it is "
            "NOT a real grading run. Run without --dry-run against LM Studio for an "
            "actual canary check.\n"
        )

    log = CanaryLog(config.canary_log_path)
    previous_by_id = latest_result_per_canary(log.read_all())

    runner = CanaryRunner(config, dry_run=args.dry_run)
    results = runner.run_all()

    any_concerning = _print_run(results, previous_by_id)

    if any_concerning and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
