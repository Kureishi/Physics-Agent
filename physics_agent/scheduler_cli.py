"""
Scheduler CLI -- the "orchestration layer that decides when to solve,
review, and practice" from the design doc's original response. See
physics_agent/scheduler/scheduler.py for the full design note.

Separate from every other CLI in this project (cli.py solves one problem,
curriculum_cli.py runs one curriculum round, meta_report.py reviews once,
problem_set_cli.py runs a fixed batch): this one is meant to keep running,
deciding for itself which of those things to do and when.

Usage:
    # One cycle, then exit -- useful for a cron job/systemd timer that
    # itself provides the "keep running" part.
    python -m physics_agent.scheduler_cli --once

    # Keep running, one cycle every 60 seconds, until interrupted.
    python -m physics_agent.scheduler_cli --loop --interval-seconds 60

    # Bounded loop, e.g. for a supervised smoke test.
    python -m physics_agent.scheduler_cli --loop --interval-seconds 5 --max-cycles 10

    # Summarize past decisions instead of running a new cycle.
    python -m physics_agent.scheduler_cli --report

This does not daemonize itself (no fork, no pidfile, no signal handling
beyond Ctrl-C) -- run it under whatever process supervisor
(systemd/cron/nohup/tmux) the deployment already uses for long-running
processes; see scheduler.py's docstring for why that's a deliberate scope
limit, not an oversight.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

from .config import Config
from .scheduler.scheduler import DecisionLog, Scheduler


def _print_decisions(decisions) -> None:
    for d in decisions:
        print(f"[{d.action}] {d.reason}")
        if d.details:
            for key, value in d.details.items():
                print(f"    {key}: {value}")


def _print_report(config: Config) -> None:
    log = DecisionLog(config.scheduler_log_path)
    decisions = log.read_all()
    print(f"Scheduler decisions logged: {len(decisions)}\n")

    if not decisions:
        print("(scheduler hasn't run yet -- try --once first)")
        return

    counts = Counter(d.action for d in decisions)
    print("Decision breakdown:")
    for action, count in counts.most_common():
        print(f"  {action:10s} {count}")

    print("\nMost recent decisions:")
    for d in decisions[-10:]:
        print(f"  [{d.action}] {d.reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduling/Decision Loop: solve, review, practice")
    parser.add_argument("--dry-run", action="store_true", help="Use a mock LLM instead of calling LM Studio")
    parser.add_argument("--once", action="store_true", help="Run a single cycle, then exit (default)")
    parser.add_argument("--loop", action="store_true", help="Keep running cycles until interrupted or --max-cycles")
    parser.add_argument(
        "--interval-seconds", type=float, default=60.0, help="Seconds to sleep between cycles in --loop mode"
    )
    parser.add_argument(
        "--max-cycles", type=int, default=None, help="With --loop, stop after this many cycles instead of forever"
    )
    parser.add_argument(
        "--report", action="store_true", help="Summarize past scheduler decisions instead of running a cycle"
    )
    args = parser.parse_args()

    config = Config()

    if args.report:
        _print_report(config)
        return

    scheduler = Scheduler(config, dry_run=args.dry_run)

    if args.loop:
        print(
            f"Starting scheduler loop (interval={args.interval_seconds}s, "
            f"max_cycles={args.max_cycles or 'unbounded'}). Ctrl-C to stop.\n"
        )
        try:
            cycles_run = 0
            while args.max_cycles is None or cycles_run < args.max_cycles:
                decisions = scheduler.run_cycle()
                print(f"--- cycle {scheduler.state.total_cycles} ---")
                _print_decisions(decisions)
                cycles_run += 1
                if args.max_cycles is None or cycles_run < args.max_cycles:
                    time.sleep(args.interval_seconds)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    # Default: a single cycle.
    decisions = scheduler.run_cycle()
    _print_decisions(decisions)


if __name__ == "__main__":
    main()
