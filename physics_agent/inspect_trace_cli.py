"""
Trace Inspector CLI.

Not a new pipeline stage -- a debugging/inspection tool for looking at one
solved problem's full trace in detail: what was retrieved, what tools were
called, every self-evaluation check's verdict, and (if any revisions
happened) each round's rationale and outcome. physics_agent.meta_report
aggregates across ALL traces; this looks at exactly ONE, which is what you
actually want when a problem_set_cli run flags something like an
unresolved trace and you need to see *why*.

Usage:
    python -m physics_agent.inspect_trace_cli "electron"      # search by substring in problem_text
    python -m physics_agent.inspect_trace_cli --id <problem_id>  # exact match by problem_id
    python -m physics_agent.inspect_trace_cli --list             # list all traces (id + short preview)
"""
from __future__ import annotations

import argparse
from typing import List, Optional

from .config import Config
from .trace import EpisodicMemory, Trace


def find_matches(episodic: EpisodicMemory, query: Optional[str], problem_id: Optional[str]) -> List[Trace]:
    traces = episodic.read_all()
    if problem_id:
        return [t for t in traces if t.problem_id == problem_id]
    if query:
        query_lower = query.lower()
        return [t for t in traces if query_lower in t.problem_text.lower()]
    return traces


def print_short(trace: Trace) -> None:
    preview = trace.problem_text[:70]
    print(f"  {trace.problem_id}  [{trace.source}]  {preview}...")


def print_full(trace: Trace) -> None:
    print("=" * 70)
    print(f"Problem ID: {trace.problem_id}")
    print(f"Source: {trace.source}")
    if trace.curriculum_target:
        print(f"Curriculum target: {trace.curriculum_target}")

    print(f"\nProblem text:\n  {trace.problem_text}")
    print(f"\nDomain tags: {trace.domain_tags}")
    print(f"Subtasks: {trace.subtasks}")

    print("\nRetrieved knowledge:")
    if not trace.retrieved_knowledge:
        print("  (none)")
    for k in trace.retrieved_knowledge:
        print(
            f"  - [{k.get('id')}] {k.get('statement')}  "
            f"(conditions: {k.get('conditions')}, confidence: {k.get('confidence')})"
        )
    print(f"\nPlanning time: {trace.planning_time_ms} ms")

    print("\nTool calls (current/final round only -- see revision history below for earlier rounds):")
    if not trace.tool_calls:
        print("  (none)")
    for tc in trace.tool_calls:
        print(f"  - {tc.tool}  ({tc.latency_ms:.1f} ms)")
        print(f"      input:  {tc.input}")
        print(f"      output: {tc.output}")
    print(f"\nOrchestration time: {trace.orchestration_time_ms} ms")

    print(f"\nInitial/current solution:\n  {trace.initial_solution}")

    print("\nSelf-evaluation (final candidate):")
    if not trace.check_details:
        print("  (not yet self-evaluated)")
    for detail in trace.check_details:
        status = "PASS" if detail["passed"] else "FAIL"
        print(f"  [{status}] {detail['check']}: {detail['details']}")
    print(f"\nChecks failed (final): {trace.checks_failed}")
    print(f"Final confidence: {trace.final_confidence}")

    print(f"\nRevision count: {trace.revision_count}")
    print(f"Resolution status: {trace.resolution_status}")
    print(f"Last detected error type: {trace.error_type}")

    print("\nRevision history:")
    if not trace.revision_history:
        print("  (no revisions needed)")
    for r in trace.revision_history:
        print(f"\n  Round {r['round']}:")
        print(f"    error_type: {r['error_type']}")
        print(f"    strategy applied: {r['strategy']}")
        print(f"    rationale: {r['rationale']}")
        print(f"    checks_failed going into this round: {r['checks_failed']}")
        for d in r["check_details"]:
            if not d["passed"]:
                print(f"      - {d['check']}: {d['details']}")
        print(f"    resolved by this round's fix: {r['resolved']}")
        print(f"    solution before this round's fix:\n      {r['initial_solution']}")

    print(f"\nFinal answer:\n  {trace.final_answer}")
    print(f"Total time to solve: {trace.time_to_solve_ms} ms")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a single solved problem's full trace")
    parser.add_argument("query", nargs="?", default=None, help="Substring to search for in problem_text")
    parser.add_argument("--id", dest="problem_id", default=None, help="Exact problem_id to look up")
    parser.add_argument("--list", action="store_true", help="List all traces instead of searching")
    args = parser.parse_args()

    config = Config()
    episodic = EpisodicMemory(config.episodic_memory_path)

    if args.list:
        traces = episodic.read_all()
        print(f"{len(traces)} trace(s) in episodic memory:\n")
        for t in traces:
            print_short(t)
        return

    if not args.query and not args.problem_id:
        parser.error("Provide a search query, --id, or --list")

    matches = find_matches(episodic, args.query, args.problem_id)

    if not matches:
        print("No matching traces found.")
        return

    if len(matches) > 1:
        print(f"{len(matches)} traces matched -- narrow with --id:\n")
        for t in matches:
            print_short(t)
        return

    print_full(matches[0])


if __name__ == "__main__":
    main()
