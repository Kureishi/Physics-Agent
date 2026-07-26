"""
Stage 1 CLI: run a physics problem through the Task Planner + Retrieval,
and write the resulting (partial) trace to episodic memory.

Usage:
    # Against a running LM Studio server (defaults to http://localhost:1234/v1)
    python -m physics_agent.cli "A 2 kg block slides down a frictionless 30 degree incline..."

    # Offline, no LM Studio needed (uses a mock LLM):
    python -m physics_agent.cli --dry-run "A 2 kg block slides down..."
"""
from __future__ import annotations

import argparse

from .config import Config
from .llm_client import LLMClient, MockLLMClient
from .orchestrator import ToolOrchestrator
from .planner import TaskPlanner
from .retrieval import SemanticStore
from .trace import Trace, EpisodicMemory


def run(problem_text: str, dry_run: bool = False, config: Config = None) -> Trace:
    config = config or Config()

    llm = (
        MockLLMClient()
        if dry_run
        else LLMClient(
            base_url=config.lm_studio_base_url,
            api_key=config.lm_studio_api_key,
            model=config.lm_studio_model,
        )
    )

    planner = TaskPlanner(llm)
    orchestrator = ToolOrchestrator(llm)
    store = SemanticStore(config.semantic_store_path)
    memory = EpisodicMemory(config.episodic_memory_path)

    trace = Trace.new(problem_text)

    # Stage 1: plan + retrieve
    plan = planner.decompose(problem_text)
    trace.domain_tags = plan["domain_tags"]
    trace.subtasks = plan["subtasks"]
    trace.planner_raw_response = plan["raw_response"]
    trace.planning_time_ms = plan["planning_time_ms"]

    trace.retrieved_knowledge = store.retrieve(problem_text, domain_tags=trace.domain_tags, k=3)

    # Stage 2: select + execute tools, synthesize an initial solution
    orchestrator.run(trace)

    memory.write(trace)
    return trace


def _print_trace(trace: Trace) -> None:
    print(f"\nProblem ID: {trace.problem_id}")
    print(f"Domain tags: {trace.domain_tags}")

    print("\nSubtasks:")
    for i, s in enumerate(trace.subtasks, 1):
        print(f"  {i}. {s}")

    print("\nRetrieved knowledge:")
    if not trace.retrieved_knowledge:
        print("  (none matched)")
    for k in trace.retrieved_knowledge:
        print(f"  - [{k['id']}] {k['statement']}")
        print(f"        conditions: {k['conditions']}  |  confidence: {k['confidence']}")

    print(f"\nPlanning time: {trace.planning_time_ms:.1f} ms")

    print("\nTool calls:")
    if not trace.tool_calls:
        print("  (none made)")
    for tc in trace.tool_calls:
        print(f"  - {tc.tool}  ({tc.latency_ms:.1f} ms)")
        print(f"      input:  {tc.input}")
        print(f"      output: {tc.output}")

    print("\nInitial solution:")
    print(f"  {trace.initial_solution}")

    print(f"\nOrchestration time: {trace.orchestration_time_ms:.1f} ms")
    print("Trace written to episodic memory.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Task Planner + Retrieval")
    parser.add_argument("problem", help="Physics problem text")
    parser.add_argument(
        "--dry-run", action="store_true", help="Use a mock LLM instead of calling LM Studio"
    )
    parser.add_argument("--memory-path", default=None, help="Override episodic memory path")
    args = parser.parse_args()

    config = Config()
    if args.memory_path:
        config.episodic_memory_path = args.memory_path

    trace = run(args.problem, dry_run=args.dry_run, config=config)
    _print_trace(trace)


if __name__ == "__main__":
    main()
