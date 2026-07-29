"""
Physics agent CLI: runs the full Stage 1-7 pipeline (plan + retrieve, tool
orchestration, self-evaluation, self-correction, memory consolidation,
meta-learning policies) on a single problem.

Usage:
    # Against a running LM Studio server (defaults to http://localhost:1234/v1)
    python -m physics_agent.cli "A 2 kg block slides down a frictionless 30 degree incline..."

    # Offline, no LM Studio needed (uses a mock LLM):
    python -m physics_agent.cli --dry-run "A 2 kg block slides down..."

See physics_agent/meta_report.py for a separate entry point that reviews
accumulated memory (check-value report, declining strategies, weak areas)
rather than solving a single problem -- consistent with meta-learning being
an "outer loop" over many solves, not a per-solve step.
"""
from __future__ import annotations

import argparse

from .config import Config
from .knowledge_graph.graph import KnowledgeGraph
from .llm_client import LLMClient, MockLLMClient
from .memory.consolidator import MemoryConsolidator
from .memory.error_memory import ErrorMemory
from .memory.procedural import ProceduralMemory
from .meta_learning.tool_policy import ToolSelectionPolicy
from .meta_learning.verification_depth import VerificationDepthPolicy
from .orchestrator import ToolOrchestrator
from .planner import TaskPlanner
from .retrieval import SemanticStore
from .self_correction.engine import SelfCorrectionEngine
from .self_eval.pipeline import SelfEvaluationPipeline
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

    store = SemanticStore(config.semantic_store_path)
    knowledge_graph = KnowledgeGraph(config.knowledge_graph_path, store)

    episodic = EpisodicMemory(config.episodic_memory_path)
    procedural = ProceduralMemory(config.procedural_memory_path)
    error_memory = ErrorMemory(config.error_memory_path)
    consolidator = MemoryConsolidator(episodic, store, procedural, error_memory)

    # Stage 7: policies computed fresh from accumulated episodic memory each
    # run. At this project's scale that's cheap enough to just do live; a
    # system solving at high volume would more likely cache these and
    # recompute on a schedule rather than every single solve, but the
    # policies themselves don't change based on which of those you pick.
    tool_policy = ToolSelectionPolicy(episodic)
    verification_depth_policy = VerificationDepthPolicy(episodic)

    planner = TaskPlanner(llm)
    orchestrator = ToolOrchestrator(llm, tool_policy=tool_policy)
    self_eval = SelfEvaluationPipeline(
        llm, knowledge_graph=knowledge_graph, verification_depth_policy=verification_depth_policy
    )
    self_correction = SelfCorrectionEngine(orchestrator, self_eval, max_revisions=config.max_revisions)

    trace = Trace.new(problem_text)

    # Stage 1: plan + retrieve
    plan = planner.decompose(problem_text)
    trace.domain_tags = plan["domain_tags"]
    trace.subtasks = plan["subtasks"]
    trace.planner_raw_response = plan["raw_response"]
    trace.planning_time_ms = plan["planning_time_ms"]

    trace.retrieved_knowledge = store.retrieve(problem_text, domain_tags=trace.domain_tags, k=3)

    # Stage 2: select + execute tools (Stage 7's tool_policy may narrow the
    # offered set), synthesize an initial solution
    orchestrator.run(trace)

    # Stage 3: self-evaluate the initial solution (Stage 7's
    # verification_depth_policy may raise ConfidenceCheck's threshold)
    self_eval.run(trace)

    # Stage 4: detect + correct, looping back through Stage 2/3 as needed
    self_correction.run(trace)

    # Stage 5: consolidate this solve into episodic/semantic/procedural/error memory
    consolidator.consolidate(trace)

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

    print("\nSelf-evaluation (final candidate):")
    for detail in trace.check_details:
        status = "PASS" if detail["passed"] else "FAIL"
        print(f"  [{status}] {detail['check']}: {detail['details']}")

    if trace.checks_failed:
        print(f"\n  {len(trace.checks_failed)} check(s) still failing: {trace.checks_failed}")
    else:
        print("\n  All checks passed.")
    print(f"  Confidence: {trace.final_confidence}")

    print(f"\nRevisions made: {trace.revision_count}")
    print(f"Resolution status: {trace.resolution_status}")
    if trace.error_type:
        print(f"Last detected error type: {trace.error_type}")
    print(f"\nFinal answer:\n  {trace.final_answer}")
    print(f"\nTotal time to solve: {trace.time_to_solve_ms:.1f} ms")

    print("\nMemory consolidated: episodic trace, semantic confidence updates, "
          "procedural + error memory (if any revisions occurred).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Task Planner + Retrieval")
    parser.add_argument("problem", help="Physics problem text")
    parser.add_argument(
        "--dry-run", action="store_true", help="Use a mock LLM instead of calling LM Studio"
    )
    parser.add_argument(
        "--memory-path", default=None, help="Override episodic memory path (procedural/error/semantic paths are set via Config or env vars)"
    )
    args = parser.parse_args()

    config = Config()
    if args.memory_path:
        config.episodic_memory_path = args.memory_path

    trace = run(args.problem, dry_run=args.dry_run, config=config)
    _print_trace(trace)


if __name__ == "__main__":
    main()
