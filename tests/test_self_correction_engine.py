import json

from physics_agent.llm_client import MockLLMClient
from physics_agent.orchestrator import ToolOrchestrator
from physics_agent.self_correction.engine import SelfCorrectionEngine
from physics_agent.self_eval.pipeline import SelfEvaluationPipeline
from physics_agent.trace import Trace


def _make_trace():
    trace = Trace.new(
        "A 2 kg block starts at rest at the top of a 5 m frictionless incline. Find its speed at the bottom."
    )
    trace.domain_tags = ["dynamics", "energy"]
    trace.subtasks = ["identify knowns", "apply conservation of energy", "solve for v"]
    trace.retrieved_knowledge = [
        {"statement": "Conservation of mechanical energy", "conditions": "no friction"}
    ]
    return trace


def test_engine_no_revision_needed_when_all_checks_pass():
    llm = MockLLMClient()  # everything passes by default
    orchestrator = ToolOrchestrator(llm)
    self_eval = SelfEvaluationPipeline(llm)
    engine = SelfCorrectionEngine(orchestrator, self_eval)

    trace = _make_trace()
    orchestrator.run(trace)
    self_eval.run(trace)

    engine.run(trace)

    assert trace.revision_count == 0
    assert trace.resolution_status == "passed_initial"
    assert trace.error_type is None
    assert trace.final_answer == trace.initial_solution
    assert trace.time_to_solve_ms is not None
    assert trace.revision_history == []


def test_engine_resolves_after_one_revision():
    # First self-eval run: math check fails. After one revision (which
    # re-runs orchestration with feedback), the mock's default tool
    # selection kicks back in and math check passes on round 2.
    call_count = {"n": 0}

    class FlakyMathCheck:
        name = "math"

        def run(self, trace):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"passed": False, "details": "solution does not satisfy equation"}
            return {"passed": True, "details": "verified"}

    llm = MockLLMClient()
    orchestrator = ToolOrchestrator(llm)

    from physics_agent.self_eval.logic_check import LogicCheck
    from physics_agent.self_eval.physics_check import PhysicsCheck
    from physics_agent.self_eval.confidence_check import ConfidenceCheck

    self_eval = SelfEvaluationPipeline(
        checks=[LogicCheck(llm), PhysicsCheck(llm), FlakyMathCheck(), ConfidenceCheck(llm)]
    )
    engine = SelfCorrectionEngine(orchestrator, self_eval, max_revisions=3)

    trace = _make_trace()
    orchestrator.run(trace)
    self_eval.run(trace)
    assert "math" in trace.checks_failed  # sanity check on the fixture

    engine.run(trace)

    assert trace.revision_count == 1
    assert trace.resolution_status == "resolved_after_revision"
    assert trace.error_type == "algebra_error"
    assert len(trace.revision_history) == 1
    assert trace.revision_history[0]["error_type"] == "algebra_error"
    assert trace.revision_history[0]["strategy"] == "rederive_math"
    assert trace.revision_history[0]["resolved"] is True
    assert trace.checks_failed == []


def test_engine_escalates_for_human_review_when_confidence_stays_low_after_escalation():
    # Confidence check always fails alone (no other check ever fails), so
    # every round classifies as "low_confidence_no_specific_fault" ->
    # strategy "escalate_verification". Round 0 should try that strategy;
    # round 1 landing on the same strategy again means the independent
    # check didn't help, and the engine should stop early rather than
    # burn through max_revisions repeating it.
    class AlwaysLowConfidenceCheck:
        name = "confidence"

        def run(self, trace):
            return {"passed": False, "details": "confidence below threshold"}

    llm = MockLLMClient()
    orchestrator = ToolOrchestrator(llm)

    from physics_agent.self_eval.logic_check import LogicCheck
    from physics_agent.self_eval.physics_check import PhysicsCheck
    from physics_agent.self_eval.math_check import MathCheck

    self_eval = SelfEvaluationPipeline(
        checks=[LogicCheck(llm), PhysicsCheck(llm), MathCheck(), AlwaysLowConfidenceCheck()]
    )
    engine = SelfCorrectionEngine(orchestrator, self_eval, max_revisions=5)

    trace = _make_trace()
    orchestrator.run(trace)
    self_eval.run(trace)
    assert trace.checks_failed == ["confidence"]  # sanity check on the fixture

    engine.run(trace)

    assert trace.resolution_status == "escalated_for_human_review"
    # Escalated after the first attempt didn't help -- one revision spent
    # on escalate_verification, then stop, well short of max_revisions=5.
    assert trace.revision_count == 1
    assert trace.revision_history[0]["strategy"] == "escalate_verification"


def test_engine_tries_escalate_verification_once_before_escalating():
    # A single round of low confidence, resolved by the escalation
    # itself (confidence check passes afterward) -- should NOT trigger
    # escalated_for_human_review, since the independent check helped.
    call_count = {"n": 0}

    class RecoveringConfidenceCheck:
        name = "confidence"

        def run(self, trace):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"passed": False, "details": "confidence below threshold"}
            return {"passed": True, "details": "confidence recovered"}

    llm = MockLLMClient()
    orchestrator = ToolOrchestrator(llm)

    from physics_agent.self_eval.logic_check import LogicCheck
    from physics_agent.self_eval.physics_check import PhysicsCheck
    from physics_agent.self_eval.math_check import MathCheck

    self_eval = SelfEvaluationPipeline(
        checks=[LogicCheck(llm), PhysicsCheck(llm), MathCheck(), RecoveringConfidenceCheck()]
    )
    engine = SelfCorrectionEngine(orchestrator, self_eval, max_revisions=3)

    trace = _make_trace()
    orchestrator.run(trace)
    self_eval.run(trace)

    engine.run(trace)

    assert trace.resolution_status == "resolved_after_revision"
    assert trace.revision_count == 1


def test_engine_stops_at_max_revisions_when_never_resolved():
    class AlwaysFailingMathCheck:
        name = "math"

        def run(self, trace):
            return {"passed": False, "details": "always wrong"}

    llm = MockLLMClient()
    orchestrator = ToolOrchestrator(llm)

    from physics_agent.self_eval.logic_check import LogicCheck
    from physics_agent.self_eval.physics_check import PhysicsCheck
    from physics_agent.self_eval.confidence_check import ConfidenceCheck

    self_eval = SelfEvaluationPipeline(
        checks=[LogicCheck(llm), PhysicsCheck(llm), AlwaysFailingMathCheck(), ConfidenceCheck(llm)]
    )
    engine = SelfCorrectionEngine(orchestrator, self_eval, max_revisions=2)

    trace = _make_trace()
    orchestrator.run(trace)
    self_eval.run(trace)

    engine.run(trace)

    assert trace.revision_count == 2  # hit the safety rail, stopped trying
    assert trace.resolution_status == "unresolved_max_revisions"
    assert trace.error_type == "algebra_error"
    assert len(trace.revision_history) == 2
    assert all(r["resolved"] is False for r in trace.revision_history)
    assert trace.checks_failed == ["math"]  # still failing when we gave up


def test_engine_records_revision_history_snapshot_before_overwriting():
    call_count = {"n": 0}

    class FlakyLogicCheck:
        name = "logic"

        def run(self, trace):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"passed": False, "details": "contradiction found"}
            return {"passed": True, "details": "consistent"}

    llm = MockLLMClient()
    orchestrator = ToolOrchestrator(llm)

    from physics_agent.self_eval.physics_check import PhysicsCheck
    from physics_agent.self_eval.math_check import MathCheck
    from physics_agent.self_eval.confidence_check import ConfidenceCheck

    self_eval = SelfEvaluationPipeline(
        checks=[FlakyLogicCheck(), PhysicsCheck(llm), MathCheck(), ConfidenceCheck(llm)]
    )
    engine = SelfCorrectionEngine(orchestrator, self_eval)

    trace = _make_trace()
    orchestrator.run(trace)
    self_eval.run(trace)
    original_solution = trace.initial_solution

    engine.run(trace)

    assert len(trace.revision_history) == 1
    snapshot = trace.revision_history[0]
    assert snapshot["initial_solution"] == original_solution  # pre-revision solution archived
    assert snapshot["checks_failed"] == ["logic"]
    # the "resynthesize" strategy shouldn't touch tool_calls, so trace's
    # current tool_calls should match what was archived
    assert [tc["tool"] for tc in snapshot["tool_calls"]] == [tc.tool for tc in trace.tool_calls]
