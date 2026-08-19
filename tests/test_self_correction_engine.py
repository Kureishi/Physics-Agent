import json
import tempfile
from pathlib import Path

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


def test_engine_applies_strategy_override_instead_of_taxonomy_default():
    class AlwaysFailingMathCheck:
        name = "math"

        def run(self, trace):
            return {"passed": False, "details": "always wrong"}

    llm = MockLLMClient()
    orchestrator = ToolOrchestrator(llm)

    from physics_agent.self_eval.logic_check import LogicCheck
    from physics_agent.self_eval.physics_check import PhysicsCheck
    from physics_agent.self_eval.confidence_check import ConfidenceCheck
    from physics_agent.memory.procedural import ProceduralMemory
    from physics_agent.meta_learning.strategy_override import StrategyOverridePolicy

    with tempfile.TemporaryDirectory() as d:
        procedural = ProceduralMemory(Path(d) / "procedural.json")
        # error_taxonomy's fixed default for a math-only failure is
        # "algebra_error" -> "rederive_math". Seed procedural memory so
        # "resynthesize" has a strong, well-sampled track record for this
        # exact (domain, error_type) pair -- enough to clear
        # StrategyOverridePolicy's bars for an untested default.
        for resolved in [True, True, True, True, False]:  # 80%
            procedural.record_outcome(
                ["dynamics", "energy"], "algebra_error", "resynthesize", resolved=resolved
            )
        policy = StrategyOverridePolicy(procedural)

        self_eval = SelfEvaluationPipeline(
            checks=[LogicCheck(llm), PhysicsCheck(llm), AlwaysFailingMathCheck(), ConfidenceCheck(llm)]
        )
        engine = SelfCorrectionEngine(
            orchestrator, self_eval, max_revisions=1, strategy_override_policy=policy
        )

        trace = _make_trace()
        orchestrator.run(trace)
        self_eval.run(trace)

        engine.run(trace)

    assert trace.error_type == "algebra_error"  # taxonomy classification unchanged
    assert trace.revision_history[0]["strategy"] == "resynthesize"  # but the applied strategy is overridden
    assert "procedural memory" in trace.revision_history[0]["rationale"]


def test_engine_without_override_policy_uses_taxonomy_default():
    # Same seeded procedural data as above, but no policy passed to the
    # engine -- confirms the override is opt-in, not automatic just
    # because procedural memory happens to have data.
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
    engine = SelfCorrectionEngine(orchestrator, self_eval, max_revisions=1)

    trace = _make_trace()
    orchestrator.run(trace)
    self_eval.run(trace)

    engine.run(trace)

    assert trace.revision_history[0]["strategy"] == "rederive_math"


def test_engine_escalates_when_the_same_strategy_never_resolves_anything():
    # Generalized escalation (see engine.py's docstring): a strategy that
    # keeps getting reapplied to a check that never once stops failing
    # after it should escalate for human review well before max_revisions
    # is exhausted -- rather than mechanically repeating an action already
    # shown, in this trace, not to work. This is the exact shape a real
    # accumulated run surfaced: `resynthesize` sitting at a flat 0% success
    # rate across many uses in several domains.
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
    engine = SelfCorrectionEngine(orchestrator, self_eval, max_revisions=3)

    trace = _make_trace()
    orchestrator.run(trace)
    self_eval.run(trace)

    engine.run(trace)

    # Escalates after the strategy's first attempt fails to resolve
    # anything, well short of the 3-revision cap -- not "stopped and
    # marked unresolved," but "stopped and flagged for a person."
    assert trace.resolution_status == "escalated_for_human_review"
    assert trace.revision_count == 1
    assert trace.error_type == "algebra_error"
    assert len(trace.revision_history) == 1
    assert trace.revision_history[0]["strategy"] == "rederive_math"
    assert trace.revision_history[0]["resolved"] is False


def test_engine_still_reaches_max_revisions_when_no_single_strategy_repeats_unsuccessfully():
    # The safety rail (max_revisions) is still reachable: if fixing one
    # check's failure coincides with a different check flaring up each
    # round, no single strategy ever accumulates two straight failed uses,
    # so escalation correctly stays out of the way and the revision cap
    # is what eventually stops the loop.
    class TogglingCheck:
        def __init__(self, name, fail_on_calls):
            self.name = name
            self._fail_on_calls = set(fail_on_calls)
            self.n_calls = 0

        def run(self, trace):
            self.n_calls += 1
            if self.n_calls in self._fail_on_calls:
                return {"passed": False, "details": f"{self.name} fails on call {self.n_calls}"}
            return {"passed": True, "details": "ok"}

    # call 1 (before engine.run): math fails, physics passes
    # call 2 (after round 0's rederive_math): math passes, physics fails
    # call 3 (after round 1's rederive_physics_setup): math fails again,
    #   physics passes -- math's ONE prior use (round 0) was resolved
    #   (still_failing came up empty against call 2's result), so
    #   reapplying rederive_math here does not count as "previously
    #   ineffective" and escalation correctly does not fire.
    math_check = TogglingCheck("math", fail_on_calls={1, 3})
    physics_check = TogglingCheck("physics", fail_on_calls={2})

    llm = MockLLMClient()
    orchestrator = ToolOrchestrator(llm)

    from physics_agent.self_eval.logic_check import LogicCheck
    from physics_agent.self_eval.confidence_check import ConfidenceCheck

    self_eval = SelfEvaluationPipeline(
        checks=[LogicCheck(llm), physics_check, math_check, ConfidenceCheck(llm)]
    )
    engine = SelfCorrectionEngine(orchestrator, self_eval, max_revisions=2)

    trace = _make_trace()
    orchestrator.run(trace)
    self_eval.run(trace)  # call 1

    engine.run(trace)

    assert trace.resolution_status == "unresolved_max_revisions"
    assert trace.revision_count == 2
    assert len(trace.revision_history) == 2
    assert trace.revision_history[0]["strategy"] == "rederive_math"
    assert trace.revision_history[0]["resolved"] is True
    assert trace.revision_history[1]["strategy"] == "rederive_physics_setup"
    assert trace.revision_history[1]["resolved"] is True
    assert trace.checks_failed == ["math"]  # still failing (call 3) when we gave up


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
