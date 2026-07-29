from physics_agent.llm_client import MockLLMClient
from physics_agent.self_eval.confidence_check import ConfidenceCheck
from physics_agent.trace import Trace


def _make_trace():
    trace = Trace.new("A ball is dropped from 10 m. Find its speed at impact.")
    trace.initial_solution = "v = sqrt(2*g*h) = 14.0 m/s"
    trace.checks_run = ["logic", "physics", "math"]
    trace.checks_failed = []
    return trace


def test_confidence_check_default_passes_and_sets_trace_field():
    llm = MockLLMClient()
    check = ConfidenceCheck(llm, threshold=0.6)
    trace = _make_trace()
    result = check.run(trace)
    assert result["passed"] is True
    assert trace.final_confidence == 0.85  # MockLLMClient's default


def test_confidence_check_fails_below_threshold():
    canned = {"confidence-check component": '{"confidence": 0.3, "rationale": "physics check failed"}'}
    llm = MockLLMClient(canned_responses=canned)
    check = ConfidenceCheck(llm, threshold=0.6)
    trace = _make_trace()
    result = check.run(trace)
    assert result["passed"] is False
    assert trace.final_confidence == 0.3


def test_confidence_check_clamps_out_of_range_values():
    canned = {"confidence-check component": '{"confidence": 1.5, "rationale": "overconfident model"}'}
    llm = MockLLMClient(canned_responses=canned)
    check = ConfidenceCheck(llm)
    trace = _make_trace()
    check.run(trace)
    assert trace.final_confidence == 1.0


def test_confidence_check_handles_unparseable_response():
    canned = {
        "unfixable confidence": "not json",
        "valid JSON": "still not json",
    }
    llm = MockLLMClient(canned_responses=canned)
    check = ConfidenceCheck(llm, max_retries=1)
    trace = _make_trace()
    trace.problem_text = "unfixable confidence problem"
    result = check.run(trace)  # should not raise
    assert result["passed"] is False
    assert trace.final_confidence == 0.0


def test_confidence_check_threshold_policy_raises_effective_threshold():
    class StubThresholdPolicy:
        def recommended_confidence_threshold(self, domain_tags, default_threshold):
            return 0.95  # simulate an overconfidence-triggered raise

    llm = MockLLMClient()  # default confidence is 0.85
    check = ConfidenceCheck(llm, threshold=0.6, threshold_policy=StubThresholdPolicy())
    trace = _make_trace()

    result = check.run(trace)

    # 0.85 clears the static 0.6 threshold but not the policy-raised 0.95
    assert result["passed"] is False
    assert "raised" in result["details"]


def test_confidence_check_without_threshold_policy_uses_static_threshold():
    llm = MockLLMClient()  # default confidence is 0.85
    check = ConfidenceCheck(llm, threshold=0.6)  # no threshold_policy -- pre-Stage-7 behavior
    trace = _make_trace()

    result = check.run(trace)

    assert result["passed"] is True
    assert "raised" not in result["details"]
