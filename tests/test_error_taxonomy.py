from physics_agent.self_correction.error_taxonomy import classify_error
from physics_agent.trace import Trace


def _trace_with(checks_failed, check_details):
    trace = Trace.new("test problem")
    trace.checks_failed = checks_failed
    trace.check_details = check_details
    return trace


def test_math_failure_classified_as_algebra_error():
    trace = _trace_with(
        ["math"],
        [{"check": "math", "passed": False, "details": "solution does not satisfy equation"}],
    )
    error_type, strategy, rationale = classify_error(trace)
    assert error_type == "algebra_error"
    assert strategy == "rederive_math"
    assert "does not satisfy" in rationale


def test_physics_disagreement_classified_as_cross_method_disagreement():
    trace = _trace_with(
        ["physics"],
        [{"check": "physics", "passed": False, "details": "symbolic_math result 9.9 disagrees with simulation's v=3.0"}],
    )
    error_type, strategy, rationale = classify_error(trace)
    assert error_type == "cross_method_disagreement"
    assert strategy == "rederive_physics_setup"


def test_physics_llm_failure_without_disagreement_text_classified_as_conceptual():
    trace = _trace_with(
        ["physics"],
        [{"check": "physics", "passed": False, "details": "formula used outside its validity range"}],
    )
    error_type, strategy, rationale = classify_error(trace)
    assert error_type == "physics_conceptual_error"
    assert strategy == "rederive_physics_setup"


def test_logic_failure_classified_as_reasoning_inconsistency():
    trace = _trace_with(
        ["logic"],
        [{"check": "logic", "passed": False, "details": "final answer contradicts an earlier step"}],
    )
    error_type, strategy, rationale = classify_error(trace)
    assert error_type == "reasoning_inconsistency"
    assert strategy == "resynthesize"


def test_confidence_only_failure_classified_as_escalate():
    trace = _trace_with(
        ["confidence"],
        [{"check": "confidence", "passed": False, "details": "confidence=0.4"}],
    )
    error_type, strategy, rationale = classify_error(trace)
    assert error_type == "low_confidence_no_specific_fault"
    assert strategy == "escalate_verification"


def test_math_takes_priority_over_physics_and_logic():
    trace = _trace_with(
        ["math", "physics", "logic"],
        [
            {"check": "math", "passed": False, "details": "bad algebra"},
            {"check": "physics", "passed": False, "details": "bad physics"},
            {"check": "logic", "passed": False, "details": "bad logic"},
        ],
    )
    error_type, strategy, rationale = classify_error(trace)
    assert error_type == "algebra_error"  # math wins priority


def test_physics_takes_priority_over_logic():
    trace = _trace_with(
        ["physics", "logic"],
        [
            {"check": "physics", "passed": False, "details": "bad physics"},
            {"check": "logic", "passed": False, "details": "bad logic"},
        ],
    )
    error_type, strategy, rationale = classify_error(trace)
    assert error_type == "physics_conceptual_error"


def test_unrecognized_combination_falls_back_to_multiple_faults():
    trace = _trace_with(
        ["some_future_check"],
        [{"check": "some_future_check", "passed": False, "details": "unknown failure"}],
    )
    error_type, strategy, rationale = classify_error(trace)
    assert error_type == "multiple_faults"
    assert strategy == "full_replan"
