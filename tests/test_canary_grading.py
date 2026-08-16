import json

from physics_agent.canary.grading import (
    best_match,
    extract_candidate_values,
    extract_numbers,
    grade_trace,
)
from physics_agent.trace import Trace, ToolCall


def test_extract_numbers_plain_decimals():
    assert extract_numbers("v = 9.8995 m/s") == [9.8995]


def test_extract_numbers_e_notation():
    assert extract_numbers("E = 3.313e-19 J") == [3.313e-19]


def test_extract_numbers_unicode_scientific_notation():
    numbers = extract_numbers("The result is 5.466×10⁻¹⁴ J.")
    assert len(numbers) == 1
    assert abs(numbers[0] - 5.466e-14) < 1e-20


def test_extract_numbers_ascii_x10_caret_notation():
    numbers = extract_numbers("about 2.5 x 10^3 Pa")
    assert len(numbers) == 1
    assert abs(numbers[0] - 2500.0) < 1e-9


def test_extract_numbers_multiple_candidates_in_prose():
    numbers = extract_numbers("m1 = 3 kg, v1 = 4 m/s, final speed = 3.0 m/s")
    assert numbers == [3.0, 4.0, 3.0]


def test_extract_numbers_empty_text():
    assert extract_numbers("") == []
    assert extract_numbers(None) == []


def test_best_match_within_tolerance():
    matched, value = best_match([1.0, 9.87, 20.0], 9.8995, 0.02)
    assert matched is True
    assert value == 9.87


def test_best_match_outside_tolerance():
    matched, value = best_match([1.0, 20.0], 9.8995, 0.02)
    assert matched is False
    assert value is None


def test_best_match_no_candidates():
    matched, value = best_match([], 9.8995, 0.02)
    assert matched is False
    assert value is None


def test_best_match_prefers_closest_of_multiple_matches():
    # Both within tolerance of 100; closer one should win.
    matched, value = best_match([98.5, 100.5], 100.0, 0.02)
    assert matched is True
    assert value == 100.5


def test_best_match_near_zero_expected_uses_absolute_tolerance():
    matched, value = best_match([1e-10, 5.0], 0.0, 0.02)
    assert matched is True
    assert value == 1e-10


def test_extract_candidate_values_prefers_tool_output_over_prose():
    trace = Trace.new("test problem")
    trace.tool_calls = [
        ToolCall(
            tool="symbolic_math",
            input="{}",
            output=json.dumps(
                {
                    "expression": "Eq(v, 9.8995)",
                    "solve_for": "v",
                    "substitutions": {},
                    "solutions": ["9.8995"],
                    "solutions_numeric": [9.8995],
                }
            ),
            latency_ms=10.0,
        )
    ]
    # Deliberately wrong/misleading prose -- tool output should win.
    trace.final_answer = "The final answer is approximately 3 kg."

    candidates, source = extract_candidate_values(trace)
    assert source == "tool_output"
    assert candidates == [9.8995]


def test_extract_candidate_values_falls_back_to_prose_with_no_tool_calls():
    trace = Trace.new("test problem")
    trace.tool_calls = []
    trace.final_answer = "The result is v = 9.8995 m/s."

    candidates, source = extract_candidate_values(trace)
    assert source == "prose_fallback"
    assert candidates == [9.8995]


def test_extract_candidate_values_skips_errored_tool_calls():
    trace = Trace.new("test problem")
    trace.tool_calls = [
        ToolCall(tool="symbolic_math", input="{}", output=json.dumps({"error": "bad input"}), latency_ms=5.0),
    ]
    trace.final_answer = "v = 9.8995 m/s"

    candidates, source = extract_candidate_values(trace)
    assert source == "prose_fallback"
    assert candidates == [9.8995]


def test_extract_candidate_values_simulation_tool_uses_final_state():
    trace = Trace.new("test problem")
    trace.tool_calls = [
        ToolCall(
            tool="simulation",
            input="{}",
            output=json.dumps(
                {
                    "state_vars": ["x", "v"],
                    "final_time": 2.0,
                    "final_state": {"x": 19.6, "v": 19.8995},
                    "t_values": [0, 2],
                    "trajectory": {"x": [0, 19.6], "v": [0, 19.8995]},
                }
            ),
            latency_ms=5.0,
        )
    ]

    candidates, source = extract_candidate_values(trace)
    assert source == "tool_output"
    assert sorted(candidates) == sorted([19.6, 19.8995])


def test_grade_trace_correct_answer():
    trace = Trace.new("test problem")
    trace.final_answer = "v = 9.8995 m/s"
    result = grade_trace(trace, expected_value=9.8995, relative_tolerance=0.02)
    assert result.answer_correct is True
    assert result.extraction_source == "prose_fallback"
    assert result.n_candidates == 1


def test_grade_trace_incorrect_answer():
    trace = Trace.new("test problem")
    trace.final_answer = "v = 500 m/s"
    result = grade_trace(trace, expected_value=9.8995, relative_tolerance=0.02)
    assert result.answer_correct is False


def test_grade_trace_no_extractable_number():
    trace = Trace.new("test problem")
    trace.final_answer = "The block accelerates and eventually stops."
    result = grade_trace(trace, expected_value=9.8995, relative_tolerance=0.02)
    assert result.answer_correct is False
    assert result.n_candidates == 0
