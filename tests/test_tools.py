import math

import pytest

from physics_agent.tools.literature import LiteratureSearchTool
from physics_agent.tools.simulation import SimulationTool
from physics_agent.tools.symbolic_math import SymbolicMathTool


# -- SymbolicMathTool ---------------------------------------------------------


def test_symbolic_math_solves_energy_conservation():
    tool = SymbolicMathTool()
    result = tool.run(
        {
            "expression": "Eq(m*g*h, 0.5*m*v**2)",
            "solve_for": "v",
            "substitutions": {"m": 2, "g": 9.8, "h": 5},
        }
    )
    numeric_solutions = [s for s in result["solutions_numeric"] if s is not None]
    expected = math.sqrt(2 * 9.8 * 5)
    assert any(abs(abs(s) - expected) < 1e-6 for s in numeric_solutions)


def test_symbolic_math_missing_fields_raises():
    tool = SymbolicMathTool()
    with pytest.raises(ValueError):
        tool.run({"expression": "Eq(x, 1)"})  # missing solve_for


def test_symbolic_math_unparseable_expression_raises():
    tool = SymbolicMathTool()
    with pytest.raises(ValueError):
        tool.run({"expression": "this is not : valid sympy (((", "solve_for": "x"})


def test_symbolic_math_no_solution_raises():
    tool = SymbolicMathTool()
    # 1 = 2 is never true -- no solution for x
    with pytest.raises(ValueError):
        tool.run({"expression": "Eq(1, 2)", "solve_for": "x"})


def test_symbolic_math_evaluates_bare_expression_when_solve_for_absent():
    # Reproduces a real failure observed running against an actual local
    # model: it computed relativistic KE by plugging every value directly
    # into the expression, then labeled the target quantity 'solve_for'
    # even though it no longer appears anywhere as a free symbol. This
    # used to raise "SymPy found no solutions for KE"; it should now just
    # evaluate the arithmetic.
    tool = SymbolicMathTool()
    result = tool.run(
        {
            "expression": "(1/sqrt(1 - (0.8)**2) - 1) * 9.11e-31 * (3e8)**2",
            "solve_for": "KE",
            "substitutions": {},
        }
    )
    assert "note" in result
    assert result["solutions_numeric"][0] is not None
    assert math.isclose(result["solutions_numeric"][0], 5.466e-14, rel_tol=1e-3)


def test_symbolic_math_evaluates_bare_expression_with_substitutions_dict():
    # Same fallback, but going through the intended substitutions path
    # rather than a fully pre-substituted expression.
    tool = SymbolicMathTool()
    result = tool.run(
        {
            "expression": "0.5 * m * v**2",
            "solve_for": "KE",  # never appears in the expression at all
            "substitutions": {"m": 2, "v": 3},
        }
    )
    assert "note" in result
    assert math.isclose(result["solutions_numeric"][0], 9.0, rel_tol=1e-9)


def test_symbolic_math_equation_with_no_matching_free_symbol_still_raises():
    # An actual Eq(...) where solve_for doesn't appear is NOT given the
    # bare-expression fallback -- ambiguous enough that erroring is still
    # correct (unlike a plain arithmetic expression, an equation implies a
    # genuine solve was intended).
    tool = SymbolicMathTool()
    with pytest.raises(ValueError):
        tool.run({"expression": "Eq(m*g*h, 0.5*m*v**2)", "solve_for": "KE", "substitutions": {"m": 2, "g": 9.8, "h": 5}})


# -- SimulationTool -----------------------------------------------------------


def test_simulation_free_fall_matches_closed_form():
    tool = SimulationTool()
    result = tool.run(
        {
            "state_vars": ["x", "v"],
            "derivatives": ["v", "-g"],
            "params": {"g": 9.8},
            "initial_conditions": {"x": 0, "v": 0},
            "t_span": [0, 1],
        }
    )
    # Closed form: v(1) = -g*1 = -9.8, x(1) = -0.5*g*1^2 = -4.9
    assert math.isclose(result["final_state"]["v"], -9.8, rel_tol=1e-3)
    assert math.isclose(result["final_state"]["x"], -4.9, rel_tol=1e-3)


def test_simulation_missing_fields_raises():
    tool = SimulationTool()
    with pytest.raises(ValueError):
        tool.run({"state_vars": ["x"], "derivatives": ["v"]})  # missing initial_conditions, t_span


def test_simulation_mismatched_lengths_raises():
    tool = SimulationTool()
    with pytest.raises(ValueError):
        tool.run(
            {
                "state_vars": ["x", "v"],
                "derivatives": ["v"],  # only one derivative for two state vars
                "params": {},
                "initial_conditions": {"x": 0, "v": 0},
                "t_span": [0, 1],
            }
        )


def test_simulation_missing_initial_condition_raises():
    tool = SimulationTool()
    with pytest.raises(ValueError):
        tool.run(
            {
                "state_vars": ["x", "v"],
                "derivatives": ["v", "-g"],
                "params": {"g": 9.8},
                "initial_conditions": {"x": 0},  # missing v
                "t_span": [0, 1],
            }
        )


# -- LiteratureSearchTool ------------------------------------------------------

FAKE_ARXIV_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title>A Study of Damped Harmonic Oscillators</title>
    <summary>This paper examines damping coefficients in oscillatory systems under various boundary conditions and presents a generalized framework for analysis across several distinct physical regimes of particular interest to experimental researchers in the field.</summary>
    <author><name>Jane Researcher</name></author>
    <author><name>John Scientist</name></author>
  </entry>
</feed>
"""


def test_literature_search_parses_fake_response():
    tool = LiteratureSearchTool(fetch_fn=lambda url: FAKE_ARXIV_RESPONSE)
    result = tool.run({"query": "damped harmonic oscillator", "max_results": 1})

    assert result["query"] == "damped harmonic oscillator"
    assert len(result["results"]) == 1
    entry = result["results"][0]
    assert entry["title"] == "A Study of Damped Harmonic Oscillators"
    assert entry["authors"] == ["Jane Researcher", "John Scientist"]
    assert entry["excerpt"].endswith("...")  # truncated since summary > 200 chars
    assert len(entry["excerpt"]) <= 203


def test_literature_search_missing_query_raises():
    tool = LiteratureSearchTool(fetch_fn=lambda url: FAKE_ARXIV_RESPONSE)
    with pytest.raises(ValueError):
        tool.run({})


def test_literature_search_fetch_failure_raises_value_error():
    def failing_fetch(url):
        raise ConnectionError("no network")

    tool = LiteratureSearchTool(fetch_fn=failing_fetch)
    with pytest.raises(ValueError):
        tool.run({"query": "anything"})


def test_literature_search_malformed_xml_raises():
    tool = LiteratureSearchTool(fetch_fn=lambda url: "not xml at all <<<")
    with pytest.raises(ValueError):
        tool.run({"query": "anything"})
