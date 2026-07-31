import tempfile
from pathlib import Path

import pytest

from physics_agent.inspect_trace_cli import find_matches, print_full, print_short
from physics_agent.trace import EpisodicMemory, Trace, ToolCall


@pytest.fixture
def episodic_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "episodic.jsonl"


def _make_trace(problem_text, source="user"):
    trace = Trace.new(problem_text)
    trace.source = source
    trace.domain_tags = ["energy"]
    trace.tool_calls = [ToolCall(tool="symbolic_math", input="{}", output="{}", latency_ms=1.0)]
    trace.check_details = [{"check": "logic", "passed": True, "details": "fine"}]
    trace.resolution_status = "passed_initial"
    return trace


def test_find_matches_by_query_substring(episodic_path):
    memory = EpisodicMemory(episodic_path)
    memory.write(_make_trace("An electron moves at 0.8c"))
    memory.write(_make_trace("A block slides down an incline"))

    matches = find_matches(memory, query="electron", problem_id=None)
    assert len(matches) == 1
    assert "electron" in matches[0].problem_text


def test_find_matches_by_query_is_case_insensitive(episodic_path):
    memory = EpisodicMemory(episodic_path)
    memory.write(_make_trace("An Electron moves at 0.8c"))

    matches = find_matches(memory, query="electron", problem_id=None)
    assert len(matches) == 1


def test_find_matches_by_exact_id(episodic_path):
    memory = EpisodicMemory(episodic_path)
    t1 = _make_trace("problem one")
    t2 = _make_trace("problem two")
    memory.write(t1)
    memory.write(t2)

    matches = find_matches(memory, query=None, problem_id=t1.problem_id)
    assert len(matches) == 1
    assert matches[0].problem_id == t1.problem_id


def test_find_matches_returns_all_when_neither_given(episodic_path):
    memory = EpisodicMemory(episodic_path)
    memory.write(_make_trace("problem one"))
    memory.write(_make_trace("problem two"))

    matches = find_matches(memory, query=None, problem_id=None)
    assert len(matches) == 2


def test_find_matches_returns_empty_for_no_match(episodic_path):
    memory = EpisodicMemory(episodic_path)
    memory.write(_make_trace("a block slides"))

    matches = find_matches(memory, query="nonexistent topic", problem_id=None)
    assert matches == []


def test_find_matches_multiple_substring_matches(episodic_path):
    memory = EpisodicMemory(episodic_path)
    memory.write(_make_trace("An electron moves fast"))
    memory.write(_make_trace("Another electron in a field"))

    matches = find_matches(memory, query="electron", problem_id=None)
    assert len(matches) == 2


def test_print_short_does_not_raise(capsys):
    trace = _make_trace("test problem")
    print_short(trace)
    captured = capsys.readouterr()
    assert trace.problem_id in captured.out
    assert "test problem" in captured.out


def test_print_full_includes_key_sections(capsys):
    trace = _make_trace("An electron moves at 0.8c")
    trace.revision_history = [
        {
            "round": 0,
            "error_type": "physics_conceptual_error",
            "strategy": "rederive_physics_setup",
            "rationale": "physics check failed",
            "tool_calls": [],
            "initial_solution": "wrong solution",
            "checks_failed": ["physics"],
            "check_details": [{"check": "physics", "passed": False, "details": "used wrong formula"}],
            "resolved": False,
        }
    ]
    trace.resolution_status = "unresolved_max_revisions"
    trace.final_answer = "some answer"

    print_full(trace)
    captured = capsys.readouterr()

    assert "An electron moves at 0.8c" in captured.out
    assert "physics_conceptual_error" in captured.out
    assert "rederive_physics_setup" in captured.out
    assert "used wrong formula" in captured.out
    assert "unresolved_max_revisions" in captured.out
    assert "some answer" in captured.out


def test_print_full_handles_no_revisions_gracefully(capsys):
    trace = _make_trace("simple problem")
    print_full(trace)
    captured = capsys.readouterr()
    assert "no revisions needed" in captured.out


def test_print_full_handles_unevaluated_trace_gracefully(capsys):
    # A trace that hasn't gone through self-eval yet (e.g. Stage 1/2 only)
    trace = Trace.new("bare problem")
    print_full(trace)  # should not raise despite mostly-empty fields
    captured = capsys.readouterr()
    assert "not yet self-evaluated" in captured.out
    assert "no revisions needed" in captured.out
