import tempfile
from pathlib import Path

import pytest

from physics_agent.meta_learning.check_value import compute_check_value_report
from physics_agent.trace import EpisodicMemory, Trace


@pytest.fixture
def episodic_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "episodic.jsonl"


def test_empty_memory_returns_known_checks_with_zero_stats(episodic_path):
    memory = EpisodicMemory(episodic_path)
    report = compute_check_value_report(memory)
    assert set(report.keys()) == {"logic", "physics", "math", "confidence"}
    for stats in report.values():
        assert stats["n_traces"] == 0
        assert stats["catch_rate"] == 0.0


def test_check_that_always_passes_has_zero_catch_rate(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(3):
        t = Trace.new("x")
        t.checks_run = ["logic", "physics", "math", "confidence"]
        t.checks_failed = []
        memory.write(t)

    report = compute_check_value_report(memory)
    assert report["math"]["catch_rate"] == 0.0
    assert report["math"]["n_traces"] == 3


def test_check_that_fails_in_final_round_is_counted(episodic_path):
    memory = EpisodicMemory(episodic_path)
    t = Trace.new("x")
    t.checks_run = ["logic", "physics", "math", "confidence"]
    t.checks_failed = ["physics"]
    memory.write(t)

    report = compute_check_value_report(memory)
    assert report["physics"]["n_ever_failed"] == 1
    assert report["physics"]["catch_rate"] == 1.0
    assert report["logic"]["catch_rate"] == 0.0


def test_check_that_only_failed_in_an_earlier_revision_round_is_still_counted(episodic_path):
    # By the time a trace resolves, trace.checks_failed reflects only the
    # FINAL round -- a check that failed and got fixed in round 0 should
    # still count as "ever failed" via revision_history.
    memory = EpisodicMemory(episodic_path)
    t = Trace.new("x")
    t.checks_run = ["logic", "physics", "math", "confidence"]
    t.checks_failed = []  # final round: all clean
    t.revision_history = [
        {
            "round": 0,
            "error_type": "algebra_error",
            "strategy": "rederive_math",
            "rationale": "x",
            "tool_calls": [],
            "initial_solution": "old",
            "checks_failed": ["math"],
            "check_details": [],
            "resolved": True,
        }
    ]
    memory.write(t)

    report = compute_check_value_report(memory)
    assert report["math"]["n_ever_failed"] == 1
    assert report["math"]["catch_rate"] == 1.0


def test_traces_with_no_checks_run_are_excluded(episodic_path):
    memory = EpisodicMemory(episodic_path)
    incomplete = Trace.new("x")  # checks_run never populated
    memory.write(incomplete)

    report = compute_check_value_report(memory)
    assert report["math"]["n_traces"] == 0
