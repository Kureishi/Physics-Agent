import tempfile
from pathlib import Path

import pytest

from physics_agent.meta_learning.anomaly import detect_check_value_anomalies
from physics_agent.trace import EpisodicMemory, Trace


@pytest.fixture
def episodic_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "episodic.jsonl"


def _write_trace(memory: EpisodicMemory, checks_failed) -> None:
    t = Trace.new("x")
    t.checks_run = ["logic", "physics", "math", "confidence"]
    t.checks_failed = list(checks_failed)
    memory.write(t)


def test_insufficient_data_when_too_few_traces(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(10):
        _write_trace(memory, [])

    report = detect_check_value_anomalies(memory, recent_window=20, min_baseline_n=20)
    assert report["status"] == "insufficient_data"
    assert report["flags"] == []


def test_no_flags_when_catch_rate_is_stable(episodic_path):
    memory = EpisodicMemory(episodic_path)
    # 30 baseline traces, math fails ~10% of the time; 20 recent traces,
    # same ~10% rate -- nothing should be flagged.
    for i in range(30):
        _write_trace(memory, ["math"] if i % 10 == 0 else [])
    for i in range(20):
        _write_trace(memory, ["math"] if i % 10 == 0 else [])

    report = detect_check_value_anomalies(memory, recent_window=20, min_baseline_n=20)
    assert report["status"] == "ok"
    assert report["n_baseline"] == 30
    assert report["n_recent"] == 20
    assert report["flags"] == []


def test_flags_a_jump_reproducing_the_mathcheck_bug_shape(episodic_path):
    memory = EpisodicMemory(episodic_path)
    # Baseline: MathCheck almost never fails (~5%).
    for i in range(30):
        _write_trace(memory, ["math"] if i == 0 else [])
    # Recent: MathCheck suddenly fails on most traces (~35 of 40, i.e.
    # ~87%) -- the real MathCheck bug's actual measured shape (190/199).
    for i in range(40):
        _write_trace(memory, ["math"] if i < 35 else [])

    report = detect_check_value_anomalies(memory, recent_window=40, min_baseline_n=20)
    assert report["status"] == "ok"

    flagged_checks = {f["check"] for f in report["flags"]}
    assert "math" in flagged_checks

    math_flag = next(f for f in report["flags"] if f["check"] == "math")
    assert math_flag["direction"] == "jump"
    assert math_flag["delta"] > 0
    assert "math" in math_flag["reason"]


def test_flags_a_collapse(episodic_path):
    memory = EpisodicMemory(episodic_path)
    # Baseline: physics check fails often (~50%).
    for i in range(30):
        _write_trace(memory, ["physics"] if i % 2 == 0 else [])
    # Recent: physics check stops catching anything at all.
    for i in range(20):
        _write_trace(memory, [])

    report = detect_check_value_anomalies(memory, recent_window=20, min_baseline_n=20)
    physics_flag = next(f for f in report["flags"] if f["check"] == "physics")
    assert physics_flag["direction"] == "collapse"
    assert physics_flag["delta"] < 0


def test_small_delta_under_threshold_is_not_flagged(episodic_path):
    memory = EpisodicMemory(episodic_path)
    # Baseline ~10% catch rate, recent ~15% -- a 5-point move, under the
    # default 15-point threshold.
    for i in range(30):
        _write_trace(memory, ["logic"] if i % 10 == 0 else [])
    for i in range(20):
        _write_trace(memory, ["logic"] if i % 7 == 0 else [])

    report = detect_check_value_anomalies(memory, recent_window=20, min_baseline_n=20)
    logic_flags = [f for f in report["flags"] if f["check"] == "logic"]
    assert logic_flags == []


def test_flags_sorted_by_magnitude_of_delta_descending(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for i in range(30):
        # math: 0% baseline. physics: 0% baseline.
        _write_trace(memory, [])
    for i in range(20):
        failed = []
        if i < 4:   # math: 20% recent -> delta 0.20
            failed.append("math")
        if i < 16:  # physics: 80% recent -> delta 0.80
            failed.append("physics")
        _write_trace(memory, failed)

    report = detect_check_value_anomalies(memory, recent_window=20, min_baseline_n=20)
    checks_in_order = [f["check"] for f in report["flags"]]
    assert checks_in_order.index("physics") < checks_in_order.index("math")
