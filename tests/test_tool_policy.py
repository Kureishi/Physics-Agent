import tempfile
from pathlib import Path

import pytest

from physics_agent.meta_learning.tool_policy import (
    ToolSelectionPolicy,
    MIN_USES_BEFORE_ACTING,
    POOR_SUCCESS_RATE_THRESHOLD,
)
from physics_agent.trace import EpisodicMemory, Trace, ToolCall


@pytest.fixture
def episodic_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "episodic.jsonl"


def _make_trace(domain_tags, tools_used, resolution_status, had_revision=False):
    trace = Trace.new("test problem")
    trace.domain_tags = domain_tags
    trace.resolution_status = resolution_status
    if had_revision:
        # simulate: round 0 used these tools, then got revised (so current
        # trace.tool_calls is a DIFFERENT round -- round 0 must come from
        # revision_history instead)
        trace.revision_history = [
            {
                "round": 0,
                "error_type": "algebra_error",
                "strategy": "rederive_math",
                "rationale": "x",
                "tool_calls": [{"tool": t, "input": "{}", "output": "{}", "latency_ms": 1.0} for t in tools_used],
                "initial_solution": "old",
                "checks_failed": ["math"],
                "check_details": [],
                "resolved": True,
            }
        ]
        trace.tool_calls = [ToolCall(tool="symbolic_math", input="{}", output="{}", latency_ms=1.0)]
    else:
        trace.tool_calls = [ToolCall(tool=t, input="{}", output="{}", latency_ms=1.0) for t in tools_used]
    return trace


def test_no_data_returns_none_success_rate(episodic_path):
    memory = EpisodicMemory(episodic_path)
    policy = ToolSelectionPolicy(memory)
    assert policy.success_rate("energy", "simulation") is None


def test_success_rate_requires_minimum_uses(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(MIN_USES_BEFORE_ACTING - 1):
        memory.write(_make_trace(["energy"], ["symbolic_math"], "passed_initial"))
    policy = ToolSelectionPolicy(memory)
    assert policy.success_rate("energy", "symbolic_math") is None  # one short of minimum


def test_success_rate_computed_once_minimum_reached(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(MIN_USES_BEFORE_ACTING):
        memory.write(_make_trace(["energy"], ["symbolic_math"], "passed_initial"))
    policy = ToolSelectionPolicy(memory)
    assert policy.success_rate("energy", "symbolic_math") == 1.0


def test_success_rate_reflects_mixed_outcomes(episodic_path):
    memory = EpisodicMemory(episodic_path)
    statuses = ["passed_initial"] * 3 + ["unresolved_max_revisions"] * 2
    for status in statuses:
        memory.write(_make_trace(["energy"], ["simulation"], status))
    policy = ToolSelectionPolicy(memory)
    assert policy.success_rate("energy", "simulation") == 0.6


def test_uses_round_0_tools_from_revision_history_when_revised(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(MIN_USES_BEFORE_ACTING):
        memory.write(
            _make_trace(["energy"], ["literature_search"], "resolved_after_revision", had_revision=True)
        )
    policy = ToolSelectionPolicy(memory)
    # literature_search was the ROUND 0 tool (from revision_history), even
    # though the final trace.tool_calls shows symbolic_math instead
    assert policy.success_rate("energy", "literature_search") is not None
    assert policy.success_rate("energy", "symbolic_math") is None  # never appeared as round-0


def test_incomplete_traces_are_ignored(episodic_path):
    memory = EpisodicMemory(episodic_path)
    incomplete = Trace.new("no resolution status set")
    incomplete.domain_tags = ["energy"]
    incomplete.tool_calls = [ToolCall(tool="symbolic_math", input="{}", output="{}", latency_ms=1.0)]
    for _ in range(MIN_USES_BEFORE_ACTING):
        memory.write(incomplete)
    policy = ToolSelectionPolicy(memory)
    assert policy.success_rate("energy", "symbolic_math") is None


def test_filter_tools_drops_poor_performer(episodic_path):
    memory = EpisodicMemory(episodic_path)
    # simulation has a well-established terrible track record for "optics"
    for _ in range(MIN_USES_BEFORE_ACTING + 2):
        memory.write(_make_trace(["optics"], ["simulation"], "unresolved_max_revisions"))
    policy = ToolSelectionPolicy(memory)

    result = policy.filter_tools(["optics"], ["simulation", "symbolic_math"])
    assert "simulation" not in result
    assert "symbolic_math" in result


def test_filter_tools_never_returns_empty(episodic_path):
    memory = EpisodicMemory(episodic_path)
    # both tools have terrible track records
    for _ in range(MIN_USES_BEFORE_ACTING + 2):
        memory.write(_make_trace(["optics"], ["simulation"], "unresolved_max_revisions"))
        memory.write(_make_trace(["optics"], ["symbolic_math"], "unresolved_max_revisions"))
    policy = ToolSelectionPolicy(memory)

    result = policy.filter_tools(["optics"], ["simulation", "symbolic_math"])
    assert len(result) == 2  # falls back to keeping everything rather than emptying


def test_filter_tools_keeps_unjudged_tools_without_penalty(episodic_path):
    memory = EpisodicMemory(episodic_path)
    # no data at all recorded -- everything should pass through unfiltered
    policy = ToolSelectionPolicy(memory)
    result = policy.filter_tools(["energy"], ["symbolic_math", "simulation", "literature_search"])
    assert set(result) == {"symbolic_math", "simulation", "literature_search"}


def test_filter_tools_orders_better_performers_first(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(MIN_USES_BEFORE_ACTING):
        memory.write(_make_trace(["energy"], ["symbolic_math"], "passed_initial"))  # 100%
        memory.write(_make_trace(["energy"], ["simulation"], "unresolved_max_revisions"))  # 0%, but above threshold? No: 0.0 < POOR_SUCCESS_RATE_THRESHOLD so dropped
    policy = ToolSelectionPolicy(memory)
    result = policy.filter_tools(["energy"], ["simulation", "symbolic_math"])
    assert result[0] == "symbolic_math"
