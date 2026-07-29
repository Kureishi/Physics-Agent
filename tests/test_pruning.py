import tempfile
from pathlib import Path

import pytest

from physics_agent.memory.procedural import ProceduralMemory
from physics_agent.meta_learning.pruning import flag_declining_strategies


@pytest.fixture
def procedural():
    with tempfile.TemporaryDirectory() as d:
        yield ProceduralMemory(Path(d) / "procedural.json")


def test_no_entries_returns_empty(procedural):
    assert flag_declining_strategies(procedural) == []


def test_flags_low_success_rate_with_enough_uses(procedural):
    for resolved in [False, False, False, False, True]:  # 1/5 = 0.2
        procedural.record_outcome(["energy"], "algebra_error", "rederive_math", resolved=resolved)

    flagged = flag_declining_strategies(procedural)
    assert len(flagged) == 1
    assert flagged[0]["strategy"] == "rederive_math"


def test_does_not_flag_below_minimum_uses(procedural):
    for resolved in [False, False]:  # only 2 uses, both failed
        procedural.record_outcome(["energy"], "algebra_error", "rederive_math", resolved=resolved)
    assert flag_declining_strategies(procedural) == []


def test_does_not_flag_good_success_rate(procedural):
    for resolved in [True, True, True, True, False]:  # 4/5 = 0.8
        procedural.record_outcome(["energy"], "algebra_error", "rederive_math", resolved=resolved)
    assert flag_declining_strategies(procedural) == []


def test_respects_custom_thresholds(procedural):
    for resolved in [True, False, False, False, False]:  # 0.2 success rate
        procedural.record_outcome(["energy"], "algebra_error", "rederive_math", resolved=resolved)

    assert flag_declining_strategies(procedural, threshold=0.1) == []  # 0.2 > 0.1, not flagged
    assert len(flag_declining_strategies(procedural, threshold=0.5)) == 1


def test_sorted_worst_first(procedural):
    for resolved in [False, False, False, False, True]:  # 0.2
        procedural.record_outcome(["energy"], "algebra_error", "strategy_a", resolved=resolved)
    for resolved in [False, False, True, True, True]:  # 0.6 -- not flagged at default threshold 0.34
        procedural.record_outcome(["energy"], "algebra_error", "strategy_b", resolved=resolved)
    for resolved in [False, False, False, False, False]:  # 0.0 -- worse than strategy_a
        procedural.record_outcome(["energy"], "physics_conceptual_error", "strategy_c", resolved=resolved)

    flagged = flag_declining_strategies(procedural)
    strategies_in_order = [e["strategy"] for e in flagged]
    assert strategies_in_order == ["strategy_c", "strategy_a"]
