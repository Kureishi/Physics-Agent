import tempfile
from pathlib import Path

import pytest

from physics_agent.memory.procedural import ProceduralMemory
from physics_agent.meta_learning.strategy_override import StrategyOverridePolicy


@pytest.fixture
def mem_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "procedural.json"


def test_no_data_at_all_keeps_default(mem_path):
    mem = ProceduralMemory(mem_path)
    policy = StrategyOverridePolicy(mem)

    strategy, reason = policy.override(["energy"], "algebra_error", "rederive_math")
    assert strategy == "rederive_math"
    assert reason is None


def test_best_strategy_is_already_the_default_keeps_default(mem_path):
    mem = ProceduralMemory(mem_path)
    for _ in range(5):
        mem.record_outcome(["energy"], "algebra_error", "rederive_math", resolved=True)
    policy = StrategyOverridePolicy(mem)

    strategy, reason = policy.override(["energy"], "algebra_error", "rederive_math")
    assert strategy == "rederive_math"
    assert reason is None


def test_alternative_below_min_uses_is_not_trusted(mem_path):
    mem = ProceduralMemory(mem_path)
    # Alternative looks perfect but only has 4 uses -- below
    # MIN_USES_TO_OVERRIDE (5), even though it clears
    # ProceduralMemory.best_strategy_for's own floor of 3.
    for _ in range(4):
        mem.record_outcome(["energy"], "algebra_error", "alternative_strategy", resolved=True)
    policy = StrategyOverridePolicy(mem)

    strategy, reason = policy.override(["energy"], "algebra_error", "rederive_math")
    assert strategy == "rederive_math"
    assert reason is None


def test_untested_default_requires_alternative_to_clear_absolute_bar(mem_path):
    mem = ProceduralMemory(mem_path)
    # Default strategy has no data at all here. Alternative has 5 uses but
    # only a 40% success rate -- below MIN_ACCEPTABLE_SUCCESS_RATE (0.5).
    for resolved in [True, True, False, False, False]:
        mem.record_outcome(["energy"], "algebra_error", "alternative_strategy", resolved=resolved)
    policy = StrategyOverridePolicy(mem)

    strategy, reason = policy.override(["energy"], "algebra_error", "rederive_math")
    assert strategy == "rederive_math"
    assert reason is None


def test_untested_default_overridden_when_alternative_clears_absolute_bar(mem_path):
    mem = ProceduralMemory(mem_path)
    for resolved in [True, True, True, True, False]:  # 80%
        mem.record_outcome(["energy"], "algebra_error", "alternative_strategy", resolved=resolved)
    policy = StrategyOverridePolicy(mem)

    strategy, reason = policy.override(["energy"], "algebra_error", "rederive_math")
    assert strategy == "alternative_strategy"
    assert reason is not None
    assert "alternative_strategy" in reason
    assert "untried" in reason


def test_tested_default_requires_improvement_margin(mem_path):
    mem = ProceduralMemory(mem_path)
    # Default: 60% success rate over 5 uses.
    for resolved in [True, True, True, False, False]:
        mem.record_outcome(["energy"], "algebra_error", "rederive_math", resolved=resolved)
    # Alternative: 70% -- only 10 points better, under the 15-point margin.
    for resolved in [True, True, True, False, False, False, True, False, False, False]:
        mem.record_outcome(["energy"], "algebra_error", "alternative_strategy", resolved=resolved)
    policy = StrategyOverridePolicy(mem)

    strategy, reason = policy.override(["energy"], "algebra_error", "rederive_math")
    # Sanity check the fixture actually produced what the comment claims.
    default_entry = mem.get(["energy"], "algebra_error", "rederive_math")
    assert default_entry["success_rate"] == 0.6

    assert strategy == "rederive_math"
    assert reason is None


def test_tested_default_overridden_when_margin_cleared(mem_path):
    mem = ProceduralMemory(mem_path)
    # Default: 40% success rate over 5 uses.
    for resolved in [True, True, False, False, False]:
        mem.record_outcome(["energy"], "algebra_error", "rederive_math", resolved=resolved)
    # Alternative: 90% over 5 uses -- comfortably past the 15-point margin.
    for resolved in [True, True, True, True, False]:
        mem.record_outcome(["energy"], "algebra_error", "alternative_strategy", resolved=resolved)
    policy = StrategyOverridePolicy(mem)

    strategy, reason = policy.override(["energy"], "algebra_error", "rederive_math")
    assert strategy == "alternative_strategy"
    assert reason is not None
    assert "rederive_math" in reason  # comparison mentions what it replaced


def test_picks_the_best_of_several_alternatives(mem_path):
    mem = ProceduralMemory(mem_path)
    for resolved in [False] * 5:
        mem.record_outcome(["energy"], "algebra_error", "rederive_math", resolved=resolved)
    for resolved in [True, True, True, False, False]:  # 60%
        mem.record_outcome(["energy"], "algebra_error", "mediocre_alternative", resolved=resolved)
    for resolved in [True, True, True, True, False]:  # 80%
        mem.record_outcome(["energy"], "algebra_error", "best_alternative", resolved=resolved)
    policy = StrategyOverridePolicy(mem)

    strategy, reason = policy.override(["energy"], "algebra_error", "rederive_math")
    assert strategy == "best_alternative"


def test_different_error_type_is_not_considered(mem_path):
    mem = ProceduralMemory(mem_path)
    for _ in range(5):
        mem.record_outcome(["energy"], "physics_conceptual_error", "alternative_strategy", resolved=True)
    policy = StrategyOverridePolicy(mem)

    # No data at all for algebra_error specifically -- must not borrow the
    # physics_conceptual_error stats.
    strategy, reason = policy.override(["energy"], "algebra_error", "rederive_math")
    assert strategy == "rederive_math"
    assert reason is None
