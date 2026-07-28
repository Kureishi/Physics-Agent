import json
import tempfile
from pathlib import Path

import pytest

from physics_agent.memory.procedural import ProceduralMemory


@pytest.fixture
def mem_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "procedural.json"


def test_creates_empty_store_if_missing(mem_path):
    mem = ProceduralMemory(mem_path)
    assert mem.all_entries() == []
    assert mem_path.exists()


def test_record_outcome_creates_new_entry(mem_path):
    mem = ProceduralMemory(mem_path)
    entry = mem.record_outcome(
        domain_tags=["energy", "dynamics"], error_type="algebra_error", strategy="rederive_math", resolved=True
    )
    assert entry["n_uses"] == 1
    assert entry["n_successes"] == 1
    assert entry["success_rate"] == 1.0
    assert entry["domain_tags"] == ["dynamics", "energy"]  # sorted


def test_record_outcome_accumulates_success_rate(mem_path):
    mem = ProceduralMemory(mem_path)
    for resolved in [True, True, False, True]:
        entry = mem.record_outcome(["energy"], "algebra_error", "rederive_math", resolved=resolved)
    assert entry["n_uses"] == 4
    assert entry["n_successes"] == 3
    assert entry["success_rate"] == 0.75


def test_different_domain_tag_order_maps_to_same_entry(mem_path):
    mem = ProceduralMemory(mem_path)
    mem.record_outcome(["energy", "dynamics"], "algebra_error", "rederive_math", resolved=True)
    mem.record_outcome(["dynamics", "energy"], "algebra_error", "rederive_math", resolved=False)
    entry = mem.get(["energy", "dynamics"], "algebra_error", "rederive_math")
    assert entry["n_uses"] == 2  # both calls hit the same entry


def test_different_strategy_is_a_different_entry(mem_path):
    mem = ProceduralMemory(mem_path)
    mem.record_outcome(["energy"], "physics_conceptual_error", "rederive_physics_setup", resolved=True)
    mem.record_outcome(["energy"], "physics_conceptual_error", "full_replan", resolved=False)
    assert len(mem.all_entries()) == 2


def test_persists_across_reload(mem_path):
    mem = ProceduralMemory(mem_path)
    mem.record_outcome(["energy"], "algebra_error", "rederive_math", resolved=True)

    reloaded = ProceduralMemory(mem_path)
    entry = reloaded.get(["energy"], "algebra_error", "rederive_math")
    assert entry is not None
    assert entry["n_uses"] == 1


def test_best_strategy_for_requires_minimum_uses(mem_path):
    mem = ProceduralMemory(mem_path)
    mem.record_outcome(["energy"], "algebra_error", "rederive_math", resolved=True)
    mem.record_outcome(["energy"], "algebra_error", "rederive_math", resolved=True)
    # only 2 uses -- below the minimum of 3
    assert mem.best_strategy_for(["energy"], "algebra_error") is None

    mem.record_outcome(["energy"], "algebra_error", "rederive_math", resolved=True)
    result = mem.best_strategy_for(["energy"], "algebra_error")
    assert result is not None
    assert result["strategy"] == "rederive_math"


def test_best_strategy_for_picks_highest_success_rate(mem_path):
    mem = ProceduralMemory(mem_path)
    for resolved in [True, True, True]:
        mem.record_outcome(["energy"], "physics_conceptual_error", "strategy_a", resolved=resolved)
    for resolved in [False, False, True]:
        mem.record_outcome(["energy"], "physics_conceptual_error", "strategy_b", resolved=resolved)

    best = mem.best_strategy_for(["energy"], "physics_conceptual_error")
    assert best["strategy"] == "strategy_a"
