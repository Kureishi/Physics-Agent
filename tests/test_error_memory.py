import tempfile
from pathlib import Path

import pytest

from physics_agent.memory.error_memory import ErrorMemory


@pytest.fixture
def mem_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "error_memory.json"


def test_creates_empty_store_if_missing(mem_path):
    mem = ErrorMemory(mem_path)
    assert mem.all_entries() == []
    assert mem_path.exists()


def test_record_creates_new_entry(mem_path):
    mem = ErrorMemory(mem_path)
    entry = mem.record(
        error_type="cross_method_disagreement",
        domain_tags=["energy", "dynamics"],
        root_cause="symbolic and simulation disagreed",
        fix_applied="rederive_physics_setup",
        resolved=True,
    )
    assert entry["frequency"] == 1
    assert entry["resolved"] is True
    assert entry["domain_tags"] == ["dynamics", "energy"]


def test_record_increments_frequency_on_recurrence(mem_path):
    mem = ErrorMemory(mem_path)
    mem.record("algebra_error", ["energy"], "bad algebra", "rederive_math", resolved=True)
    mem.record("algebra_error", ["energy"], "bad algebra again", "rederive_math", resolved=False)
    entry = mem.get("algebra_error", ["energy"])
    assert entry["frequency"] == 2
    # most recent occurrence's details win
    assert entry["root_cause"] == "bad algebra again"
    assert entry["resolved"] is False


def test_different_domain_is_a_different_signature(mem_path):
    mem = ErrorMemory(mem_path)
    mem.record("algebra_error", ["energy"], "x", "rederive_math", resolved=True)
    mem.record("algebra_error", ["quantum-mechanics"], "y", "rederive_math", resolved=True)
    assert len(mem.all_entries()) == 2


def test_most_frequent_orders_by_frequency(mem_path):
    mem = ErrorMemory(mem_path)
    for _ in range(5):
        mem.record("algebra_error", ["energy"], "x", "rederive_math", resolved=True)
    for _ in range(2):
        mem.record("reasoning_inconsistency", ["energy"], "y", "resynthesize", resolved=True)

    top = mem.most_frequent(limit=1)
    assert len(top) == 1
    assert top[0]["error_type"] == "algebra_error"
    assert top[0]["frequency"] == 5


def test_persists_across_reload(mem_path):
    mem = ErrorMemory(mem_path)
    mem.record("algebra_error", ["energy"], "x", "rederive_math", resolved=True)

    reloaded = ErrorMemory(mem_path)
    entry = reloaded.get("algebra_error", ["energy"])
    assert entry is not None
    assert entry["frequency"] == 1
