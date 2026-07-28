import json
import tempfile
from pathlib import Path

import pytest

from physics_agent.memory.consolidator import MemoryConsolidator
from physics_agent.memory.error_memory import ErrorMemory
from physics_agent.memory.procedural import ProceduralMemory
from physics_agent.retrieval import SemanticStore
from physics_agent.trace import EpisodicMemory, Trace


SEED_ENTRIES = [
    {
        "id": "eng-001",
        "statement": "KE = 0.5*m*v^2",
        "conditions": "Non-relativistic",
        "confidence": 0.9,
        "provenance": "seed",
        "tags": ["energy"],
        "last_validated": 0,
    }
]


@pytest.fixture
def consolidator():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        with (d / "semantic.json").open("w") as f:
            json.dump(SEED_ENTRIES, f)

        episodic = EpisodicMemory(d / "episodic.jsonl")
        semantic = SemanticStore(d / "semantic.json")
        procedural = ProceduralMemory(d / "procedural.json")
        error = ErrorMemory(d / "error.json")

        yield MemoryConsolidator(episodic, semantic, procedural, error), episodic, semantic, procedural, error


def _make_trace(checks_failed=None, revision_history=None):
    trace = Trace.new("A ball is dropped from 10 m. Find its speed at impact.")
    trace.domain_tags = ["energy"]
    trace.retrieved_knowledge = [{"id": "eng-001", "statement": "KE = 0.5*m*v^2", "conditions": "..."}]
    trace.checks_failed = checks_failed or []
    trace.revision_history = revision_history or []
    return trace


def test_consolidate_writes_episodic_trace(consolidator):
    consolidator_obj, episodic, semantic, procedural, error = consolidator
    trace = _make_trace()

    consolidator_obj.consolidate(trace)

    all_traces = episodic.read_all()
    assert len(all_traces) == 1
    assert all_traces[0].problem_id == trace.problem_id


def test_consolidate_bumps_semantic_confidence_up_on_success(consolidator):
    consolidator_obj, episodic, semantic, procedural, error = consolidator
    trace = _make_trace(checks_failed=[])  # no failures -> success

    consolidator_obj.consolidate(trace)

    entry = next(e for e in semantic.entries if e["id"] == "eng-001")
    assert entry["confidence"] > 0.9


def test_consolidate_bumps_semantic_confidence_down_on_failure(consolidator):
    consolidator_obj, episodic, semantic, procedural, error = consolidator
    trace = _make_trace(checks_failed=["physics"])  # still failing -> not success

    consolidator_obj.consolidate(trace)

    entry = next(e for e in semantic.entries if e["id"] == "eng-001")
    assert entry["confidence"] < 0.9


def test_consolidate_updates_procedural_and_error_memory_per_revision_round(consolidator):
    consolidator_obj, episodic, semantic, procedural, error = consolidator
    trace = _make_trace(
        checks_failed=[],
        revision_history=[
            {
                "round": 0,
                "error_type": "algebra_error",
                "strategy": "rederive_math",
                "rationale": "math check failed",
                "tool_calls": [],
                "initial_solution": "old",
                "checks_failed": ["math"],
                "check_details": [],
                "resolved": True,
            }
        ],
    )

    consolidator_obj.consolidate(trace)

    proc_entry = procedural.get(["energy"], "algebra_error", "rederive_math")
    assert proc_entry is not None
    assert proc_entry["n_uses"] == 1
    assert proc_entry["success_rate"] == 1.0

    err_entry = error.get("algebra_error", ["energy"])
    assert err_entry is not None
    assert err_entry["frequency"] == 1
    assert err_entry["resolved"] is True


def test_consolidate_handles_no_revisions_gracefully(consolidator):
    consolidator_obj, episodic, semantic, procedural, error = consolidator
    trace = _make_trace(checks_failed=[], revision_history=[])

    consolidator_obj.consolidate(trace)  # should not raise

    assert procedural.all_entries() == []
    assert error.all_entries() == []


def test_consolidate_multiple_revision_rounds_each_recorded(consolidator):
    consolidator_obj, episodic, semantic, procedural, error = consolidator
    trace = _make_trace(
        checks_failed=["physics"],
        revision_history=[
            {
                "round": 0,
                "error_type": "algebra_error",
                "strategy": "rederive_math",
                "rationale": "r1",
                "tool_calls": [],
                "initial_solution": "old1",
                "checks_failed": ["math"],
                "check_details": [],
                "resolved": True,
            },
            {
                "round": 1,
                "error_type": "cross_method_disagreement",
                "strategy": "rederive_physics_setup",
                "rationale": "r2",
                "tool_calls": [],
                "initial_solution": "old2",
                "checks_failed": ["physics"],
                "check_details": [],
                "resolved": False,
            },
        ],
    )

    consolidator_obj.consolidate(trace)

    assert len(procedural.all_entries()) == 2
    assert len(error.all_entries()) == 2
    assert error.get("cross_method_disagreement", ["energy"])["resolved"] is False
