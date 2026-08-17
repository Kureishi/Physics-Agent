import json

import pytest

from physics_agent.meta_learning.knowledge_growth import (
    ProposedFactsRegistry,
    find_candidate_facts,
    propose_and_add,
)
from physics_agent.retrieval import SemanticStore
from physics_agent.trace import EpisodicMemory, Trace, ToolCall


@pytest.fixture
def episodic_path(tmp_path):
    return tmp_path / "episodic.jsonl"


@pytest.fixture
def semantic_path(tmp_path):
    path = tmp_path / "semantic.json"
    with path.open("w") as f:
        json.dump([], f)
    return path


@pytest.fixture
def registry_path(tmp_path):
    return tmp_path / "proposed_facts.json"


def _symbolic_math_call(expression, solve_for="v", solutions_numeric=None):
    output = {
        "expression": expression,
        "solve_for": solve_for,
        "substitutions": {},
        "solutions": [str(v) for v in (solutions_numeric or [9.9])],
        "solutions_numeric": solutions_numeric or [9.9],
    }
    return ToolCall(tool="symbolic_math", input="{}", output=json.dumps(output), latency_ms=5.0)


def _write_trace(memory, expression, domain_tags, resolution_status="passed_initial", solve_for="v"):
    t = Trace.new("x")
    t.domain_tags = domain_tags
    t.resolution_status = resolution_status
    t.tool_calls = [_symbolic_math_call(expression, solve_for=solve_for)]
    memory.write(t)
    return t


# -- find_candidate_facts -----------------------------------------------


def test_no_candidates_on_empty_memory(episodic_path):
    memory = EpisodicMemory(episodic_path)
    assert find_candidate_facts(memory) == []


def test_below_min_observations_not_a_candidate(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(4):
        _write_trace(memory, "Eq(m*g*h, 0.5*m*v**2)", ["dynamics"])

    assert find_candidate_facts(memory, min_observations=5) == []


def test_at_min_observations_becomes_a_candidate(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(5):
        _write_trace(memory, "Eq(m*g*h, 0.5*m*v**2)", ["dynamics"])

    candidates = find_candidate_facts(memory, min_observations=5)
    assert len(candidates) == 1
    c = candidates[0]
    assert c["expression"] == "Eq(m*g*h, 0.5*m*v**2)"
    assert c["solve_for"] == "v"
    assert c["n_observations"] == 5
    assert c["domain_tags"] == ["dynamics"]


def test_whitespace_variants_are_normalized_to_the_same_signature(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(3):
        _write_trace(memory, "Eq(m*g*h, 0.5*m*v**2)", ["dynamics"])
    for _ in range(2):
        _write_trace(memory, "Eq(m*g*h,   0.5*m*v**2)", ["dynamics"])  # extra whitespace

    candidates = find_candidate_facts(memory, min_observations=5)
    assert len(candidates) == 1
    assert candidates[0]["n_observations"] == 5


def test_different_expressions_are_separate_candidates(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(5):
        _write_trace(memory, "Eq(m*g*h, 0.5*m*v**2)", ["dynamics"])
    for _ in range(5):
        _write_trace(memory, "Eq(v**2, 2*g*h)", ["dynamics"])  # equivalent physics, different string

    candidates = find_candidate_facts(memory, min_observations=5)
    assert len(candidates) == 2


def test_unresolved_traces_do_not_count(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(5):
        _write_trace(memory, "Eq(m*g*h, 0.5*m*v**2)", ["dynamics"], resolution_status="unresolved_max_revisions")

    assert find_candidate_facts(memory, min_observations=5) == []


def test_escalated_traces_do_not_count(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(5):
        _write_trace(memory, "Eq(m*g*h, 0.5*m*v**2)", ["dynamics"], resolution_status="escalated_for_human_review")

    assert find_candidate_facts(memory, min_observations=5) == []


def test_resolved_after_revision_does_count(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(5):
        _write_trace(memory, "Eq(m*g*h, 0.5*m*v**2)", ["dynamics"], resolution_status="resolved_after_revision")

    assert len(find_candidate_facts(memory, min_observations=5)) == 1


def test_non_symbolic_math_tool_calls_are_ignored(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(5):
        t = Trace.new("x")
        t.domain_tags = ["optics"]
        t.resolution_status = "passed_initial"
        t.tool_calls = [
            ToolCall(tool="literature_search", input="{}", output=json.dumps({"results": []}), latency_ms=5.0)
        ]
        memory.write(t)

    assert find_candidate_facts(memory, min_observations=5) == []


def test_errored_tool_calls_are_ignored(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(5):
        t = Trace.new("x")
        t.domain_tags = ["dynamics"]
        t.resolution_status = "passed_initial"
        t.tool_calls = [
            ToolCall(tool="symbolic_math", input="{}", output=json.dumps({"error": "bad input"}), latency_ms=5.0)
        ]
        memory.write(t)

    assert find_candidate_facts(memory, min_observations=5) == []


def test_domain_tags_union_across_contributing_traces(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(3):
        _write_trace(memory, "Eq(m*g*h, 0.5*m*v**2)", ["dynamics"])
    for _ in range(2):
        _write_trace(memory, "Eq(m*g*h, 0.5*m*v**2)", ["energy"])

    candidates = find_candidate_facts(memory, min_observations=5)
    assert candidates[0]["domain_tags"] == ["dynamics", "energy"]


def test_candidates_ranked_by_observation_count_descending(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(5):
        _write_trace(memory, "Eq(a, 1)", ["dynamics"])
    for _ in range(8):
        _write_trace(memory, "Eq(b, 2)", ["dynamics"])

    candidates = find_candidate_facts(memory, min_observations=5)
    assert [c["expression"] for c in candidates] == ["Eq(b, 2)", "Eq(a, 1)"]


def test_only_final_round_tool_calls_are_considered():
    # trace.tool_calls holds only the FINAL round's calls by the time
    # solving finishes (per self_correction/engine.py) -- a candidate's
    # expression should reflect that, not anything from revision_history.
    t = Trace.new("x")
    t.domain_tags = ["dynamics"]
    t.resolution_status = "resolved_after_revision"
    t.tool_calls = [_symbolic_math_call("Eq(final, 1)")]
    t.revision_history = [
        {
            "tool_calls": [{"tool": "symbolic_math", "input": "{}", "output": json.dumps({"expression": "Eq(stale, 0)", "solve_for": "v", "solutions": [], "solutions_numeric": []}), "latency_ms": 1.0}],
            "checks_failed": ["math"],
            "strategy": "rederive_math",
            "resolved": True,
        }
    ]
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        memory = EpisodicMemory(Path(d) / "episodic.jsonl")
        for _ in range(5):
            fresh = Trace.new("x")
            fresh.domain_tags = t.domain_tags
            fresh.resolution_status = t.resolution_status
            fresh.tool_calls = t.tool_calls
            fresh.revision_history = t.revision_history
            memory.write(fresh)

        candidates = find_candidate_facts(memory, min_observations=5)
        expressions = [c["expression"] for c in candidates]
        assert "Eq(final, 1)" in expressions
        assert "Eq(stale, 0)" not in expressions


# -- propose_and_add / ProposedFactsRegistry -----------------------------


def test_propose_and_add_commits_qualifying_candidates(semantic_path, registry_path):
    semantic = SemanticStore(semantic_path)
    registry = ProposedFactsRegistry(registry_path)
    candidates = [
        {
            "signature": "Eq(m*g*h,0.5*m*v**2)",
            "expression": "Eq(m*g*h, 0.5*m*v**2)",
            "solve_for": "v",
            "domain_tags": ["dynamics", "energy"],
            "trace_ids": ["t1", "t2", "t3", "t4", "t5"],
            "n_observations": 5,
        }
    ]

    added = propose_and_add(semantic, registry, candidates)

    assert len(added) == 1
    assert added[0]["n_observations"] == 5
    assert len(semantic.entries) == 1

    entry = semantic.entries[0]
    assert entry["statement"] == "Eq(m*g*h, 0.5*m*v**2), solved for v"
    assert entry["confidence"] == 0.3
    assert "self_derived" in entry["provenance"]
    assert "Unverified" in entry["conditions"]
    assert entry["tags"] == ["dynamics", "energy"]


def test_propose_and_add_is_idempotent_across_runs(semantic_path, registry_path):
    semantic = SemanticStore(semantic_path)
    registry = ProposedFactsRegistry(registry_path)
    candidates = [
        {
            "signature": "Eq(a,1)",
            "expression": "Eq(a, 1)",
            "solve_for": "a",
            "domain_tags": ["dynamics"],
            "trace_ids": ["t1", "t2", "t3", "t4", "t5"],
            "n_observations": 5,
        }
    ]

    first = propose_and_add(semantic, registry, candidates)
    assert len(first) == 1

    # Re-running with the same candidate (as find_candidate_facts would
    # return again on the next scheduler cycle) must not duplicate it.
    second = propose_and_add(semantic, registry, candidates)
    assert second == []
    assert len(semantic.entries) == 1


def test_propose_and_add_persists_registry_across_instances(semantic_path, registry_path):
    semantic = SemanticStore(semantic_path)
    registry1 = ProposedFactsRegistry(registry_path)
    candidates = [
        {
            "signature": "Eq(a,1)",
            "expression": "Eq(a, 1)",
            "solve_for": "a",
            "domain_tags": ["dynamics"],
            "trace_ids": ["t1", "t2", "t3", "t4", "t5"],
            "n_observations": 5,
        }
    ]
    propose_and_add(semantic, registry1, candidates)

    registry2 = ProposedFactsRegistry(registry_path)  # simulates a fresh process
    assert registry2.already_proposed("Eq(a,1)")
    assert propose_and_add(semantic, registry2, candidates) == []


def test_propose_and_add_truncates_long_trace_id_lists_in_provenance(semantic_path, registry_path):
    semantic = SemanticStore(semantic_path)
    registry = ProposedFactsRegistry(registry_path)
    trace_ids = [f"t{i}" for i in range(12)]
    candidates = [
        {
            "signature": "Eq(a,1)",
            "expression": "Eq(a, 1)",
            "solve_for": "a",
            "domain_tags": ["dynamics"],
            "trace_ids": trace_ids,
            "n_observations": 12,
        }
    ]

    propose_and_add(semantic, registry, candidates)
    provenance = semantic.entries[0]["provenance"]
    assert "+7 more" in provenance
    assert "t0" in provenance
    assert "t11" not in provenance  # only the first MAX_PROVENANCE_TRACE_IDS shown


def test_same_signature_always_maps_to_same_entry_id(semantic_path, registry_path):
    semantic = SemanticStore(semantic_path)
    registry = ProposedFactsRegistry(registry_path)
    candidate = {
        "signature": "Eq(a,1)",
        "expression": "Eq(a, 1)",
        "solve_for": "a",
        "domain_tags": ["dynamics"],
        "trace_ids": ["t1", "t2", "t3", "t4", "t5"],
        "n_observations": 5,
    }

    added1 = propose_and_add(semantic, registry, [candidate])
    entry_id = added1[0]["entry_id"]

    # A fresh registry (simulating the registry file being lost while the
    # semantic store survives) re-deriving the id from the same signature
    # should collide with the existing entry id, not invent a new one.
    fresh_registry = ProposedFactsRegistry(registry.path.parent / "other_registry.json")
    added2 = propose_and_add(semantic, fresh_registry, [candidate])
    assert added2[0]["entry_id"] == entry_id


def test_no_candidates_qualify_leaves_store_untouched(semantic_path, registry_path):
    semantic = SemanticStore(semantic_path)
    registry = ProposedFactsRegistry(registry_path)
    added = propose_and_add(semantic, registry, [])
    assert added == []
    assert semantic.entries == []
