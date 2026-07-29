import json
import tempfile
from pathlib import Path

import pytest

from physics_agent.knowledge_graph.graph import KnowledgeGraph
from physics_agent.memory.error_memory import ErrorMemory
from physics_agent.meta_learning.curriculum_signals import weak_areas
from physics_agent.retrieval import SemanticStore
from physics_agent.trace import EpisodicMemory, Trace


@pytest.fixture
def env():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        semantic_path = d / "semantic.json"
        with semantic_path.open("w") as f:
            json.dump(
                [
                    {
                        "id": "grav-001",
                        "statement": "gravitation",
                        "conditions": "x",
                        "confidence": 0.4,
                        "provenance": "seed",
                        "tags": ["gravitation"],
                        "last_validated": 0,
                    },
                    {
                        "id": "orbit-001",
                        "statement": "orbits",
                        "conditions": "x",
                        "confidence": 0.3,
                        "provenance": "seed",
                        "tags": ["gravitation", "orbital-mechanics"],
                        "last_validated": 0,
                    },
                ],
                f,
            )
        store = SemanticStore(semantic_path)
        graph = KnowledgeGraph(d / "edges.json", store)
        graph.add_edge("orbit-001", relation="derives_from", condition="x", target="grav-001")

        episodic = EpisodicMemory(d / "episodic.jsonl")
        error_memory = ErrorMemory(d / "error.json")

        yield episodic, error_memory, graph


def test_empty_everything_returns_empty(env):
    episodic, error_memory, _unused_graph = env
    # use a separate graph with no low-confidence nodes, since the shared
    # fixture's graph deliberately has some (for other tests)
    with tempfile.TemporaryDirectory() as d2:
        semantic_path = Path(d2) / "semantic.json"
        with semantic_path.open("w") as f:
            json.dump([], f)
        store = SemanticStore(semantic_path)
        graph = KnowledgeGraph(Path(d2) / "edges.json", store)
        result = weak_areas(episodic, error_memory, graph)
        assert result == []


def test_error_memory_signal_included(env):
    episodic, error_memory, graph = env
    error_memory.record("algebra_error", ["energy"], "root cause", "rederive_math", resolved=True)

    result = weak_areas(episodic, error_memory, graph)
    sources = [s["source"] for s in result]
    assert "error_memory" in sources


def test_unresolved_traces_signal_included(env):
    episodic, error_memory, graph = env
    t = Trace.new("x")
    t.domain_tags = ["thermodynamics"]
    t.resolution_status = "unresolved_max_revisions"
    episodic.write(t)

    result = weak_areas(episodic, error_memory, graph)
    episodic_signals = [s for s in result if s["source"] == "episodic_memory"]
    assert len(episodic_signals) == 1
    assert episodic_signals[0]["domain_tags"] == ["thermodynamics"]


def test_knowledge_graph_cluster_signal_included_with_resolved_tags(env):
    episodic, error_memory, graph = env
    result = weak_areas(episodic, error_memory, graph)
    kg_signals = [s for s in result if s["source"] == "knowledge_graph"]
    assert len(kg_signals) == 1
    assert set(kg_signals[0]["domain_tags"]) == {"gravitation", "orbital-mechanics"}


def test_results_ranked_by_weight_descending(env):
    episodic, error_memory, graph = env
    error_memory.record("algebra_error", ["energy"], "root cause", "rederive_math", resolved=True)
    error_memory.record("algebra_error", ["energy"], "root cause", "rederive_math", resolved=True)
    error_memory.record("algebra_error", ["energy"], "root cause", "rederive_math", resolved=True)

    t = Trace.new("x")
    t.domain_tags = ["thermodynamics"]
    t.resolution_status = "unresolved_max_revisions"
    episodic.write(t)

    result = weak_areas(episodic, error_memory, graph, limit=10)
    weights = [s["weight"] for s in result]
    assert weights == sorted(weights, reverse=True)


def test_respects_limit(env):
    episodic, error_memory, graph = env
    for i in range(5):
        error_memory.record(f"error_type_{i}", ["energy"], "x", "strategy", resolved=True)

    result = weak_areas(episodic, error_memory, graph, limit=2)
    assert len(result) == 2
