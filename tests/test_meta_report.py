import json
import tempfile
from pathlib import Path

import pytest

from physics_agent.knowledge_graph.graph import KnowledgeGraph
from physics_agent.memory.error_memory import ErrorMemory
from physics_agent.memory.procedural import ProceduralMemory
from physics_agent.meta_learning.report import build_report
from physics_agent.retrieval import SemanticStore
from physics_agent.trace import EpisodicMemory, Trace


@pytest.fixture
def env():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        semantic_path = d / "semantic.json"
        with semantic_path.open("w") as f:
            json.dump([], f)
        store = SemanticStore(semantic_path)
        graph = KnowledgeGraph(d / "edges.json", store)
        episodic = EpisodicMemory(d / "episodic.jsonl")
        procedural = ProceduralMemory(d / "procedural.json")
        error_memory = ErrorMemory(d / "error.json")
        yield episodic, procedural, error_memory, graph


def test_report_has_expected_top_level_keys(env):
    episodic, procedural, error_memory, graph = env
    report = build_report(episodic, procedural, error_memory, graph)
    assert set(report.keys()) == {"n_traces", "check_value", "declining_strategies", "weak_areas"}


def test_report_reflects_written_data(env):
    episodic, procedural, error_memory, graph = env
    t = Trace.new("x")
    t.checks_run = ["logic", "physics", "math", "confidence"]
    t.checks_failed = ["math"]
    episodic.write(t)

    for resolved in [False, False, False, False, True]:
        procedural.record_outcome(["energy"], "algebra_error", "rederive_math", resolved=resolved)

    report = build_report(episodic, procedural, error_memory, graph)
    assert report["n_traces"] == 1
    assert report["check_value"]["math"]["n_ever_failed"] == 1
    assert len(report["declining_strategies"]) == 1
