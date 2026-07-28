from physics_agent.llm_client import MockLLMClient
from physics_agent.self_eval.pipeline import SelfEvaluationPipeline
from physics_agent.trace import Trace


def _make_trace():
    trace = Trace.new("A 2 kg block starts at rest at the top of a 5 m frictionless incline.")
    trace.subtasks = ["identify knowns", "apply conservation of energy", "solve for v"]
    trace.initial_solution = "Using energy conservation, v = sqrt(2*g*h) ~= 9.9 m/s."
    trace.retrieved_knowledge = [
        {"statement": "Conservation of mechanical energy", "conditions": "no friction"}
    ]
    return trace


def test_pipeline_runs_all_four_checks_with_defaults():
    llm = MockLLMClient()
    pipeline = SelfEvaluationPipeline(llm)
    trace = _make_trace()

    pipeline.run(trace)

    assert trace.checks_run == ["logic", "physics", "math", "confidence"]
    assert trace.checks_failed == []  # all defaults pass
    assert len(trace.check_details) == 4
    assert trace.final_confidence == 0.85


def test_pipeline_records_failed_checks():
    canned = {"logic-check component": '{"passed": false, "issues": ["contradiction"]}'}
    llm = MockLLMClient(canned_responses=canned)
    pipeline = SelfEvaluationPipeline(llm)
    trace = _make_trace()

    pipeline.run(trace)

    assert "logic" in trace.checks_failed
    assert "physics" not in trace.checks_failed
    logic_detail = next(d for d in trace.check_details if d["check"] == "logic")
    assert logic_detail["passed"] is False


def test_pipeline_survives_a_check_that_raises():
    class ExplodingCheck:
        name = "exploding"

        def run(self, trace):
            raise RuntimeError("boom")

    pipeline = SelfEvaluationPipeline(checks=[ExplodingCheck()])
    trace = _make_trace()

    pipeline.run(trace)  # must not raise

    assert trace.checks_run == ["exploding"]
    assert trace.checks_failed == ["exploding"]
    assert "boom" in trace.check_details[0]["details"]


def test_pipeline_confidence_check_sees_prior_failures():
    # Confidence check receives checks_run/checks_failed as context; verify
    # a failing physics check is visible to it by the time it runs (order
    # dependency: confidence runs last).
    canned = {"physics-check component": '{"passed": false, "issues": ["bad units"]}'}
    llm = MockLLMClient(canned_responses=canned)
    pipeline = SelfEvaluationPipeline(llm)
    trace = _make_trace()

    pipeline.run(trace)

    assert trace.checks_run[-1] == "confidence"
    assert "physics" in trace.checks_failed


def test_pipeline_threads_knowledge_graph_into_physics_check(tmp_path):
    import json as _json

    from physics_agent.knowledge_graph.graph import KnowledgeGraph
    from physics_agent.retrieval import SemanticStore

    semantic_path = tmp_path / "semantic.json"
    with semantic_path.open("w") as f:
        _json.dump(
            [
                {
                    "id": "eng-001",
                    "statement": "KE = 0.5*m*v^2",
                    "conditions": "Non-relativistic",
                    "confidence": 0.99,
                    "provenance": "seed",
                    "tags": ["energy"],
                    "last_validated": 0,
                }
            ],
            f,
        )
    store = SemanticStore(semantic_path)
    graph = KnowledgeGraph(tmp_path / "edges.json", store)
    graph.add_edge("eng-001", relation="requires_assumption", condition="non_relativistic")

    llm = MockLLMClient()  # LLM critique defaults to passing
    pipeline = SelfEvaluationPipeline(llm, knowledge_graph=graph)

    trace = _make_trace()
    trace.domain_tags = ["special-relativity"]
    trace.retrieved_knowledge = [{"id": "eng-001", "statement": "KE = 0.5*m*v^2", "conditions": "Non-relativistic"}]

    pipeline.run(trace)

    assert "physics" in trace.checks_failed  # caught via the knowledge graph, not the LLM
