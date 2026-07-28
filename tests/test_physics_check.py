import json

from physics_agent.llm_client import MockLLMClient
from physics_agent.self_eval.physics_check import PhysicsCheck
from physics_agent.trace import Trace, ToolCall


def _make_trace(tool_calls=None):
    trace = Trace.new("A 2 kg block starts at rest atop a 5 m frictionless incline.")
    trace.retrieved_knowledge = [
        {
            "statement": "KE = 0.5*m*v^2",
            "conditions": "Non-relativistic",
        }
    ]
    trace.initial_solution = "v = sqrt(2*g*h) = 9.9 m/s"
    trace.tool_calls = tool_calls or []
    return trace


def _symbolic_call(value):
    output = {
        "expression": "Eq(m*g*h, 0.5*m*v**2)",
        "solve_for": "v",
        "solutions_numeric": [value],
    }
    return ToolCall(tool="symbolic_math", input="{}", output=json.dumps(output), latency_ms=1.0)


def _simulation_call(final_state):
    output = {"final_state": final_state}
    return ToolCall(tool="simulation", input="{}", output=json.dumps(output), latency_ms=1.0)


def test_physics_check_passes_when_no_tools_to_cross_check():
    llm = MockLLMClient()  # default physics response passes
    check = PhysicsCheck(llm)
    result = check.run(_make_trace())
    assert result["passed"] is True


def test_physics_check_passes_when_symbolic_and_simulation_agree():
    llm = MockLLMClient()
    check = PhysicsCheck(llm)
    tool_calls = [_symbolic_call(9.9), _simulation_call({"v": 9.85, "x": 5.0})]
    result = check.run(_make_trace(tool_calls))
    assert result["passed"] is True
    assert "agrees" in result["details"]


def test_physics_check_fails_when_symbolic_and_simulation_disagree():
    llm = MockLLMClient()
    check = PhysicsCheck(llm)
    tool_calls = [_symbolic_call(9.9), _simulation_call({"v": 3.0})]
    result = check.run(_make_trace(tool_calls))
    assert result["passed"] is False
    assert "disagrees" in result["details"]


def test_physics_check_fails_when_llm_critique_fails_even_if_tools_agree():
    canned = {"physics-check component": '{"passed": false, "issues": ["formula used outside validity range"]}'}
    llm = MockLLMClient(canned_responses=canned)
    check = PhysicsCheck(llm)
    tool_calls = [_symbolic_call(9.9), _simulation_call({"v": 9.9})]
    result = check.run(_make_trace(tool_calls))
    assert result["passed"] is False
    assert "validity range" in result["details"]


def test_physics_check_ignores_errored_tool_calls_for_cross_check():
    llm = MockLLMClient()
    check = PhysicsCheck(llm)
    errored = ToolCall(tool="symbolic_math", input="{}", output=json.dumps({"error": "bad input"}), latency_ms=1.0)
    result = check.run(_make_trace([errored]))
    # no valid symbolic result -> cross-check doesn't apply -> falls back to LLM critique (passes by default)
    assert result["passed"] is True


# -- Stage 6: knowledge graph integration ---------------------------------------


def _make_knowledge_graph(tmp_path, edges=()):
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
    for edge in edges:
        graph.add_edge(**edge)
    return graph


def test_physics_check_no_knowledge_graph_skips_that_subcheck():
    llm = MockLLMClient()
    check = PhysicsCheck(llm, knowledge_graph=None)
    result = check.run(_make_trace())
    assert result["passed"] is True  # unaffected -- same as pre-Stage-6 behavior


def test_physics_check_flags_knowledge_graph_violation(tmp_path):
    graph = _make_knowledge_graph(
        tmp_path,
        edges=[{"source": "eng-001", "relation": "requires_assumption", "condition": "non_relativistic"}],
    )
    llm = MockLLMClient()  # LLM critique defaults to passing
    check = PhysicsCheck(llm, knowledge_graph=graph)

    trace = _make_trace()
    trace.domain_tags = ["special-relativity", "energy"]
    trace.retrieved_knowledge = [{"id": "eng-001", "statement": "KE = 0.5*m*v^2", "conditions": "Non-relativistic"}]

    result = check.run(trace)
    assert result["passed"] is False
    assert "eng-001" in result["details"]
    assert "non_relativistic" in result["details"]


def test_physics_check_knowledge_graph_passes_when_no_conflict(tmp_path):
    graph = _make_knowledge_graph(
        tmp_path,
        edges=[{"source": "eng-001", "relation": "requires_assumption", "condition": "non_relativistic"}],
    )
    llm = MockLLMClient()
    check = PhysicsCheck(llm, knowledge_graph=graph)

    trace = _make_trace()
    trace.domain_tags = ["dynamics", "energy"]  # no relativity conflict
    trace.retrieved_knowledge = [{"id": "eng-001", "statement": "KE = 0.5*m*v^2", "conditions": "Non-relativistic"}]

    result = check.run(trace)
    assert result["passed"] is True


def test_physics_check_knowledge_graph_violation_fails_even_if_other_subchecks_pass(tmp_path):
    graph = _make_knowledge_graph(
        tmp_path,
        edges=[{"source": "eng-001", "relation": "requires_assumption", "condition": "non_relativistic"}],
    )
    llm = MockLLMClient()  # LLM critique passes, no tool calls to disagree
    check = PhysicsCheck(llm, knowledge_graph=graph)

    trace = _make_trace()
    trace.domain_tags = ["special-relativity"]
    trace.retrieved_knowledge = [{"id": "eng-001", "statement": "KE = 0.5*m*v^2", "conditions": "Non-relativistic"}]
    trace.tool_calls = []  # cross-tool check doesn't apply either

    result = check.run(trace)
    assert result["passed"] is False  # kg violation alone is enough to fail
