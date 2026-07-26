from physics_agent.tools.registry import ToolRegistry


def test_get_returns_known_tool():
    registry = ToolRegistry()
    tool = registry.get("symbolic_math")
    assert tool.name == "symbolic_math"


def test_get_unknown_tool_raises_key_error():
    registry = ToolRegistry()
    try:
        registry.get("not_a_real_tool")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_relevant_tools_for_known_domain():
    registry = ToolRegistry()
    tools = registry.relevant_tools(["kinematics"])
    assert "symbolic_math" in tools
    assert "simulation" in tools
    assert "literature_search" not in tools  # not hinted for kinematics


def test_relevant_tools_unions_across_multiple_tags():
    registry = ToolRegistry()
    tools = registry.relevant_tools(["kinematics", "special-relativity"])
    assert "symbolic_math" in tools
    assert "simulation" in tools  # from kinematics
    assert "literature_search" in tools  # from special-relativity


def test_relevant_tools_falls_back_to_all_when_no_tags_match():
    registry = ToolRegistry()
    tools = registry.relevant_tools(["not-a-real-domain"])
    assert set(tools) == set(registry.names())


def test_relevant_tools_empty_tags_falls_back_to_all():
    registry = ToolRegistry()
    tools = registry.relevant_tools([])
    assert set(tools) == set(registry.names())
