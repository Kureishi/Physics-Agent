import pytest

from physics_agent.llm_client import MockLLMClient
from physics_agent.planner import TaskPlanner, DOMAIN_TAXONOMY


def test_decompose_happy_path():
    llm = MockLLMClient()
    planner = TaskPlanner(llm)
    result = planner.decompose("A 2 kg block slides down a frictionless incline.")

    assert isinstance(result["domain_tags"], list)
    assert isinstance(result["subtasks"], list)
    assert len(result["subtasks"]) > 0
    assert result["planning_time_ms"] >= 0
    for tag in result["domain_tags"]:
        assert tag in DOMAIN_TAXONOMY


def test_decompose_strips_markdown_fences():
    canned = {
        "fenced problem": (
            "```json\n"
            '{"domain_tags": ["energy"], "subtasks": ["a", "b", "c"]}\n'
            "```"
        )
    }
    llm = MockLLMClient(canned_responses=canned)
    planner = TaskPlanner(llm)
    result = planner.decompose("fenced problem test case")

    assert result["domain_tags"] == ["energy"]
    assert result["subtasks"] == ["a", "b", "c"]


def test_decompose_recovers_from_bad_json_via_retry():
    # First call (matching "flaky problem") returns garbage; the planner's
    # retry message is appended to `messages`, but MockLLMClient only looks
    # at messages[-1]["content"] each call, so we simulate "the second
    # attempt succeeds" by having the *retry prompt text* itself route to a
    # good canned response.
    canned = {
        "not valid": (
            '{"domain_tags": ["momentum"], "subtasks": ["step1", "step2"]}'
        ),
        "flaky problem": "This is not JSON at all, sorry!",
    }
    llm = MockLLMClient(canned_responses=canned)
    planner = TaskPlanner(llm, max_retries=1)
    result = planner.decompose("flaky problem")

    assert result["domain_tags"] == ["momentum"]
    assert len(llm.calls) == 2  # confirms a retry actually happened


def test_decompose_raises_after_exhausting_retries():
    # Match both the original prompt AND the planner's corrective retry
    # message (which contains "valid JSON"), so every attempt fails,
    # exercising the exhausted-retries path rather than falling through to
    # MockLLMClient's valid-JSON default response.
    canned = {
        "unfixable": "still not json",
        "valid JSON": "still not json either",
    }
    llm = MockLLMClient(canned_responses=canned)
    planner = TaskPlanner(llm, max_retries=1)
    with pytest.raises(ValueError):
        planner.decompose("unfixable")


def test_unknown_domain_tags_are_filtered_out():
    canned = {
        "weird tag": '{"domain_tags": ["astrology", "energy"], "subtasks": ["a"]}'
    }
    llm = MockLLMClient(canned_responses=canned)
    planner = TaskPlanner(llm)
    result = planner.decompose("weird tag problem")
    assert result["domain_tags"] == ["energy"]
