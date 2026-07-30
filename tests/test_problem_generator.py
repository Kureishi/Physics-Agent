import pytest

from physics_agent.curriculum.problem_generator import ProblemGenerator
from physics_agent.llm_client import MockLLMClient
from physics_agent.tools.literature import LiteratureSearchTool

FAKE_ARXIV_RESPONSE = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title>A Study of Damped Oscillators</title>
    <summary>This paper studies damping in oscillatory systems under various conditions relevant to the topic at hand and presents a broad framework applicable across several regimes.</summary>
    <author><name>Jane Researcher</name></author>
  </entry>
</feed>
"""


def _signal(source="error_memory", domain_tags=None, error_type="algebra_error"):
    return {
        "source": source,
        "domain_tags": domain_tags or ["energy"],
        "error_type": error_type,
        "reason": f"error type '{error_type}' recurred 5x",
        "weight": 5,
    }


def test_generate_returns_expected_shape_with_default_mock():
    llm = MockLLMClient()
    generator = ProblemGenerator(llm)
    result = generator.generate(_signal())

    assert result["problem_text"]
    assert isinstance(result["target_concepts"], list)
    assert result["literature_context"] is None  # no literature_tool provided


def test_generate_without_literature_tool_has_no_context():
    llm = MockLLMClient()
    generator = ProblemGenerator(llm, literature_tool=None)
    result = generator.generate(_signal())
    assert result["literature_context"] is None


def test_generate_with_literature_tool_includes_context():
    llm = MockLLMClient()
    literature_tool = LiteratureSearchTool(fetch_fn=lambda url: FAKE_ARXIV_RESPONSE)
    generator = ProblemGenerator(llm, literature_tool=literature_tool)

    result = generator.generate(_signal())

    assert result["literature_context"] is not None
    assert "Damped Oscillators" in result["literature_context"]


def test_generate_literature_failure_falls_back_to_none_context():
    llm = MockLLMClient()

    def failing_fetch(url):
        raise ConnectionError("no network")

    literature_tool = LiteratureSearchTool(fetch_fn=failing_fetch)
    generator = ProblemGenerator(llm, literature_tool=literature_tool)

    result = generator.generate(_signal())
    assert result["literature_context"] is None  # failure handled gracefully, not raised


def test_generate_recovers_via_retry_on_bad_json():
    canned = {
        "not valid JSON": (
            '{"problem_text": "A ball falls.", "target_concepts": ["kinematics"], "rationale": "x"}'
        ),
        "flaky curriculum problem": "not json at all",
    }
    llm = MockLLMClient(canned_responses=canned)
    generator = ProblemGenerator(llm, max_retries=1)

    signal = _signal()
    signal["reason"] = "flaky curriculum problem"
    result = generator.generate(signal)

    assert result["problem_text"] == "A ball falls."
    assert len(llm.calls) == 2


def test_generate_raises_after_exhausting_retries():
    canned = {
        "unfixable curriculum": "still not json",
        "valid JSON": "still not json either",
    }
    llm = MockLLMClient(canned_responses=canned)
    generator = ProblemGenerator(llm, max_retries=1)

    signal = _signal()
    signal["reason"] = "unfixable curriculum problem"
    with pytest.raises(ValueError):
        generator.generate(signal)


def test_generate_raises_when_problem_text_missing():
    canned = {"missing text signal": '{"target_concepts": ["x"], "rationale": "y"}'}
    llm = MockLLMClient(canned_responses=canned)
    generator = ProblemGenerator(llm, max_retries=0)

    signal = _signal()
    signal["reason"] = "missing text signal"
    with pytest.raises(ValueError):
        generator.generate(signal)
