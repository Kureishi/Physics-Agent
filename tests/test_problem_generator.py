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


def test_generate_includes_avoid_list_in_prompt_when_given():
    llm = MockLLMClient()
    generator = ProblemGenerator(llm)

    generator.generate(_signal(), avoid=["A ball is dropped from a height.", "A car accelerates."])

    user_content = llm.calls[0][1]["content"]
    assert "avoid_duplicating" in user_content
    assert "A ball is dropped" in user_content


def test_generate_omits_avoid_key_when_not_given():
    llm = MockLLMClient()
    generator = ProblemGenerator(llm)

    generator.generate(_signal())

    user_content = llm.calls[0][1]["content"]
    assert "avoid_duplicating" not in user_content


def test_generate_truncates_long_avoid_entries():
    llm = MockLLMClient()
    generator = ProblemGenerator(llm)

    long_text = "A" * 500
    generator.generate(_signal(), avoid=[long_text])

    user_content = llm.calls[0][1]["content"]
    assert "A" * 500 not in user_content  # full text shouldn't appear
    assert "A" * 200 in user_content  # truncated prefix should


# -- resilience to the LLM call itself failing (not just bad JSON) ----------


class _FlakyLLM:
    """Simulates a real API-level failure (e.g. openai.BadRequestError from
    a local inference engine returning a raw server error), succeeding on
    a later attempt -- reproducing a real crash seen running
    generate_problem_set_cli.py against LM Studio."""

    def __init__(self, fail_times: int, good_response: str):
        self.fail_times = fail_times
        self.good_response = good_response
        self.calls = 0

    def chat(self, messages, temperature=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': 'The model produced "
                "output that does not match the expected format'}}"
            )
        return self.good_response


def test_generate_retries_past_a_raw_llm_call_exception():
    llm = _FlakyLLM(
        fail_times=1,
        good_response='{"problem_text": "A ball falls.", "target_concepts": ["kinematics"], "rationale": "x"}',
    )
    generator = ProblemGenerator(llm, max_retries=1)

    result = generator.generate(_signal())

    assert result["problem_text"] == "A ball falls."
    assert llm.calls == 2  # first call raised, second succeeded


def test_generate_raises_clean_valueerror_after_exhausting_retries_on_call_failures():
    llm = _FlakyLLM(fail_times=99, good_response="irrelevant")  # always fails
    generator = ProblemGenerator(llm, max_retries=1)

    # Should raise a plain ValueError (catchable by generate_for_domains'
    # existing `except ValueError`), never the original RuntimeError/API
    # exception -- that's what lets a batch script skip this one signal
    # and keep going instead of crashing entirely.
    with pytest.raises(ValueError) as exc_info:
        generator.generate(_signal())
    assert not isinstance(exc_info.value, RuntimeError)
    assert llm.calls == 2  # initial attempt + 1 retry, then gave up


def test_generate_mixed_failure_then_bad_json_then_success():
    # A more realistic sequence: the call fails outright once, then
    # succeeds but with bad JSON, then succeeds properly -- exercises both
    # failure-handling branches within a single generate() call.
    class MixedLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, temperature=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient server error")
            if self.calls == 2:
                return "not valid json"
            return '{"problem_text": "A ball falls.", "target_concepts": [], "rationale": "x"}'

    llm = MixedLLM()
    generator = ProblemGenerator(llm, max_retries=2)

    result = generator.generate(_signal())

    assert result["problem_text"] == "A ball falls."
    assert llm.calls == 3
