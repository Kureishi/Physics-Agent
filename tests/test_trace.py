import tempfile
from pathlib import Path

from physics_agent.trace import Trace, EpisodicMemory, ToolCall


def test_trace_roundtrip():
    trace = Trace.new("A ball is thrown upward at 10 m/s.")
    trace.domain_tags = ["kinematics"]
    trace.subtasks = ["identify knowns", "apply kinematics equation"]
    trace.tool_calls = [ToolCall(tool="sim", input="v0=10", output="t_max=1.02s", latency_ms=12.3)]

    d = trace.to_dict()
    restored = Trace.from_dict(d)

    assert restored.problem_id == trace.problem_id
    assert restored.domain_tags == ["kinematics"]
    assert restored.tool_calls[0].tool == "sim"
    assert restored.tool_calls[0].latency_ms == 12.3


def test_episodic_memory_write_and_read():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "episodic.jsonl"
        memory = EpisodicMemory(path)
        assert len(memory) == 0

        t1 = Trace.new("Problem 1")
        t2 = Trace.new("Problem 2")
        memory.write(t1)
        memory.write(t2)

        assert len(memory) == 2
        loaded = memory.read_all()
        assert {t.problem_text for t in loaded} == {"Problem 1", "Problem 2"}


def test_episodic_memory_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "nested" / "dir" / "episodic.jsonl"
        memory = EpisodicMemory(path)
        memory.write(Trace.new("x"))
        assert path.exists()
        assert len(memory) == 1


def test_from_dict_ignores_unknown_fields():
    d = Trace.new("x").to_dict()
    d["some_future_field"] = "value that doesn't exist yet"
    restored = Trace.from_dict(d)
    assert restored.problem_text == "x"
