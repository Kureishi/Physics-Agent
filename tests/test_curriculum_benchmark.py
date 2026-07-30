from physics_agent.curriculum.benchmark import summarize


def _entry(source, before, after):
    return {
        "targeted_signal": {"source": source},
        "metric_before": before,
        "metric_after": after,
    }


def test_empty_entries_returns_empty():
    assert summarize([]) == {}


def test_error_memory_fewer_recurrences_is_improved():
    result = summarize([_entry("error_memory", 5.0, 3.0)])
    assert result["error_memory"]["n_improved"] == 1


def test_error_memory_more_recurrences_is_regressed():
    result = summarize([_entry("error_memory", 3.0, 5.0)])
    assert result["error_memory"]["n_regressed"] == 1


def test_episodic_memory_fewer_unresolved_is_improved():
    result = summarize([_entry("episodic_memory", 4.0, 2.0)])
    assert result["episodic_memory"]["n_improved"] == 1


def test_knowledge_graph_higher_confidence_is_improved():
    result = summarize([_entry("knowledge_graph", 0.4, 0.6)])
    assert result["knowledge_graph"]["n_improved"] == 1


def test_knowledge_graph_lower_confidence_is_regressed():
    result = summarize([_entry("knowledge_graph", 0.6, 0.4)])
    assert result["knowledge_graph"]["n_regressed"] == 1


def test_unchanged_metric_counted_separately_from_improved_or_regressed():
    result = summarize([_entry("error_memory", 3.0, 3.0)])
    assert result["error_memory"]["n_unchanged"] == 1
    assert result["error_memory"]["n_improved"] == 0
    assert result["error_memory"]["n_regressed"] == 0


def test_unmeasurable_when_either_side_is_none():
    result = summarize([_entry("knowledge_graph", None, 0.5)])
    assert result["knowledge_graph"]["n_unmeasurable"] == 1


def test_aggregates_across_multiple_rounds_and_sources():
    entries = [
        _entry("error_memory", 5.0, 3.0),  # improved
        _entry("error_memory", 3.0, 3.0),  # unchanged
        _entry("knowledge_graph", 0.3, 0.2),  # regressed
    ]
    result = summarize(entries)
    assert result["error_memory"]["n_rounds"] == 2
    assert result["error_memory"]["n_improved"] == 1
    assert result["error_memory"]["n_unchanged"] == 1
    assert result["knowledge_graph"]["n_rounds"] == 1
    assert result["knowledge_graph"]["n_regressed"] == 1
