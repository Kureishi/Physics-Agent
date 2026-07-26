import json
import tempfile
from pathlib import Path

import pytest

from physics_agent.retrieval import SemanticStore


SAMPLE_ENTRIES = [
    {
        "id": "eng-001",
        "statement": "KE = 0.5*m*v^2 (kinetic energy)",
        "conditions": "Non-relativistic",
        "confidence": 0.99,
        "provenance": "seed",
        "tags": ["energy", "dynamics"],
        "last_validated": 0,
    },
    {
        "id": "grav-001",
        "statement": "F = G*m1*m2/r^2 (Newton's law of universal gravitation)",
        "conditions": "Point masses",
        "confidence": 0.99,
        "provenance": "seed",
        "tags": ["gravitation"],
        "last_validated": 0,
    },
]


@pytest.fixture
def store_path():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "seed.json"
        with path.open("w") as f:
            json.dump(SAMPLE_ENTRIES, f)
        yield path


def test_missing_seed_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SemanticStore(tmp_path / "does_not_exist.json")


def test_retrieve_by_keyword_overlap(store_path):
    store = SemanticStore(store_path)
    results = store.retrieve("what is the kinetic energy of a moving mass")
    assert len(results) >= 1
    assert results[0]["id"] == "eng-001"


def test_retrieve_uses_domain_tag_bonus(store_path):
    store = SemanticStore(store_path)
    # Query text overlaps weakly with both, but tag bonus should push
    # gravitation to the top when domain_tags say so.
    results = store.retrieve("two objects pulling on each other", domain_tags=["gravitation"])
    assert results[0]["id"] == "grav-001"


def test_retrieve_respects_k(store_path):
    store = SemanticStore(store_path)
    results = store.retrieve("energy gravitation force mass", k=1)
    assert len(results) == 1


def test_add_persists_to_disk(store_path):
    store = SemanticStore(store_path)
    store.add(
        entry_id="mom-999",
        statement="p = m*v test entry",
        conditions="test",
        confidence=0.5,
        provenance="unit-test",
        tags=["momentum"],
    )

    reloaded = SemanticStore(store_path)
    ids = [e["id"] for e in reloaded.entries]
    assert "mom-999" in ids
