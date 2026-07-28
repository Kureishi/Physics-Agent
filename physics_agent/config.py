"""
Configuration for the physics agent.

All values can be overridden with environment variables so the same code
runs unchanged across dev machines / LM Studio instances. Defaults match
LM Studio's default local server settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    # LM Studio exposes an OpenAI-compatible server. Defaults match LM
    # Studio's out-of-the-box local server settings (Developer tab -> Start
    # Server). The api_key is unused by LM Studio but the OpenAI client
    # requires some non-empty value.
    lm_studio_base_url: str = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    lm_studio_api_key: str = os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
    # Set this to whatever model identifier LM Studio shows for the model
    # you've loaded (visible in the LM Studio server logs / UI). LM Studio
    # will generally accept any string here and route to the loaded model,
    # but setting it correctly avoids ambiguity if you have multiple models
    # loaded.
    lm_studio_model: str = os.environ.get("LM_STUDIO_MODEL", "local-model")

    episodic_memory_path: str = os.environ.get("EPISODIC_MEMORY_PATH", "memory/episodic.jsonl")
    semantic_store_path: str = os.environ.get("SEMANTIC_STORE_PATH", "data/semantic_seed.json")
    procedural_memory_path: str = os.environ.get("PROCEDURAL_MEMORY_PATH", "memory/procedural.json")
    error_memory_path: str = os.environ.get("ERROR_MEMORY_PATH", "memory/error_memory.json")
    knowledge_graph_path: str = os.environ.get("KNOWLEDGE_GRAPH_PATH", "data/knowledge_graph_edges.json")

    # Safety rail for the Stage 4 self-correction loop: stop retrying after
    # this many revision attempts and ship the best-effort answer marked
    # "unresolved_max_revisions" rather than looping indefinitely.
    max_revisions: int = int(os.environ.get("MAX_REVISIONS", "3"))
