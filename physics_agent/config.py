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

    # Both exist because of a real failure mode: a "thinking"/reasoning
    # model (or one that simply never emits a stop token in a given
    # quantization) can appear to hang forever with a non-streaming client,
    # since nothing returns until the whole response is done. See
    # llm_client.LLMClient's docstring for the full reasoning. If you're
    # using a model that legitimately needs long chain-of-thought before it
    # answers, raise both of these -- the trade-off is longer waits, not
    # reliability.
    lm_studio_timeout_seconds: float = float(os.environ.get("LM_STUDIO_TIMEOUT_SECONDS", "120"))
    lm_studio_max_tokens: int = int(os.environ.get("LM_STUDIO_MAX_TOKENS", "2048"))

    episodic_memory_path: str = os.environ.get("EPISODIC_MEMORY_PATH", "memory/episodic.jsonl")
    semantic_store_path: str = os.environ.get("SEMANTIC_STORE_PATH", "data/semantic_seed.json")
    procedural_memory_path: str = os.environ.get("PROCEDURAL_MEMORY_PATH", "memory/procedural.json")
    error_memory_path: str = os.environ.get("ERROR_MEMORY_PATH", "memory/error_memory.json")
    knowledge_graph_path: str = os.environ.get("KNOWLEDGE_GRAPH_PATH", "data/knowledge_graph_edges.json")
    curriculum_log_path: str = os.environ.get("CURRICULUM_LOG_PATH", "memory/curriculum_log.jsonl")

    # Ground-truth canary problems (Safety Rails): a small, fixed,
    # human-verified problem set (see data/canary_problems.json) solved
    # through the full pipeline and graded against known-correct answers,
    # independent of the pipeline's own self-consistency. canary_log_path
    # accumulates one entry per canary per run, so drift in how often
    # checks disagree with ground truth is visible over time.
    canary_problems_path: str = os.environ.get("CANARY_PROBLEMS_PATH", "data/canary_problems.json")
    canary_log_path: str = os.environ.get("CANARY_LOG_PATH", "memory/canary_log.jsonl")

    # Safety rail for the Stage 4 self-correction loop: stop retrying after
    # this many revision attempts and ship the best-effort answer marked
    # "unresolved_max_revisions" rather than looping indefinitely.
    max_revisions: int = int(os.environ.get("MAX_REVISIONS", "3"))

    # Scheduling/Decision Loop: the background process that decides when to
    # solve, review, and practice, instead of a person typing each command
    # by hand. See physics_agent/scheduler/scheduler.py for the full design
    # note; briefly: scheduler_queue_path is a plain problem-set-shaped JSON
    # file (same schema as data/problem_sets/*.json) the scheduler consumes
    # from and generate_problem_set_cli.py can refill; scheduler_state_path
    # persists cadence counters across restarts; scheduler_log_path is the
    # append-only decision log (one entry per action taken, with why).
    scheduler_queue_path: str = os.environ.get("SCHEDULER_QUEUE_PATH", "data/problem_sets/scheduler_queue.json")
    scheduler_state_path: str = os.environ.get("SCHEDULER_STATE_PATH", "memory/scheduler_state.json")
    scheduler_log_path: str = os.environ.get("SCHEDULER_LOG_PATH", "memory/scheduler_log.jsonl")

    # Run a meta-learning review (physics_agent.meta_learning.report.build_report)
    # after this many problems have been solved since the last one.
    scheduler_review_every_n_solves: int = int(os.environ.get("SCHEDULER_REVIEW_EVERY_N_SOLVES", "20"))

    # A curriculum round is only triggered when the top weak_areas() signal's
    # weight is at least this high -- the same "don't act on a one-off"
    # philosophy as escalation.py's DEFAULT_MIN_RECURRING_UNRESOLVED.
    scheduler_curriculum_weight_threshold: float = float(
        os.environ.get("SCHEDULER_CURRICULUM_WEIGHT_THRESHOLD", "5")
    )
    # Even if the threshold above is crossed every single cycle (a
    # persistently weak domain), don't run a curriculum round more often
    # than once every this many cycles -- prevents one standing weakness
    # from monopolizing every cycle's practice slot before its own most
    # recent round has even been measured.
    scheduler_curriculum_min_cycles_between_rounds: int = int(
        os.environ.get("SCHEDULER_CURRICULUM_MIN_CYCLES_BETWEEN_ROUNDS", "5")
    )
    scheduler_curriculum_n_problems: int = int(os.environ.get("SCHEDULER_CURRICULUM_N_PROBLEMS", "1"))
