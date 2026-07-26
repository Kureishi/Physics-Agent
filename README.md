# Physics Agent — Stage 1: Task Planner + Retrieval

This is Stage 1 of the roadmap: **task planner + retrieval**, plus the
**trace schema** that every later stage (tool orchestration, multi-agent
critique, structured memory, self-correction, meta-learning) will read and
write. See `physics_agent/trace.py` for the full schema and a field-by-field
note on which stage owns which field.

## What this does right now

Given a raw physics problem, the pipeline:
1. **Classifies** it into 1-3 domain tags from a fixed physics taxonomy.
2. **Decomposes** it into an ordered list of subtasks.
3. **Retrieves** relevant formulas/concepts from a seeded semantic-memory store.
4. **Writes a trace** of all of the above to append-only episodic memory (JSONL).

It does *not* yet solve the problem, run tool orchestration, verify an
answer, or self-correct — those are Stages 2, 2, 3, and 5 respectively.
This stage's job is to produce reliable decomposition + retrieval, and to
start populating the trace log those later stages depend on.

## Setup

```bash
pip install -r requirements.txt
```

### Running against LM Studio

1. Open LM Studio, load a model, go to the **Developer** tab, and click
   **Start Server**. By default it serves an OpenAI-compatible API at
   `http://localhost:1234/v1`.
2. Run:

```bash
python -m physics_agent.cli "A 2 kg block slides down a frictionless 30 degree incline of length 4 m. Find its speed at the bottom."
```

If your LM Studio server runs on a different host/port, or you want to name
a specific loaded model, set environment variables before running:

```bash
export LM_STUDIO_BASE_URL="http://localhost:1234/v1"
export LM_STUDIO_MODEL="your-model-name"
python -m physics_agent.cli "..."
```

### Running without LM Studio (offline / dry run)

```bash
python -m physics_agent.cli --dry-run "A 2 kg block slides down a frictionless incline..."
```

`--dry-run` swaps in `MockLLMClient`, which returns canned, schema-valid
JSON so you can exercise the whole pipeline (parsing, retrieval, trace
writing) without a model running at all. This is also what the test suite
uses — the tests never require LM Studio to be running.

## Project layout

```
physics_agent/
  config.py        LM Studio connection settings + file paths (env-overridable)
  llm_client.py     LLMClient (real, OpenAI-compatible) + MockLLMClient (offline/tests)
  planner.py        TaskPlanner: domain classification + subtask decomposition
  retrieval.py      SemanticStore: keyword-scored retrieval over seeded physics facts
  trace.py          Trace schema + EpisodicMemory (JSONL append-only store)
  cli.py            Entry point wiring the above together
data/
  semantic_seed.json  Seed knowledge base (~13 core physics formulas across domains)
tests/
  test_trace.py       Trace roundtrip + episodic memory read/write
  test_planner.py      Decomposition, JSON-parsing robustness, retry/failure paths
  test_retrieval.py    Keyword scoring, domain-tag bonus, persistence
memory/
  episodic.jsonl      Created at runtime — one JSON line per problem run
```

## Running the tests

```bash
pytest tests/ -v
```

All 14 tests run offline (no LM Studio required) using `MockLLMClient`.

## Design notes carried over from the spec

- **Domain taxonomy is fixed** (`planner.DOMAIN_TAXONOMY`). Tags outside it
  are silently dropped rather than polluting downstream memory/knowledge-graph
  keys with one-off strings a model happened to invent.
- **JSON parsing from the planner is defensive**: local models served
  through LM Studio don't always respect "output only JSON" — the parser
  strips markdown fences and extracts the first `{...}` block, and the
  planner retries once with a corrective follow-up before failing loudly.
  It fails loudly (raises `ValueError`) rather than silently returning an
  empty plan, since a silently-empty plan would poison the trace log.
- **Retrieval is keyword-based, not embeddings-based**, on purpose — this
  keeps Stage 1 dependency-light. `SemanticStore.retrieve` is the seam to
  swap in an embedding-based version later without touching planner.py or
  cli.py.
- **The trace schema already has empty placeholders for Stage 2-5 fields**
  (`tool_calls`, `checks_run`, `checks_failed`, `error_type`,
  `revision_count`, `final_*`). This is deliberate, per the earlier design
  discussion: retrofitting the schema after those stages exist would mean
  losing structured data for every problem solved before the retrofit.

## Next steps (Stage 2)

Tool orchestration: wire in a symbolic math tool (e.g. SymPy), a numerical
simulation tool, and a literature-retrieval tool, each populating
`trace.tool_calls` as they're invoked from the planner's subtask list.
