# Physics Agent — Stage 1 + Stage 2 + Stage 3

Stage 1: **task planner + retrieval**, plus the **trace schema** that every
later stage reads and writes. Stage 2: **tool orchestration** — symbolic
math, numerical simulation, and literature search, producing an initial
solution. Stage 3: **self-evaluation pipeline** — Logic, Physics, Math, and
Confidence checks that independently critique that initial solution. See
`physics_agent/trace.py` for the full schema and a field-by-field note on
which stage owns which field.

## What this does right now

Given a raw physics problem, the pipeline:
1. **Classifies** it into 1-3 domain tags from a fixed physics taxonomy. *(Stage 1)*
2. **Decomposes** it into an ordered list of subtasks. *(Stage 1)*
3. **Retrieves** relevant formulas/concepts from a seeded semantic-memory store. *(Stage 1)*
4. **Selects tools** relevant to the problem's domain and decides which to call, with what inputs. *(Stage 2)*
5. **Executes** those tool calls — symbolic algebra (SymPy), numerical ODE
   integration (SciPy), or arXiv literature search — capturing every call
   (including failures) into the trace. *(Stage 2)*
6. **Synthesizes an initial solution** from the tool outputs. *(Stage 2)*
7. **Self-evaluates** that solution with four independent checks:
   - **Logic Check** (LLM): is the reasoning internally consistent, does it
     actually address every subtask, no contradictions or unjustified leaps?
   - **Physics Check** (deterministic + LLM): do `symbolic_math` and
     `simulation` results agree with each other, and does an LLM critique
     find dimensional inconsistencies or conservation-law violations?
   - **Math Check** (deterministic): does re-substituting each
     `symbolic_math` solution back into its original equation actually
     satisfy it?
   - **Confidence Check** (LLM): a calibrated 0-1 confidence estimate,
     informed by whether the other three checks passed.
   *(Stage 3)*
8. **Writes a trace** of all of the above to append-only episodic memory (JSONL).

It does *not* yet act on a failed check by revising the solution — that's
Stage 5 (self-correction). This stage's job is to independently verify the
initial solution and produce a structured, trustworthy signal
(`checks_run`, `checks_failed`, `check_details`, `final_confidence`) about
whether it's actually right.

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
  config.py         LM Studio connection settings + file paths (env-overridable)
  llm_client.py      LLMClient (real, OpenAI-compatible) + MockLLMClient (offline/tests)
  json_utils.py      Shared defensive JSON extraction, used across planner/orchestrator/checks
  planner.py         TaskPlanner: domain classification + subtask decomposition        (Stage 1)
  retrieval.py        SemanticStore: keyword-scored retrieval over seeded physics facts  (Stage 1)
  orchestrator.py     ToolOrchestrator: tool selection, execution, solution synthesis   (Stage 2)
  tools/
    registry.py        ToolRegistry + domain-tag -> tool hints ("Physics Tool Selection")
    symbolic_math.py    SymbolicMathTool: SymPy-backed equation solving
    simulation.py        SimulationTool: SciPy-backed numerical ODE integration
    literature.py         LiteratureSearchTool: arXiv search (injectable fetch for tests)
  self_eval/
    pipeline.py          SelfEvaluationPipeline: runs all four checks, never crashes on one
    logic_check.py        LogicCheck: LLM-based internal-consistency review
    physics_check.py       PhysicsCheck: cross-tool agreement + LLM physics critique
    math_check.py           MathCheck: deterministic re-substitution verification
    confidence_check.py     ConfidenceCheck: calibrated confidence, sees prior check results
  trace.py            Trace schema + EpisodicMemory (JSONL append-only store)
  cli.py              Entry point wiring the full Stage 1 + 2 + 3 pipeline together
data/
  semantic_seed.json  Seed knowledge base (~13 core physics formulas across domains)
tests/
  test_trace.py         Trace roundtrip + episodic memory read/write
  test_planner.py         Decomposition, JSON-parsing robustness, retry/failure paths
  test_retrieval.py       Keyword scoring, domain-tag bonus, persistence
  test_tools.py            SymPy/SciPy/arXiv tools: correctness + failure handling
  test_registry.py          Domain-tag -> tool hint mapping
  test_orchestrator.py       Tool selection, execution, failure capture, synthesis
  test_logic_check.py         LogicCheck behavior + retry/failure handling
  test_physics_check.py        Cross-tool agreement logic + LLM critique combination
  test_math_check.py            Re-substitution verification, correct + incorrect solutions
  test_confidence_check.py       Threshold behavior, clamping, unparseable responses
  test_self_eval_pipeline.py      Full pipeline, crash isolation, check-ordering dependency
memory/
  episodic.jsonl      Created at runtime — one JSON line per problem run
```

## Running the tests

```bash
pytest tests/ -v
```

All 60 tests run offline (no LM Studio required) using `MockLLMClient`.
The physics tools themselves (SymPy solving, SciPy integration) and the
Math Check's re-substitution verification are exercised with real
computation, not mocked — only the LLM calls (planning, tool selection,
synthesis, logic/physics/confidence critique) are mocked.

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

## Design notes for Stage 2 specifically

- **Tool selection is domain-restricted, not open-ended.** `ToolRegistry.relevant_tools`
  filters which tools the LLM is even offered, based on the Stage 1 domain
  tags (`tools/registry.py::DOMAIN_TOOL_HINTS`). This cuts down on the model
  reaching for an irrelevant tool (e.g. literature search for a basic
  kinematics problem) and keeps the tool-selection call cheap.
- **Hallucinated tool names are silently dropped, not errored on.** If the
  model names a tool that wasn't offered, `_select_tool_calls` filters it
  out rather than crashing — a local model naming a tool it wasn't given is
  a parsing/prompt-following issue, not grounds to abort the whole solve.
- **Tool execution failures are captured, never raised.** Every tool
  (`SymbolicMathTool`, `SimulationTool`, `LiteratureSearchTool`) raises
  `ValueError` on bad input internally, and the orchestrator catches that
  and writes `{"error": "..."}` as the tool call's output rather than
  letting the exception propagate. This is deliberate: a failed tool call
  is exactly the kind of signal Stage 3 (self-evaluation) and Stage 5
  (self-correction) need to see in the trace later. Silently swallowing or
  crashing on it would destroy that signal.
- **Simulation and symbolic math are independent methods**, both callable
  on the same problem. This is what enables the "simulation and closed-form
  disagree" self-correction check from the design doc — Stage 3 can compare
  `symbolic_math`'s answer against `simulation`'s numerical integration of
  the same physical setup once both are called.
- **Literature search returns excerpts, not full abstracts**, and the tool
  is scoped to point at sources rather than to be treated as ground truth
  text to quote from downstream.

## Design notes for Stage 3 specifically

- **Checks are split into deterministic vs. LLM-based on purpose, not
  convenience.** Math Check and the cross-tool-agreement half of Physics
  Check never call the LLM at all — they re-verify actual computed results
  with SymPy. This matters because a check that just asks "is this correct?"
  to the same kind of model that produced the answer is weak evidence;
  independent, deterministic re-derivation is what the design doc's
  "self-correction" concept actually depends on. LLM critique is reserved
  for judgments that genuinely require language understanding (does the
  reasoning follow logically, are conservation laws respected given the
  problem's own narrative).
- **Physics Check directly implements the "simulation and closed-form
  disagree" row from the self-correction mapping table**: if both a
  `symbolic_math` and a `simulation` tool call were made, their numeric
  answers are compared (5% relative tolerance) with no LLM involved, and
  disagreement fails the check outright regardless of what an LLM critique
  says.
- **Math Check re-verifies by substitution, not by re-solving.** It
  substitutes each reported solution back into the original equation and
  checks the residual is ~0, rather than re-running `sympy.solve` and
  comparing — this catches cases where the *solve* was fine but something
  corrupted the reported solution en route (e.g. a transcription/formatting
  bug), which re-solving wouldn't catch since it would just reproduce the
  same computation.
- **A check that crashes is recorded as a failed check, not a pipeline
  crash.** `SelfEvaluationPipeline.run` wraps every check in try/except —
  consistent with Stage 2's tool-failure handling. A check raising is
  itself useful signal (a mis-behaving check needs Stage 7 meta-learning's
  attention), not a reason to lose the whole trace.
- **Confidence Check runs last on purpose** — it's given the other three
  checks' pass/fail results as part of its own input, so a failed Physics
  Check should (and, per the design doc's mapping, generally does) pull
  confidence down rather than the two signals being independent.
- **`trace.final_confidence` is named for its eventual role, not its
  current one.** Right now it's an unrevised first-pass estimate. Once
  Stage 5 exists, a correction pass may update it; the field itself doesn't
  change, only who last wrote to it.

## Next steps (Stage 5)

Self-correction engine: build the error-taxonomy mapping from the design
doc (detected check failure -> root-cause hypothesis -> corrective action),
an Error Detector that classifies `trace.checks_failed` + `check_details`
into a `trace.error_type`, and a Revision Planner that re-invokes Stage 2
tools differently based on that classification, incrementing
`trace.revision_count` and looping back through Stage 3 until checks pass
or a max-retry safety rail is hit.
