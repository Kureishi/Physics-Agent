# Physics Agent — Stage 1 + Stage 2 + Stage 3 + Stage 4

Stage 1: **task planner + retrieval**, plus the **trace schema** that every
later stage reads and writes. Stage 2: **tool orchestration** — symbolic
math, numerical simulation, and literature search, producing an initial
solution. Stage 3: **self-evaluation pipeline** — Logic, Physics, Math, and
Confidence checks that independently critique that initial solution.
Stage 4: **self-correction mapping** — a deterministic Error Detector plus
Revision Planner that loop back through Stages 2/3 until the candidate
passes verification or a safety rail is hit. See `physics_agent/trace.py`
for the full schema and a field-by-field note on which stage owns which field.

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
7. **Self-evaluates** that solution with four independent checks: Logic,
   Physics (cross-tool agreement + LLM critique), Math (re-substitution
   verification), Confidence. *(Stage 3)*
8. **If any check failed:** classifies *why* using a deterministic error
   taxonomy (algebra error, physics setup error, reasoning inconsistency,
   or unexplained low confidence), applies the matching correction strategy
   (re-derive math, re-derive physics setup, resynthesize the write-up, or
   escalate with an extra literature search), and **re-runs Stage 3** on the
   updated candidate. Repeats up to `max_revisions` times (default 3). *(Stage 4)*
9. **Writes a trace** of all of the above — including a full
   `revision_history` of every corrected attempt — to append-only episodic
   memory (JSONL).

It does *not* yet learn across problems (no meta-learning, no
tool-selection-policy adjustment, no knowledge-graph updates) — this stage
fixes *this* problem's answer; it doesn't yet make the agent better at the
next one. That's the outer loop from the design doc, still to come.

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
  self_correction/
    error_taxonomy.py      classify_error: deterministic checks_failed -> (error_type, strategy)
    revision_planner.py     RevisionPlanner: strategy -> concrete orchestrator action
    engine.py                 SelfCorrectionEngine: the detect-revise-reverify loop + safety rail
  trace.py            Trace schema + EpisodicMemory (JSONL append-only store)
  cli.py              Entry point wiring the full Stage 1 + 2 + 3 + 4 pipeline together
data/
  semantic_seed.json  Seed knowledge base (~13 core physics formulas across domains)
tests/
  test_trace.py         Trace roundtrip + episodic memory read/write
  test_planner.py         Decomposition, JSON-parsing robustness, retry/failure paths
  test_retrieval.py       Keyword scoring, domain-tag bonus, persistence
  test_tools.py            SymPy/SciPy/arXiv tools: correctness + failure handling
  test_registry.py          Domain-tag -> tool hint mapping
  test_orchestrator.py       Tool selection, execution, failure capture, synthesis, revision methods
  test_logic_check.py         LogicCheck behavior + retry/failure handling
  test_physics_check.py        Cross-tool agreement logic + LLM critique combination
  test_math_check.py            Re-substitution verification, correct + incorrect solutions
  test_confidence_check.py       Threshold behavior, clamping, unparseable responses
  test_self_eval_pipeline.py      Full pipeline, crash isolation, check-ordering dependency
  test_error_taxonomy.py           Every classification rule + priority ordering
  test_self_correction_engine.py    Full loop: resolves, exhausts retries, archives history
memory/
  episodic.jsonl      Created at runtime — one JSON line per problem run
```

## Running the tests

```bash
pytest tests/ -v
```

All 75 tests run offline (no LM Studio required) using `MockLLMClient`.
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

## Design notes for Stage 4 specifically

- **The Error Detector is a deterministic lookup table, not another LLM
  call.** By the time it runs, Stage 3 has already produced structured
  signal (`checks_failed`, `check_details`); re-asking an LLM "why did this
  fail" would just be re-deriving something already known, less reliably.
  `error_taxonomy.classify_error` is a fixed, inspectable mapping — exactly
  the kind of thing a later meta-learning stage could retune (e.g. "does
  the 'rederive_physics_setup' strategy actually resolve
  `cross_method_disagreement` more often than not?") if it stayed an LLM
  judgment call, there'd be nothing stable to retune.
- **Priority order encodes a root-cause assumption, made explicit rather
  than implicit.** If math, physics, and logic all fail simultaneously,
  the algebra error is treated as the fix to try first, since a wrong
  derivation can itself produce what looks like a physics disagreement or
  an incoherent write-up. This is a real modeling assumption, not a neutral
  default — it's worth revisiting once there's enough revision-history data
  to check whether it actually holds.
- **Revising overwrites `trace.tool_calls`/`checks_failed`/`check_details`
  each round; nothing is lost.** Stage 3's checks are written to evaluate
  "the current candidate" — if a corrected attempt's tool calls sat
  alongside the original broken ones, Math Check would fail forever on a
  mistake that's no longer part of the answer. `trace.revision_history`
  is where the pre-revision snapshot goes instead, so the full story
  survives without corrupting what the checks operate on.
- **The three "rederive" strategies and "resynthesize" are different
  amounts of work on purpose.** A Logic Check failure means the tools and
  physics were fine — only the write-up's reasoning was off — so
  `resynthesize` skips re-running any tools at all. Redoing tool calls for
  every failure type would waste real compute (a live LM Studio call, a
  fresh SymPy solve) fixing things that were never broken.
- **`escalate_verification` deliberately doesn't loop harder on the same
  approach.** A confidence-only failure means nothing specific was flagged
  as wrong — repeating the exact same tool calls would likely just
  reproduce the exact same (already-passing) result. Pulling in one
  external signal (literature search) is a genuinely different kind of
  evidence, which is the point.
- **The safety rail stops *trying*, it doesn't fabricate success.** At
  `max_revisions`, `resolution_status` is explicitly set to
  `"unresolved_max_revisions"` and `checks_failed` is left non-empty in the
  trace — nothing here quietly reports a passing result that didn't
  actually pass.

## Next steps

With Stages 1-4 done, the agent can solve a problem, verify its own answer
against independent methods, and correct itself when verification fails —
but every problem still starts from zero. What's still missing, per the
design doc's roadmap: **structured memory** (persisting what was learned
from each solve — not just the trace log, but semantic/procedural/error
memories that get *retrieved* on future problems), a **knowledge graph**
(confidence-tracked concepts and their conditions of validity, built up
over many solves), **meta-learning** (adjusting tool-selection policy,
verification depth, and the error-taxonomy priority ordering itself based
on what the accumulating revision_history actually shows works), and an
**autonomous curriculum** (generating new practice problems targeted at
whatever the knowledge graph shows is weakest). Let me know which of these
you'd like to tackle next.
