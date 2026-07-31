# Physics Agent — Stage 1 through Stage 8

Stage 1: **task planner + retrieval**, plus the **trace schema** that every
later stage reads and writes. Stage 2: **tool orchestration** — symbolic
math, numerical simulation, and literature search, producing an initial
solution. Stage 3: **self-evaluation pipeline** — Logic, Physics, Math, and
Confidence checks that independently critique that initial solution.
Stage 4: **self-correction mapping** — a deterministic Error Detector plus
Revision Planner that loop back through Stages 2/3 until the candidate
passes verification or a safety rail is hit. Stage 5: **memory
architecture** — episodic, semantic, procedural, and error memory, plus the
consolidator that writes to all four after each solve. Stage 6:
**knowledge graph** — typed edges between semantic-memory facts, enabling
a deterministic "is this formula valid here" check and low-confidence
clustering. Stage 7: **meta-learning / adaptive adjustments** — a
tool-selection policy and a verification-depth policy that actually change
future behavior based on accumulated outcomes, plus reporting tools for a
human to act on. Stage 8: **autonomous curriculum** — closes the loop:
generates new practice problems targeting Stage 7's ranked weak areas,
solves them through the exact same pipeline as any other problem, and
measures (honestly, not just optimistically) whether the practice actually
moved the underlying metric that was flagged. See `physics_agent/trace.py`
for the full schema and a field-by-field note on which stage owns which field.

## Five separate entry points

- **`python -m physics_agent.cli "<problem>"`** — solves one problem
  (Stages 1-7). What a person interacts with directly.
- **`python -m physics_agent.meta_report`** — reviews accumulated memory
  (check-value rates, declining strategies, ranked weak areas) without
  changing anything. An outer loop over many past solves, not a step in
  solving any one of them.
- **`python -m physics_agent.curriculum_cli`** — generates and solves new
  practice problems targeting the current weak areas (`--n N`), or
  summarizes past curriculum rounds (`--report`). The one entry point that
  writes new episodic traces without a person supplying the problem text —
  every generated trace is tagged `source="curriculum"` specifically so
  it stays distinguishable from ones a person submitted.
- **`python -m physics_agent.problem_set_cli`** — a batch harness that
  runs a whole JSON problem set through `cli.run()` in one command and
  prints a summary (resolution-status breakdown, average revisions,
  average confidence, planner domain-classification accuracy against the
  set's own hints). Not a new pipeline stage — just the fastest way to
  build up enough real data for Stage 7's policies and Stage 8's
  curriculum to have anything meaningful to act on. See
  `data/problem_sets/intro_physics_set.json` for a ready-made 24-problem
  set spanning all 14 domain tags.
- **`python -m physics_agent.inspect_trace_cli`** — dumps the full detail
  of exactly ONE solved problem: what was retrieved, every tool call, each
  self-evaluation check's verdict, and (if any revisions happened) each
  round's rationale, which check(s) failed, and whether that round's fix
  actually worked. `meta_report.py` aggregates across all traces;
  `problem_set_cli.py`'s summary tells you *that* one problem ended up
  `unresolved_max_revisions` — this tells you *why*, round by round.
  ```bash
  python -m physics_agent.inspect_trace_cli --list          # see what's in episodic memory
  python -m physics_agent.inspect_trace_cli "electron"       # search by substring in problem_text
  python -m physics_agent.inspect_trace_cli --id <problem_id>  # exact match
  ```

## What this does right now

Given a raw physics problem, the solving pipeline (`cli.py`):
1. **Classifies** it into 1-3 domain tags from a fixed physics taxonomy. *(Stage 1)*
2. **Decomposes** it into an ordered list of subtasks. *(Stage 1)*
3. **Retrieves** relevant formulas/concepts from a seeded semantic-memory store. *(Stage 1)*
4. **Selects tools** relevant to the problem's domain — narrowed by
   Stage 7's learned tool policy — and decides which to call, with what inputs. *(Stage 2, adjusted Stage 7)*
5. **Executes** those tool calls — symbolic algebra (SymPy), numerical ODE
   integration (SciPy), or arXiv literature search — capturing every call
   (including failures) into the trace. *(Stage 2)*
6. **Synthesizes an initial solution** from the tool outputs. *(Stage 2)*
7. **Self-evaluates** that solution with four independent checks: Logic,
   Physics (cross-tool agreement + knowledge graph validity + LLM
   critique), Math (re-substitution verification), Confidence — whose pass
   threshold may be raised by Stage 7's verification-depth policy. *(Stage 3, adjusted Stage 7)*
8. **If any check failed:** classifies *why*, applies the matching
   correction strategy, and re-runs Stage 3 on the updated candidate, up to
   `max_revisions` times. *(Stage 4)*
9. **Consolidates the solve into memory:** episodic, semantic confidence
   updates, procedural strategy success rates, error-signature frequency
   tracking. *(Stage 5)*
10. **Knowledge graph relationships** let `PhysicsCheck` query formula
    validity directly. *(Stage 6)*

Separately, `curriculum_cli.py` (Stage 8):
1. Calls Stage 7's `weak_areas()` to get the current top N ranked weak spots.
2. For each, measures the exact underlying metric that flagged it (error
   recurrence frequency, unresolved-problem count, or knowledge-graph
   cluster confidence) — the "before" snapshot.
3. Generates a new, self-contained practice problem targeting it (LLM-based,
   optionally grounded in a real `literature_search` result — never
   reproducing source text verbatim; see `problem_generator.py`).
4. Solves that problem through the *identical* Stage 1-7 pipeline
   (`physics_agent.cli.run`), tagged `source="curriculum"`.
5. Re-measures the same metric — the "after" snapshot — and logs the
   before/after comparison, honestly, to `memory/curriculum_log.jsonl`.

Confirmed live, not just unit-tested: seeded 3 recurrences of a
`cross_method_disagreement` error, ran a curriculum round targeting it, and
watched it generate a real practice problem, solve it end-to-end
(`resolution_status: passed_initial`), and correctly report the metric as
**unchanged** (3.0 → 3.0) rather than fabricating an improvement — the
practice problem simply didn't happen to trigger the same failure again,
and the benchmark says so plainly.

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

### Running the problem set end-to-end (recommended first real test)

This is the fastest way to see all eight stages working together on real
data, rather than one-off single problems.

1. Make sure LM Studio's server is running (see above).

2. Run the full problem set (24 problems spanning all 14 domain tags —
   kinematics, dynamics, energy, momentum, rotational dynamics,
   gravitation, oscillations, thermodynamics, electromagnetism, optics,
   fluid mechanics, special relativity, quantum mechanics, and statistical
   mechanics):
   ```bash
   python -m physics_agent.problem_set_cli
   ```
   This solves each problem through the complete Stage 1-7 pipeline (plan
   → tools → self-eval → self-correction → memory consolidation) and
   prints a running log plus a final summary: resolution-status
   breakdown, average revisions needed, average confidence, and how often
   the planner's own domain classification agreed with the problem set's
   human-assigned domain hint.

   Options:
   ```bash
   python -m physics_agent.problem_set_cli --limit 5              # just the first 5, for a quick smoke test
   python -m physics_agent.problem_set_cli --dry-run                # offline structural check, no LM Studio needed
   python -m physics_agent.problem_set_cli path/to/other_set.json     # run a different problem set
   ```
   Problem sets are plain JSON: a list of `{"id", "domain_hint", "problem_text"}`
   objects (`domain_hint` is optional, only used for the summary's
   classification-accuracy check). See
   `data/problem_sets/intro_physics_set.json` for the format.

3. If the summary flags anything unresolved, or you're just curious how a
   specific problem went, inspect it directly:
   ```bash
   python -m physics_agent.inspect_trace_cli --list
   python -m physics_agent.inspect_trace_cli "electron"   # substring of the problem text
   ```
   This shows exactly what was retrieved, every tool call, each
   self-evaluation check's verdict, and — for anything that needed
   revision — each round's rationale, which check(s) failed, and whether
   that round's fix actually worked. This is what tells you *why* a
   problem ended up `unresolved_max_revisions`, not just *that* it did.

4. Review what accumulated in memory:
   ```bash
   python -m physics_agent.meta_report
   ```
   With only 24 problems this likely won't cross the minimum-sample-size
   gates that let Stage 7's policies actually change behavior (5+ traces
   per domain for the tool policy and the confidence-calibration policy) —
   that's expected and by design, not a bug. Run the problem set a few
   more times (or add more problems to the JSON file) to build up enough
   history to see `check_value`, `declining_strategies`, and `weak_areas`
   populate with real signal.

5. Once there's enough history, run a curriculum round:
   ```bash
   python -m physics_agent.curriculum_cli --n 3
   ```
   This generates 3 new practice problems targeting the current top weak
   areas, solves them the same way, and logs before/after measurements.
   Review accumulated curriculum rounds any time with:
   ```bash
   python -m physics_agent.curriculum_cli --report
   ```

**A note on repeatability:** every run above writes to the same files
under `memory/`, and mutates `data/semantic_seed.json`'s confidence values
(Stage 5's `record_outcome`) plus `data/knowledge_graph_edges.json` if you
ever call `add_edge` directly. For a clean slate, delete `memory/`'s
contents (keep `.gitkeep`) and reset the two `data/*.json` files to their
originally committed values first.

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
    engine.py                 SelfCorrectionEngine: detect-revise-reverify loop + safety rail
  memory/
    procedural.py            ProceduralMemory: strategy success rates per (domain, error_type)
    error_memory.py           ErrorMemory: recurring failure signatures, root cause, fix, frequency
    consolidator.py            MemoryConsolidator: writes all four memory types after a solve
  knowledge_graph/
    graph.py                  KnowledgeGraph: typed edges over SemanticStore's nodes, validity
                              queries, low-confidence clustering, contradiction surfacing
  meta_learning/
    tool_policy.py             ToolSelectionPolicy: learns per-domain tool track records (ACTIVE --
                              wired into ToolOrchestrator)
    verification_depth.py       VerificationDepthPolicy: calibrates confidence threshold per domain,
                              only ever raises it (ACTIVE -- wired into ConfidenceCheck)
    check_value.py               compute_check_value_report: per-check catch rate (reporting only)
    pruning.py                     flag_declining_strategies: read-only signal over procedural memory
    curriculum_signals.py            weak_areas: ranks recurring errors / unresolved traces / low-
                              confidence KG clusters, with structured fields (error_type, node_ids)
                              for Stage 8 to re-measure without re-parsing text
    report.py                          build_report: ties the reporting-only signals together
  curriculum/
    problem_generator.py        ProblemGenerator: LLM-generated practice problems, optionally
                              grounded in a real literature_search result (never verbatim)
    curriculum_runner.py         CurriculumRunner: generate -> solve (via cli.run) -> measure
                              before/after; CurriculumLog persists round results
    benchmark.py                   summarize: honest improved/regressed/unchanged breakdown
                              per signal source, across many rounds
  trace.py            Trace schema + EpisodicMemory (JSONL append-only store, now with queries)
  retrieval.py        SemanticStore (Stage 1 retrieval + Stage 5 record_outcome confidence updates)
  cli.py              Solves one problem through the full Stage 1-7 pipeline
  meta_report.py       Periodic review of accumulated memory (Stage 7) -- no problem-solving
  curriculum_cli.py     Generates + solves practice problems, or reports on past rounds (Stage 8)
  problem_set_cli.py     Batch harness: runs a whole JSON problem set through cli.run(), prints summary
  inspect_trace_cli.py    Dumps full detail (checks, revision history, rationale) for one trace
data/
  semantic_seed.json         Seed knowledge base (~13 core physics formulas across domains)
  knowledge_graph_edges.json  Seed edges over those 13 formulas (derives_from/special_case_of/
                              requires_assumption relationships)
  problem_sets/
    intro_physics_set.json     24 problems spanning all 14 domain tags, for problem_set_cli.py
tests/
  test_trace.py         Trace roundtrip + episodic memory read/write
  test_planner.py         Decomposition, JSON-parsing robustness, retry/failure paths
  test_retrieval.py       Keyword scoring, domain-tag bonus, persistence, confidence updates
  test_tools.py            SymPy/SciPy/arXiv tools: correctness + failure handling
  test_registry.py          Domain-tag -> tool hint mapping
  test_orchestrator.py       Tool selection/execution/synthesis/revision + tool_policy wiring
  test_logic_check.py         LogicCheck behavior + retry/failure handling
  test_physics_check.py        Cross-tool agreement, knowledge graph validity, LLM critique
  test_math_check.py            Re-substitution verification, correct + incorrect solutions
  test_confidence_check.py       Threshold behavior, clamping + threshold_policy wiring
  test_self_eval_pipeline.py      Full pipeline, crash isolation, knowledge graph wiring
  test_error_taxonomy.py           Every classification rule + priority ordering
  test_self_correction_engine.py    Full loop: resolves, exhausts retries, archives history
  test_procedural_memory.py          Success-rate tracking, key normalization, min-uses gating
  test_error_memory.py                Recurrence frequency, signature grouping, persistence
  test_memory_consolidator.py          All four memory types updated correctly from one trace
  test_knowledge_graph.py               Edges, validity queries, clustering, contradictions
  test_tool_policy.py                    Round-0 tool extraction, success rates, filtering, safety
  test_verification_depth_policy.py       Overconfidence detection, one-directional threshold raise
  test_check_value.py                      Catch-rate computation across final + archived rounds
  test_pruning.py                           Declining-strategy flagging, thresholds, sort order
  test_curriculum_signals.py                 Weak-area ranking + structured fields for Stage 8
  test_meta_report.py                          Consolidated report shape and data reflection
  test_problem_generator.py                     Generation, literature grounding, retry/failure
  test_curriculum_runner.py                      Full generate -> solve -> measure integration
  test_curriculum_benchmark.py                    Improved/regressed/unchanged classification
  test_problem_set_cli.py                          Batch loading, crash isolation, limit handling
  test_inspect_trace_cli.py                          Search/lookup logic, full-detail printing
memory/
  episodic.jsonl      Created at runtime — one JSON line per problem run
  procedural.json      Created at runtime — strategy success-rate table
  error_memory.json     Created at runtime — recurring failure catalog
  curriculum_log.jsonl   Created at runtime — one JSON line per curriculum round
```

## Running the tests

```bash
pytest tests/ -v
```

All 204 tests run offline (no LM Studio required) using `MockLLMClient`.
The physics tools themselves (SymPy solving, SciPy integration), the Math
Check's re-substitution verification, and the knowledge graph's validity
queries are exercised with real computation, not mocked — only the LLM
calls (planning, tool selection, synthesis, logic/physics/confidence
critique, curriculum problem generation) are mocked.

To review accumulated memory after solving several problems:

```bash
python -m physics_agent.meta_report
```

To run a curriculum round (generate + solve practice problems targeting
current weak areas), or review past curriculum rounds:

```bash
python -m physics_agent.curriculum_cli --dry-run --n 2
python -m physics_agent.curriculum_cli --report
```

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

## Design notes for Stage 5 specifically

- **Consolidation and adaptation are kept as separate steps on purpose.**
  `MemoryConsolidator` only *writes* — it never changes error_taxonomy's
  fixed strategy choices or tool-selection behavior based on what it's
  recording. `ProceduralMemory.best_strategy_for` exists and is tested, but
  nothing calls it yet. Mixing "record what happened" with "change future
  behavior because of it" in one component would make both harder to get
  right and harder to audit — a later meta-learning stage is a better home
  for the second half.
- **Error signatures are coarse `(error_type, domain_tags)` pairs, not
  full check-detail text.** A finer-grained signature would fragment into
  near-unique entries and never accumulate a meaningful `frequency` —
  the whole point of error memory is noticing *patterns* across many
  different problems, which requires deliberately losing some
  per-incident detail (the `root_cause` field keeps the most recent
  occurrence's specific text, so nothing is fully discarded).
- **Semantic memory's confidence update is a soft nudge, not a hard
  overwrite** (`SemanticStore.record_outcome`, exponential moving average).
  One solve's outcome is weak evidence about a fact that may have been
  correctly applied in dozens of prior solves — a single confused problem
  shouldn't be able to tank a formula's confidence in one step.
- **Procedural memory requires a minimum of 3 uses before
  `best_strategy_for` returns anything**, specifically to avoid a single
  lucky/unlucky outcome looking like a real success-rate signal. This
  matters more here than almost anywhere else in the system, since
  procedural memory is the thing most likely to eventually *change* what
  the agent does automatically — bad early data compounding there is a
  worse failure mode than in, say, episodic memory, which is just a log.
- **The cross-run persistence was verified directly, not just unit
  tested**: two separate Python processes, each loading `ProceduralMemory`
  / `ErrorMemory` fresh from disk, showed `n_uses`/`frequency` correctly
  accumulate from 1 → 2 across the "sessions" — confirming this is real
  memory, not just per-run bookkeeping that happens to look like memory
  in an in-process test.

## Design notes for Stage 6 specifically

- **Nodes are not duplicated data.** `KnowledgeGraph.get_node` delegates
  straight to `SemanticStore` by id — the graph adds edges *on top of* the
  facts Stage 1/5 already maintain, rather than maintaining a second copy
  of confidence/provenance that could drift out of sync with what
  `record_outcome` is actually updating.
- **`check_validity` is honest about how narrow it is.** Only one
  assumption tag (`non_relativistic`, checked against a
  `special-relativity` domain tag) has a genuinely reliable signal
  derivable from domain tags alone. Every other seeded assumption
  (`ideal_gas_approximation`, `rigid_body`, `constant_acceleration`, etc.)
  is recorded as an edge — so the *structure* exists and grows as more
  reliable checks become possible — but deliberately can't fail a check
  yet, since a domain tag alone can't confirm or deny it. A clean
  `check_validity` result is evidence against one specific, checkable
  failure mode, not proof of correct usage.
- **Confirmed the catch is real, not just unit-tested**: ran `PhysicsCheck`
  against the actual seed data and actual graph (not fixtures) with a
  problem tagged `special-relativity` retrieving the classical
  (non-relativistic) kinetic energy formula, with the LLM critique mocked
  to pass — and the knowledge-graph sub-check failed it independently.
  That's the difference between "the check exists" and "the check catches
  something the other checks wouldn't."
- **Edge confidence is seeded, not dynamically updated, in this stage.**
  Per the design doc, node confidence updates via verification outcomes
  (already true, via `SemanticStore.record_outcome`, reused unchanged from
  Stage 5). Edges represent *structural* relationships between formulas
  (this is a special case of that; this requires that assumption) which
  don't get individually exercised by a single solve the way a retrieved
  fact does — there's no clean signal from one solve about whether a
  `derives_from` edge itself was "used correctly." Extending edges to
  update their own confidence would need a real signal source, which
  doesn't exist yet; recorded here as a known gap rather than faked with an
  arbitrary update rule.
- **`find_low_confidence_clusters` only groups nodes that are BOTH
  connected AND individually low-confidence** — not just anything
  connected to a low-confidence node. Verified directly: after repeatedly
  recording failures against `eng-002` (dragging its confidence down) while
  its `derives_from` neighbor `grav-001` stayed untouched, the resulting
  cluster was `["eng-002"]` alone, not `["eng-002", "grav-001"]`. A
  neighbor being high-confidence isn't itself suspect just for being
  adjacent to a weak fact.

## Design notes for Stage 7 specifically

- **Only two signals actually change behavior; the rest only report, and
  that split is deliberate.** `ToolSelectionPolicy` and
  `VerificationDepthPolicy` are consulted live in Stages 2/3.
  `check_value.py`, `pruning.py`, and `curriculum_signals.py` only compute
  and return data. The dividing line isn't arbitrary: the two active
  policies each have a single, auditable, one-directional lever (narrow
  tool choices given a proven-poor track record; raise a safety threshold
  given proven overconfidence) where the worst case of being wrong is
  "slightly less efficient," never "slightly less safe." Auto-disabling a
  verification check or auto-retiring a correction strategy is a
  fundamentally different kind of lever — the worst case there is losing a
  safety guarantee based on a proxy signal — so those stay as reports for
  a human (or a more conservative future mechanism) to act on.
- **Every active policy is gated by a minimum sample size**
  (`MIN_USES_BEFORE_ACTING`, `MIN_TRACES_BEFORE_ACTING`) and has a defined
  "not enough data yet" return value (`None`, or the untouched default).
  This mirrors `ProceduralMemory.best_strategy_for`'s pattern from Stage 5
  for the same reason: a policy that acts on a 1-sample or 2-sample rate is
  more likely to encode noise than signal, and unlike a report a human can
  sanity-check, an active policy's mistakes get repeated automatically.
- **`VerificationDepthPolicy` only ever raises the threshold, never lowers
  it below the system default.** This is the one asymmetric design choice
  worth calling out explicitly: a domain that looks "underconfident" (low
  reported confidence, but outcomes are actually fine) never gets an
  automatically *relaxed* bar, even though the data would technically
  support it. Lowering the bar trades safety for speed on a proxy signal;
  this system doesn't make that trade automatically, only the reverse one.
- **`ToolSelectionPolicy` reconstructs "what tools were used in the very
  first attempt"** from `trace.revision_history[0]` when any revision ever
  happened, rather than reading `trace.tool_calls` directly — a reminder of
  Stage 4's invariant that `tool_calls` always reflects only the *current*
  round. Getting this wrong would have quietly attributed a later,
  corrected round's tool choices to "what worked from the start."
- **Both active policies were confirmed changing real behavior, not just
  passing unit tests against fixtures**: seeded synthetic episodic history
  showing `simulation` never leading to a clean optics solve, and watched
  `ToolOrchestrator` genuinely stop offering it on a new problem; seeded
  history showing `quantum-mechanics` reporting 0.9 confidence while
  frequently ending up unresolved, and watched `ConfidenceCheck` correctly
  fail a new 0.85-confidence solution that a static 0.6 threshold would
  have passed.
- **Policies are recomputed fresh from disk on every single solve**, not
  cached and refreshed on a schedule, even though the design doc frames
  meta-learning as something that runs "periodically... over a batch." At
  this project's scale (a JSONL file, read linearly) that's cheap enough
  not to matter; a system solving at high volume would more likely
  recompute on a schedule instead of per-solve, but the policies'
  *behavior* wouldn't need to change to support that -- just how often
  `ToolSelectionPolicy(episodic)` gets re-instantiated.

## Design notes for Stage 8 specifically

- **Generated problems go through the exact same pipeline, not a
  simplified one.** `CurriculumRunner.run_round` calls
  `physics_agent.cli.run` directly — the same function a person's typed
  problem goes through. This was a deliberate reuse decision, not just
  convenience: a curriculum whose practice problems were solved by a
  different, "practice-mode" code path could easily drift from what the
  agent actually does in production, silently making the benchmark
  meaningless.
- **"Read literature" is honestly scoped.** `ProblemGenerator` can see a
  real `literature_search` title and short excerpt and is explicitly told
  it may draw on the *general subject matter*, but the prompt forbids
  copying phrases verbatim — consistent with how `LiteratureSearchTool`
  itself is scoped (see Stage 2's notes). This system does not "reproduce
  papers" in the sense of full comprehension or replication; claiming that
  would overstate what a short title+excerpt grounding actually provides.
- **Before/after measurement re-derives the exact metric that produced the
  signal**, not a proxy for it (`_measure_signal` dispatches on
  `signal["source"]` and reads the same store — `ErrorMemory`,
  `EpisodicMemory`, or the knowledge graph's node confidences — that
  `weak_areas()` itself read). This is why `weak_areas()` gained
  structured fields (`error_type`, `node_ids`) in this stage rather than
  requiring the runner to re-parse a human-readable "reason" string.
- **The benchmark reports "unchanged" and "regressed" as real, distinct
  outcomes, not just "improved" with noise.** Confirmed directly: a
  practice problem that didn't happen to retrigger a targeted
  `cross_method_disagreement` error left that error's frequency at 3.0
  both before and after, and the round was logged and reported as
  `unchanged` — not silently folded into "improved" just because nothing
  got worse. A system that only ever reports success isn't measuring
  anything.
- **Curriculum-generated problems are tagged, not hidden.**
  `trace.source == "curriculum"` and `trace.curriculum_target` make every
  self-generated practice problem distinguishable from a person's problem
  in episodic memory — including to Stage 7's own policies, which read all
  traces indiscriminately today. Whether curriculum-generated traces
  *should* be weighted differently in `ToolSelectionPolicy` or
  `VerificationDepthPolicy` (e.g. discounted, since their difficulty was
  chosen by the system itself rather than arriving organically) is a real
  open question this implementation doesn't resolve — it makes the
  distinction available rather than making that judgment call silently.
- **One-shot generation, not curated.** If the LLM produces an ill-posed or
  unsolvable problem, nothing here filters it out before solving — it goes
  through the pipeline like anything else, and a bad practice problem is
  itself visible in the resulting trace (e.g. tool failures, a low
  confidence score, or an unresolved status) rather than silently
  discarded. Generation failures (bad JSON after retries) are the one
  thing skipped outright, consistent with every other component's stance
  of not crashing the whole round over one bad sub-step.

## Where this leaves the design doc's roadmap

All eight stages from the original roadmap are now implemented:
task planner + retrieval, tool orchestration, self-evaluation,
self-correction, memory architecture, knowledge graph, meta-learning, and
the autonomous curriculum. The loop the very first design doc described —
solving and learning as separate processes that feed each other — is now
actually closed: solved problems accumulate into memory and a knowledge
graph; meta-learning reads that accumulation and changes future
tool-selection and verification behavior; the curriculum reads the same
accumulation to generate new problems targeting exactly what's weakest;
and those new problems solve through the identical pipeline, feeding the
same memory that started the cycle.

What would be worth doing next isn't a new stage so much as deepening the
existing ones with real usage: running this against an actual LM Studio
model on a real problem set to see which of the deliberately conservative
choices throughout (confidence-only threshold raises, minimum sample sizes
before any policy acts, narrow assumption-validity checks) turn out to be
well-calibrated in practice versus too cautious or not cautious enough.
Let me know if you'd like to do that, revisit anything in Stages 1-8, or
take the project somewhere the original roadmap didn't anticipate.
