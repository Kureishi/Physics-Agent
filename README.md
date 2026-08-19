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

## Entry points

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
  `data/problem_sets/intro_physics_set.json` for a ready-made 206-problem
  set spanning all 15 domain tags.
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
- **`python -m physics_agent.generate_problem_set_cli`** — bulk-generates
  new practice problems to expand what `problem_set_cli.py` runs against,
  reusing Stage 8's `ProblemGenerator` directly but without needing any
  accumulated weak-area data first (unlike `curriculum_cli.py`, this
  doesn't solve what it generates — it's purely for growing the problem
  set). Runs at a higher temperature than solving on purpose, feeds each
  domain's own previously-generated problems back in as a capped "avoid
  duplicating" list so N calls in a row don't just restate the same
  scenario with different numbers, and retries any failed attempt (empty
  response, malformed JSON, or a raw call error) at a lower, more
  conservative temperature.
  ```bash
  python -m physics_agent.generate_problem_set_cli --n-per-domain 5 --out data/problem_sets/expanded_set.json
  python -m physics_agent.generate_problem_set_cli --domains energy dynamics --n-per-domain 10
  python -m physics_agent.generate_problem_set_cli --n-per-domain 3 --append data/problem_sets/intro_physics_set.json
  python -m physics_agent.generate_problem_set_cli --n-per-domain 5 --max-retries 4   # if seeing many "generation failed" skips
  ```
- **`python -m physics_agent.canary_cli`** — solves the fixed, hand-verified
  canary set (`data/canary_problems.json`) and grades the self-eval checks
  against known-correct answers, not just against themselves; exits
  nonzero if any check disagreed with ground truth. `--dry-run` is an
  explicit structural smoke test only (the mock LLM ignores
  problem-specific numbers, so it can't actually grade); `--report`
  reviews past runs.
  ```bash
  python -m physics_agent.canary_cli
  python -m physics_agent.canary_cli --report
  ```
- **`python -m physics_agent.scheduler_cli`** — the background
  decision loop: decides for itself when to solve (from a queue), review,
  or practice, instead of a person running each workflow by hand. See
  "The Scheduling/Decision Loop" below.
  ```bash
  python -m physics_agent.scheduler_cli --once
  python -m physics_agent.scheduler_cli --loop --interval-seconds 60
  python -m physics_agent.scheduler_cli --report
  ```
- **`python -m physics_agent.knowledge_growth_cli`** — checks for
  repeated tool-call expressions across many cleanly-solved traces and
  proposes them as new, low-confidence semantic facts (`--dry-run-proposals`
  to preview without committing; `--min-observations N` to lower the bar).
  Also runs periodically as the scheduler's "grow" decision.
  ```bash
  python -m physics_agent.knowledge_growth_cli --dry-run-proposals
  python -m physics_agent.knowledge_growth_cli
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

Two more worth knowing about if you're trying a model you haven't used
with this before, especially a "thinking"/reasoning-tuned one:

```bash
export LM_STUDIO_TIMEOUT_SECONDS="120"   # default; raise if a model needs longer per call
export LM_STUDIO_MAX_TOKENS="2048"        # default; raise if a model needs a longer response/reasoning trace
```

Chat completions here are non-streaming, so nothing returns until the
whole response is done — a model that reasons at length before answering,
or one that never emits a stop token in a particular quantization, can
look exactly like a hang with no way to tell the two apart from outside.
Both settings exist specifically so that turns into a clean, bounded
failure instead (see "Bug 5" near the end of this document for the full story).

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

2. Run the full problem set (206 problems spanning all 15 domain tags —
   kinematics, dynamics, energy, momentum, rotational dynamics,
   gravitation, oscillations, thermodynamics, electromagnetism, optics,
   fluid mechanics, special relativity, quantum mechanics, statistical
   mechanics, and nuclear physics):
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
   At ~14 problems per domain, this comfortably crosses the
   minimum-sample-size gates that let Stage 7's policies actually change
   behavior (5+ traces per domain for the tool policy and the
   confidence-calibration policy) — so after running the full set once,
   `check_value`, `declining_strategies`, and `weak_areas` should already
   show real signal rather than empty/insufficient-data results. (If
   you're running a smaller or custom problem set instead, that's when
   you'd still need multiple passes or more problems to clear those gates
   — see the note on gate thresholds under "Getting more problems" below.)

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
  json_utils.py      Shared defensive JSON extraction + backslash/LaTeX sanitization, used across
                     planner/orchestrator/self-eval checks/problem generator
  planner.py         TaskPlanner: domain classification + subtask decomposition        (Stage 1)
  retrieval.py        SemanticStore: keyword-scored retrieval over seeded physics facts  (Stage 1)
  orchestrator.py     ToolOrchestrator: tool selection, execution, solution synthesis with
                      retry-on-empty-response   (Stage 2)
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
    engine.py                 SelfCorrectionEngine: detect-revise-reverify loop + safety rail +
                              escalated_for_human_review outcome + strategy-override hook
    escalation.py               detect_escalations: per-problem escalations + domains where
                              unresolved_max_revisions recurs heavily (reporting only)
  canary/
    problems.py                CanaryProblem/load_canary_problems: the fixed, hand-verified set
    grading.py                   Deterministic numeric grading, tool-output preferred over prose
    runner.py                     CanaryRunner: solves + grades checks against ground truth
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
    strategy_override.py         StrategyOverridePolicy: replaces error_taxonomy's fixed default
                              strategy once procedural memory has a stronger track record (ACTIVE
                              -- wired into SelfCorrectionEngine)
    check_value.py               compute_check_value_report: per-check catch rate (reporting only)
    anomaly.py                     detect_check_value_anomalies: recent catch rate vs. historical
                              baseline per check, flags jump/collapse (reporting only)
    knowledge_growth.py            find_candidate_facts/propose_and_add: proposes new low-
                              confidence semantic facts from repeated tool-call expressions
                              (ACTIVE -- wired into the scheduler's "grow" decision)
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
  scheduler/
    queue.py                    ProblemQueue: FIFO over a problem-set-shaped JSON file
    state.py                     SchedulerState: persisted cadence counters across restarts
    scheduler.py                   Scheduler: run_cycle() decides solve/review/practice/grow/idle,
                              each logged as a Decision with its own reason
  trace.py            Trace schema + EpisodicMemory (JSONL append-only store, now with queries)
  retrieval.py        SemanticStore (Stage 1 retrieval + Stage 5 record_outcome confidence updates)
  cli.py              Solves one problem through the full Stage 1-7 pipeline
  meta_report.py       Periodic review of accumulated memory (Stage 7) -- no problem-solving
  curriculum_cli.py     Generates + solves practice problems, or reports on past rounds (Stage 8)
  problem_set_cli.py     Batch harness: runs a whole JSON problem set through cli.run(), prints summary
  inspect_trace_cli.py    Dumps full detail (checks, revision history, rationale) for one trace
  generate_problem_set_cli.py  Bulk-generates new problems across domain tags, no weak-area data needed
  canary_cli.py                 Solves + grades the fixed canary set against ground truth (Safety Rails)
  scheduler_cli.py                Runs the background decision loop once, in a bounded/unbounded
                              loop, or reports past decisions
  knowledge_growth_cli.py           Proposes new semantic facts from repeated tool use, or previews
                              candidates without committing (Autonomous Knowledge Growth)
data/
  semantic_seed.json         Seed knowledge base (~13 core physics formulas across domains) --
                              also where self-derived facts (Knowledge Growth) get appended
  knowledge_graph_edges.json  Seed edges over those 13 formulas (derives_from/special_case_of/
                              requires_assumption relationships)
  canary_problems.json        10 hand-verified problems with computed expected answers (Safety Rails)
  problem_sets/
    intro_physics_set.json     206 problems spanning all 15 domain tags, for problem_set_cli.py
tests/
  test_trace.py         Trace roundtrip + episodic memory read/write
  test_planner.py         Decomposition, JSON-parsing robustness, retry/failure paths
  test_retrieval.py       Keyword scoring, domain-tag bonus, persistence, confidence updates
  test_tools.py            SymPy/SciPy/arXiv tools: correctness + failure handling
  test_registry.py          Domain-tag -> tool hint mapping
  test_orchestrator.py       Tool selection/execution/synthesis/revision, tool_policy wiring,
                      and synthesis retry-on-empty-response (Bug 8)
  test_logic_check.py         LogicCheck behavior + retry/failure handling
  test_physics_check.py        Cross-tool agreement, knowledge graph validity, LLM critique
  test_math_check.py            Re-substitution verification, correct + incorrect solutions
  test_confidence_check.py       Threshold behavior, clamping + threshold_policy wiring
  test_self_eval_pipeline.py      Full pipeline, crash isolation, knowledge graph wiring
  test_error_taxonomy.py           Every classification rule + priority ordering
  test_self_correction_engine.py    Full loop: resolves, exhausts retries, archives history,
                      generalized escalation (Bug 9)
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
  test_generate_problem_set_cli.py                     Domain coverage, unique ids, avoid-list growth
  test_llm_client.py                                     Timeout/max_tokens threading, per-call overrides
  test_json_utils.py                                     Backslash/LaTeX sanitization, incl. exact real crash
  test_canary_grading.py                                 Numeric extraction, tool-output vs. prose preference
  test_canary_problems.py                                Loading the fixed canary set, field validation
  test_canary_runner.py                                  Full solve->grade->verdict classification
  test_anomaly.py                                        Jump/collapse detection, threshold, insufficient data
  test_escalation.py                                     Per-problem + recurring-domain escalation grouping
  test_strategy_override.py                              Override bars: min uses, improvement margin, absolute floor
  test_scheduler.py                                      Full solve/review/practice/idle decision integration
  test_scheduler_queue.py                                FIFO semantics, persistence, external-file compatibility
  test_scheduler_state.py                                Cadence counter persistence roundtrip
  test_knowledge_growth.py                               Candidate detection, propose_and_add, idempotent registry
memory/
  episodic.jsonl      Created at runtime — one JSON line per problem run
  procedural.json      Created at runtime — strategy success-rate table
  error_memory.json     Created at runtime — recurring failure catalog
  curriculum_log.jsonl   Created at runtime — one JSON line per curriculum round
  canary_log.jsonl         Created at runtime — one JSON line per canary problem per run
  scheduler_state.json      Created at runtime — scheduler cadence counters
  scheduler_log.jsonl        Created at runtime — one JSON line per scheduler decision
  proposed_facts.json          Created at runtime — signature -> entry_id registry (Knowledge Growth)
```

## Running the tests

```bash
pytest tests/ -v
```

All 370 tests run offline (no LM Studio required) using `MockLLMClient`.
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

## Beyond the original 8 stages: Safety Rails, closing Stage 7, a Scheduler, and Knowledge Growth

A follow-up round of work, done against the already-complete 8-stage
system above, picking up threads the Stage 7 design discussion had
explicitly flagged as deliberately left undone rather than forgotten.

### Safety Rails

**Ground-truth canary problems.** `data/canary_problems.json` is a small
(10-problem), hand-verified, fixed set spanning 10 of the 15 domains, each
with a computed `expected_value` and a documented derivation
(`verified_by`). `physics_agent/canary/` solves each one through the exact
same pipeline as any other problem, then grades not just the *answer* but
whether the self-eval checks *agreed with ground truth* — four possible
verdicts (`correct_and_passed`, `correct_but_flagged`,
`incorrect_but_passed`, `incorrect_and_flagged`), where the middle two are
the interesting ones: a check disagreeing with a known-correct answer in
either direction. This is a meaningfully different signal from anything
else in the system, which otherwise only ever checks itself against
itself. Grading prefers the deterministic `symbolic_math`/`simulation`
tool output over parsing `trace.final_answer` prose (the same reasoning as
Stage 3's "independent, deterministic re-derivation" principle), falling
back to prose extraction only when no tool output is usable.
`python -m physics_agent.canary_cli` runs the suite and exits nonzero on
any concerning verdict; `--report` reviews history.

**Check-value anomaly detection.** `meta_learning/anomaly.py` compares
each self-eval check's catch rate over the most recent N traces (default
20) against everything before that, flagging a `jump` or `collapse` past
an absolute threshold (default 15 points). This exists specifically
because of Bug 7 above: `MathCheck`'s catch rate jumping from ~5% to
~87% on essentially the same data is exactly the shape that bug had, and
would have surfaced it in one `meta_report` run instead of requiring a
manual scan of 183 accumulated traces after the fact. Confirmed live, not
just unit-tested: run against this project's own real accumulated
`memory/episodic.jsonl`, it immediately found two genuine anomalies
(`logic` and `confidence` both roughly tripling their catch rate over the
last 20 traces) — a real signal worth investigating, surfaced by the tool
built to surface it.

**An escalation path distinct from `max_revisions`.** Previously, a trace
that exhausted its revision budget only ever ended up
`unresolved_max_revisions` — "stop and mark unresolved," with no "stop and
ask a person" outcome. `self_correction/engine.py` now recognizes when a
correction strategy is selected *again* on a later round for the same
trace after every earlier use of that exact strategy failed to resolve
anything — meaning repeating it again wouldn't add new signal — and stops
early with a new `resolution_status`, `escalated_for_human_review`,
instead of burning the remaining revision budget on an action already
shown not to work in this trace. (A strategy that *did* succeed once, then
gets reused later for a newly-appeared issue, isn't penalized — only a
strategy with a track record of pure failure triggers this.) This started
narrower — originally scoped just to `escalate_verification` repeating —
and was generalized after a real accumulated run showed the same shape on
a much more common path: `resynthesize` sitting at a flat 0% success rate
across several domains, each trace still burning all 3 revisions before
landing on `unresolved_max_revisions` regardless. Applied retroactively to
that real data, the generalized version would have caught **19 of 19** of
those unresolved traces early instead of letting every one of them
exhaust its revision budget. Separately, `self_correction/escalation.py`
looks across *many* traces for domains where `unresolved_max_revisions`
recurs past a threshold (default 5) — a pattern, not a one-off hard
problem — and flags that too. Both surface in `meta_report.py`'s new
"Escalations" section.

### Stage 7, closed: procedural memory actually overriding the error taxonomy

The original Stage 7 notes above say it plainly: `ProceduralMemory.
best_strategy_for` existed and was tested, but nothing read it to change
behavior — `error_taxonomy.py`'s strategy mapping stayed a fixed lookup
table. `meta_learning/strategy_override.py` closes that: once an
alternative corrective strategy has a strong-enough track record for a
specific `(domain, error_type)` pair, it replaces the taxonomy's
hardcoded default. The bar for this is deliberately higher than
`best_strategy_for`'s own floor of 3 uses — 5 uses minimum, and either a
15-point improvement over the default's own tracked success rate (if the
default has real data here too) or a 50%-or-better absolute success rate
(if it doesn't) — because "trusted enough to report" and "trusted enough
to act on automatically, unreviewed" aren't the same bar. This is the
third policy (joining `ToolSelectionPolicy` and `VerificationDepthPolicy`)
that changes live solving behavior rather than only reporting. Confirmed
directly: with procedural memory seeded to show `resynthesize` resolving
a math error 80% of the time versus the untested default `rederive_math`,
the engine applied and archived `resynthesize`, not the taxonomy default
— and a companion test with the identical seeded data but no policy
attached confirmed the override is opt-in, not automatic just because the
data exists.

### Autonomous Knowledge Growth

Every fact in `data/semantic_seed.json` and every edge in
`data/knowledge_graph_edges.json` was hand-written before this round of
work — there was no mechanism for the agent to notice it kept re-deriving
the same formula from scratch across many problems and propose adding it
as its own memory entry. `meta_learning/knowledge_growth.py` closes that,
narrowly: it looks at the `symbolic_math` tool-call `expression` behind
every cleanly-resolved trace's *final* round (per Stage 4's own invariant
that `trace.tool_calls` holds only the current, passing candidate by the
time solving finishes — not a stale pre-correction attempt), and when the
same expression (whitespace-normalized) recurs across at least
`MIN_OBSERVATIONS_TO_PROPOSE` (default 5) independently-solved traces, it
becomes a candidate for a new semantic-memory fact.

Two clearly separated steps, like every other Stage 7 signal:
`find_candidate_facts()` is read-only; `propose_and_add()` is what
actually calls `SemanticStore.add()` for candidates that clear the bar and
haven't been proposed before (tracked in a small persisted registry,
`memory/proposed_facts.json`, keyed by a hash of the expression signature,
so re-running this doesn't duplicate the same fact every time). A
proposed fact starts at confidence 0.3 — well below every seeded fact
(roughly 0.85–0.99) — with `provenance` explicitly labeled
`self_derived` and `conditions` honestly stating that validity hasn't been
characterized by a person, unlike a seeded fact's curated conditions text.
From there it's refined exactly like any other fact: future solves that
retrieve it nudge its confidence via the same `record_outcome` seeded
facts already use.

This is a documented, deliberate simplification, not an oversight: matching
is a plain normalized-string comparison, not symbolic equivalence
checking, so `Eq(m*g*h, 0.5*m*v**2)` and the algebraically-identical
`Eq(v**2, 2*g*h)` track as two separate candidates rather than being
recognized as the same physics. The proposed "statement" is built directly
from the tool call itself (e.g. `"Eq(m*g*h, 0.5*m*v**2), solved for v"`)
rather than LLM-paraphrased into confident prose nobody has vetted the
wording of. And no attempt is made to detect whether a seeded fact's
hand-written prose already covers the same physics as a self-derived
expression signature — the two vocabularies essentially never collide as
strings, so a resulting near-duplicate is accepted as low-cost (a second,
low-confidence description isn't a correctness bug) rather than engineered
around.

`python -m physics_agent.knowledge_growth_cli` runs it standalone
(`--dry-run-proposals` to preview without committing, `--min-observations
N` to lower the bar); it also runs as a fourth scheduler decision,
**Grow**, gated by its own, longer cooldown
(`scheduler_growth_min_cycles_between_rounds`, default 10 cycles — writing
new persistent knowledge is a heavier action than a practice round, so it
isn't checked as eagerly) and logged the same way solve/review/practice
are. Confirmed directly against this project's own real accumulated
`memory/episodic.jsonl`: at a lowered threshold of 2 observations, it
correctly surfaced 6 genuine repeated-formula patterns — the thin-lens
equation (used 4 times across separate optics problems),
momentum-conservation, mass-defect-to-energy conversion, and others — and
committing them added exactly 6 new low-confidence entries to
`semantic_seed.json`, each correctly tagged `self_derived` at confidence
0.3. Re-running immediately afterward correctly added nothing (`"All
qualifying candidates were already proposed in an earlier run"`),
confirming the idempotence the registry exists for.

### The Scheduling/Decision Loop

Every entry point above (`cli.py`, `curriculum_cli.py`, `meta_report.py`,
`problem_set_cli.py`) is a manual trigger a person has to remember to run
and decide *when* to run. `physics_agent/scheduler/` is the missing
decision layer: `Scheduler.run_cycle()` does whatever combination of
solving, reviewing, practicing, and growing is actually warranted right
now, using the exact same underlying functions those CLIs already call
rather than reimplementing any of them —

1. **Solve** — pop one problem from a queue (`scheduler/queue.py`, the
   same JSON shape as `data/problem_sets/*.json`, fillable by
   `generate_problem_set_cli.py` or by hand) and run it through `cli.run`.
2. **Review** — once `scheduler_review_every_n_solves` (default 20) new
   solves have accumulated, run the exact `build_report()`
   `meta_report.py` prints.
3. **Practice** — once `weak_areas()`'s top signal clears a weight
   threshold (default 5) *and* a cycle cooldown has passed since the last
   round, run one `CurriculumRunner` round targeting it — decoupled from
   whether a solve happened this specific cycle, so a persistently weak
   domain still gets addressed even on an otherwise-idle cycle.
4. **Grow** — once its own, longer cycle cooldown has passed, check
   `knowledge_growth.find_candidate_facts()` and commit any that clear the
   bar via `propose_and_add()`. See "Autonomous Knowledge Growth" above.

Every action taken — including doing nothing — is logged as a `Decision`
with a plain-English reason to `memory/scheduler_log.jsonl`, so an empty
log is distinguishable from "never ran" rather than looking the same as
"ran and correctly found nothing to do." `python -m
physics_agent.scheduler_cli --once` runs a single cycle; `--loop
--interval-seconds N [--max-cycles N]` keeps running (meant to sit under
systemd/cron/nohup, not daemonize itself); `--report` reviews the
decision history. Confirmed live against a real queued problem set: two
cycles correctly solved both queued problems in order, and a third
correctly went idle the moment the queue drained rather than doing nothing
silently.

All of the above is exercised by 370 tests (up from the original 256),
all still offline via `MockLLMClient` — none of it required LM Studio to
verify at the unit-test level, though every piece was also smoke-tested
against this project's real accumulated `memory/` and a real `--dry-run`
CLI invocation, not just fixtures.

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

## Adding a new domain

`planner.DOMAIN_TAXONOMY` is a fixed vocabulary, not something the system
infers on its own — a problem the planner can't classify into one of these
tags gets its unrecognized tag silently dropped
(`planner.py`'s `domain_tags = [t for t in domain_tags if t in DOMAIN_TAXONOMY]`).
Adding a genuinely new domain (nuclear physics is the 15th, added exactly
this way, live, as a worked example — see `git diff`-equivalent below):

**Required — the system won't recognize the domain at all without this:**
1. Add the tag string to `planner.DOMAIN_TAXONOMY`. That's it for
   "the system accepts and classifies problems into this domain" — the
   task-planner prompt is built from this list via an f-string, so nothing
   else needs to change for classification to start working.

**Strongly recommended — the system works without these, but with
degraded quality:**
2. Add an entry to `tools.registry.DOMAIN_TOOL_HINTS` mapping the new tag
   to whichever of `symbolic_math` / `simulation` / `literature_search`
   are actually relevant. Skipping this isn't a bug —
   `ToolRegistry.relevant_tools` falls back to offering *all* tools when a
   domain has no hint entry — but it means no domain-specific narrowing,
   which was the whole point of that mapping (Stage 2's design notes).
3. Add a few seed facts to `data/semantic_seed.json`, tagged with the new
   domain. Without this, `SemanticStore.retrieve`'s tag-bonus scoring has
   nothing to reward, and retrieval falls back to whatever keyword-only
   matches happen to score highest — possibly nothing relevant at all, as
   opposed to the domain's own formulas.
4. Add corresponding `data/knowledge_graph_edges.json` entries for any new
   facts' real `requires_assumption` / `special_case_of` / `derives_from`
   relationships, so `PhysicsCheck`'s knowledge-graph sub-check (Stage 6)
   has something to work with for the new domain.

**Optional, situational:**
5. If the new domain has an assumption whose violation is *reliably
   detectable from a domain tag alone* — the way `non_relativistic` is
   reliably violated by a `special-relativity` tag — add it to
   `knowledge_graph.graph.ASSUMPTION_DOMAIN_CONFLICTS`. Most physics
   assumptions aren't like this (see that dict's docstring); nuclear
   physics's two new assumptions (`large_sample_statistical_average`,
   `rest_frame_measurement`) don't have an analogous tag-detectable
   conflict in the current taxonomy, so nothing was added there — the
   edges still exist and are still inspectable, they just can't fail a
   check on domain-tag evidence alone yet.
6. Add example problems tagged with the new `domain_hint` to a problem set
   for `problem_set_cli.py` to exercise it.

**Nothing else needs to change.** `generate_problem_set_cli.py` imports
`DOMAIN_TAXONOMY` directly, so its `--domains` default and validation stay
in sync automatically. Every Stage 7 meta-learning policy
(`ToolSelectionPolicy`, `VerificationDepthPolicy`), Stage 8's curriculum
(`weak_areas`, `CurriculumRunner`), and `error_taxonomy.classify_error`
all operate generically over whatever `domain_tags` show up in traces —
none of them hardcode a domain list, so a new domain starts participating
in all of them automatically as soon as traces with that tag exist.

This was verified directly, not just described: after making steps 1-4
above (nuclear physics), retrieval correctly surfaced `nuc-001` (the decay
law) as the top match for a half-life problem tagged `nuclear-physics`,
`ToolRegistry.relevant_tools(["nuclear-physics"])` returned exactly the
three intended tools, and both new knowledge-graph edges resolved and
passed `check_validity` correctly.

## Getting more problems to run against

Three ways to scale up beyond the 206-problem starter set, roughly in
order of effort:

1. **Bulk-generate with `generate_problem_set_cli.py`** (built for exactly
   this). Reuses the same `ProblemGenerator` Stage 8 already has, at a
   higher temperature than solving (diversity matters more here than
   accuracy), feeding each domain's own prior outputs back in as an
   "avoid duplicating" list so a run of N calls doesn't just restate the
   same block-on-an-incline scenario with different numbers:
   ```bash
   python -m physics_agent.generate_problem_set_cli --n-per-domain 5 --out data/problem_sets/expanded_set.json
   python -m physics_agent.problem_set_cli data/problem_sets/expanded_set.json
   ```
   This doesn't require any accumulated weak-area data first (unlike
   `curriculum_cli.py`, which targets a *specific, measured* weakness) —
   it works against a completely empty `memory/`. Generation and solving
   are two separate steps on purpose: generate once, keep the JSON file,
   re-run `problem_set_cli.py` against it as many times as you want
   without regenerating.

2. **Parametrize existing problems by hand.** Every problem in
   `intro_physics_set.json` has explicit numeric values — copy one, change
   the numbers, give it a new `id`. Zero LLM calls, zero risk of a
   malformed or physically nonsensical generated problem, but it only
   exercises the same *scenarios* repeatedly, not new ones — useful for
   building up Stage 7's per-domain sample counts quickly, less useful for
   finding new failure modes the way the electron problem did.

3. **Adapt problems from an existing textbook or problem bank** (e.g. an
   OpenStax physics text, or a published problem set) as inspiration,
   rewritten in your own words rather than copied — the same "ground it,
   don't reproduce it" stance this system already takes with
   `literature_search`. This is the slowest option but gives you problems
   whose difficulty and correctness someone else has already vetted,
   which neither of the other two options guarantees (a generated or
   hand-parametrized problem's correctness rests entirely on your read of
   it, or the LLM's).

Whichever way you add problems, keep the same `{"id", "domain_hint",
"problem_text"}` shape so `problem_set_cli.py` and `inspect_trace_cli.py`
both work with them unchanged.

**Gate thresholds, for reference** — these are what actually determine
whether you have "enough" problems, not any specific total count:

| Policy | Threshold | Scope |
|---|---|---|
| `ToolSelectionPolicy` | 5 uses | per (domain, tool) pair |
| `VerificationDepthPolicy` | 5 traces | per domain |
| `flag_declining_strategies` | 5 uses | per (domain, error_type, strategy) |
| `ProceduralMemory.best_strategy_for` | 3 uses | per (domain, error_type, strategy) |

The 206-problem set (~14 per domain) clears the per-domain gates on the
first full pass; the per-`(domain, tool)` and per-`(domain, error_type,
strategy)` gates depend on how consistently the same tool or correction
strategy actually gets used within a domain, which varies problem to
problem — not something a flat per-domain count guarantees by itself.

## Fixes from real-world testing against LM Studio

That testing happened, and it surfaced two real bugs worth documenting —
both found via `inspect_trace_cli.py` on an actual unresolved trace, not
in any unit test fixture.

**The problem that exposed them:** "An electron (rest mass 9.11e-31 kg)
moves at 0.8c. Find its relativistic kinetic energy." ended
`unresolved_max_revisions` after all 3 revision attempts — despite the
model correctly deriving `KE = (γ-1)mc² = 5.466×10⁻¹⁴ J` in every single
round.

**Bug 1 — the Physics Check's knowledge-graph sub-check couldn't tell
"retrieved" from "used."** `retrieved_knowledge` returns the top-k
keyword matches for a query, which pulled in both the correct
relativistic formula (`rel-001`) *and* the classical one (`eng-001`,
`requires_assumption: non_relativistic`) — the model correctly used only
the former, but the check flagged the latter's presence as a violation
regardless, and no revision could ever fix a failure that wasn't actually
in the solution. **Fix:** `PhysicsCheck`'s knowledge-graph sub-check
(`self_eval/physics_check.py`) now checks whether a valid,
graph-connected alternative was *also* retrieved (via
`KnowledgeGraph.neighbors`) before flagging a violation — `rel-001` being
a `special_case_of` `eng-001` and itself passing validity is enough to
suppress the flag. A violation with no valid alternative present is still
flagged exactly as before (verified directly: retrieving `eng-001` alone
still fails the check; retrieving it alongside `rel-001` now passes, with
a transparent note explaining why).

**Bug 2 — `symbolic_math` couldn't handle a model that plugs in numbers
directly.** The tool call itself failed
(`"SymPy found no solutions for KE..."`) because the model gave a fully
numeric expression with `solve_for: "KE"`, even though `KE` never appears
as a free symbol anywhere in it — there was nothing to *solve*, only
something to *evaluate*, and the tool wasn't built for that case. The
model's synthesis step then did the arithmetic by hand in prose instead,
meaning the actual computation happened completely outside any tool,
invisible to `MathCheck`. **Fix:** `SymbolicMathTool`
(`tools/symbolic_math.py`) now detects when `solve_for` isn't a free
symbol and the expression has no free symbols left at all, and evaluates
it directly rather than erroring — confirmed to reproduce the exact
value from the real trace (5.466×10⁻¹⁴ J) instead of failing. An actual
`Eq(...)` with no matching free symbol still raises, on the reasoning
that an equation implies a genuine solve was intended, unlike bare
arithmetic — that ambiguity is different enough to keep erroring on.

Both fixes were verified together against the unmodified real seed data
and knowledge graph (not synthetic fixtures): the exact retrieval,
formula, and domain tags from the failing trace now produce a passing
Physics Check on the first attempt. This problem would no longer need
even one revision, let alone exhaust all 3 and still fail.

**Bug 3 — the shared JSON parser choked on LaTeX.** Bulk-generating
practice problems with `generate_problem_set_cli.py` crashed with
`Invalid \escape: line 1 column 240` on an electromagnetism problem the
model wrote using LaTeX math notation (`$R = 5.0\text{ mm}$`,
`$r \le R$`, `$\mu_0$`) directly inside a JSON string, with the
backslashes left unescaped. Some of these (`\l`, `\m`, `\p`, `\c`) aren't
valid JSON escapes at all, so `json.loads` raised outright; others are
worse — `\t` *is* a valid JSON escape (tab), so `\text{mm}` would have
"succeeded" silently as a tab character followed by garbled literal text
`ext{mm}`, with no error at all. **Fix:** `json_utils.extract_json` (the
one function every JSON-producing component in this system already
shares — planner, orchestrator, all three LLM-based self-eval checks,
and the problem generator) now sanitizes any backslash that isn't
unambiguously a real JSON escape (`\"`, `\\`, `\/`, `\u...`) before
parsing, converting stray ones into literal, parseable text instead of
crashing or silently corrupting content. `ProblemGenerator`'s system
prompt was also strengthened to ask for plain-ASCII math notation instead
of LaTeX in the first place — prevention alongside the parsing fix, not
instead of it, since prompt compliance alone isn't reliable. Verified
against the exact real failing string (reproduced in
`tests/test_json_utils.py`), plus adversarial cases the fix specifically
had to get right without introducing a new bug: Greek-letter LaTeX
commands that start with a letter that's *also* a valid JSON escape code
(`\rho`, `\tau`, `\nabla`, `\beta`, `\frac` — none of these may be
misread as `\r`/`\t`/`\n`/`\b`/`\f` followed by garbled text), and
genuine legitimate escapes (`\n`, `\t`, `\"`, `\\`, `\uXXXX`) continuing
to parse correctly when a model actually means them. One known, accepted
limitation: because LaTeX commands and genuine deliberate control
characters can start identically (both are backslash + letter,
indistinguishable from the character alone), a model deliberately writing
`\n` inside a short JSON string field would now survive as a literal
2-character sequence rather than an actual newline — a reasonable trade
given this system's fields are short, single-paragraph prose where that's
rare, versus the alternative of continuing to crash or corrupt on the far
more common case of physics LaTeX.

**Bug 4 — one bad LLM call crashed the entire generation batch.** The 5th
quantum-mechanics problem in the same run hit `openai.BadRequestError`
("the model produced output that does not match the expected...
format") — a raw server-side failure from the local inference engine
itself, not something in this codebase's control. The real bug was that
nothing caught it: `ProblemGenerator.generate()`'s retry loop only caught
`(ValueError, json.JSONDecodeError)` around the *parsing* step, so an
exception from the LLM call itself propagated straight through, uncaught,
and killed the whole multi-problem, multi-domain batch script over a
single flaky call. **Fix:** the underlying `self.llm.chat(...)` call is
now wrapped in its own try/except inside the retry loop — a raised
exception is treated the same as an unparseable response (worth a retry
with the identical request), and if it keeps failing, surfaces as the
same plain `ValueError` a JSON-parsing failure already would, which
`generate_for_domains()` already knew how to catch and skip past.
Verified with a reproduction of the exact scenario (an LLM client that
raises on its first call and succeeds on retry, and separately, one that
never succeeds) confirming the batch continues past the failure instead
of crashing, and that the original exception type never propagates out of
`generate()`.

**Bug 5 — a different local model (Qwen 3.6 35B A3B) appeared to hang
indefinitely with no output at all.** Unlike Bugs 3-4, this one couldn't
be conclusively root-caused remotely — the most likely explanation is a
"thinking"/reasoning-tuned model producing a long chain-of-thought before
any visible output, or a model that simply never emits a stop token in
that specific quantization. Either way, the same underlying gap in this
codebase made it worse: `LLMClient` set no request timeout and no
`max_tokens` cap on any call. Chat completions here are non-streaming, so
nothing returns until the entire response — including any hidden
reasoning — finishes; with nothing to cut a stuck or slow generation off,
that's indistinguishable from hanging forever, whatever the actual cause.
**Fix:** `LLMClient` now sets both by default (120s timeout, 2048 max
tokens, both configurable via `Config`/env vars:
`LM_STUDIO_TIMEOUT_SECONDS`, `LM_STUDIO_MAX_TOKENS`). This doesn't fix
whatever the model itself was doing — it can't, from here — but it
converts "hangs forever with zero recovery" into "fails after a bounded
wait with a clear, catchable exception," which the retry/skip handling
already built for Bug 4 then takes over from there. Genuinely worth
knowing before relying on this: if you're using a model that legitimately
needs a long reasoning trace before it can answer, raising both values is
the right move, not evidence the fix is wrong — the trade-off is longer
waits, and a `max_tokens` cut set too low will make that specific model's
calls fail cleanly every time rather than hang, which is strictly better
but still means "increase the cap" is the actual fix needed for that model.

**Bug 6 — bulk problem generation kept failing with an empty model
response, and raising `max_tokens` alone made it worse, not better.**
After Bug 5's fix, a real batch run against Gemma showed no crashes but a
substantial fraction of generations failing with
`No JSON object found in model output: ''` — a genuinely empty response,
not malformed JSON. Raising `max_tokens` from 2048 to 8192 didn't reduce
how often this happened; a comparison run with the higher cap actually
succeeded on *fewer* problems (44 vs. 52). That negative result rules out
simple token starvation and points instead to a "thinking"-style model
spending an unbounded, prompt-dependent share of its budget on an internal
reasoning phase before ever starting the visible answer — a bigger cap
just gives it more room to think longer, not proportionally more room to
answer. **Fix:** rather than a bigger token budget, `ProblemGenerator`
now retries at a **lower temperature** after any failed attempt (empty
response, malformed JSON, or a raw call exception) — high temperature
(0.9, deliberately set for generation diversity) is exactly the setting
most likely to produce a degenerate completion, and backing off to a
conservative `retry_temperature` (default 0.3, configurable) trades some
diversity specifically to prioritize getting a usable response at all.
Empty responses also now get their own explicit, readable error message
instead of inheriting `extract_json`'s generic "no JSON object found."
Two smaller companion changes: `generate_problem_set_cli.py`'s default
retry budget went from 1 to 2 attempts beyond the first (configurable via
`--max-retries`), since this looks like a stochastic per-call failure
where more attempts compounds toward success; and the "avoid duplicating"
list is now capped to the most recent 3 prior problems rather than every
one generated so far in a domain, removing unbounded prompt growth as a
plausible contributor to later problems in a batch failing more often
than earlier ones. None of this can fix whatever a specific model is
actually doing internally — that's genuinely outside this codebase's
control — but it gives failures more chances to resolve themselves via a
lever (temperature) that's directly relevant to *this* failure mode,
unlike the token budget, which the data showed wasn't.

**Bug 7 — a self-inflicted regression: `MathCheck` was flagging correct
answers as wrong, essentially every time `SymbolicMathTool`'s
direct-evaluation fallback (Bug 2's fix) fired.** Found by analyzing a
real accumulated `memory/` directory from an actual run (183 solved
traces) rather than a single failing case. The mechanism: when
`solve_for` isn't actually a free symbol in the expression — e.g. a model
computing `m * (v_f - v_i)` and labeling the result `delta_p`, which never
appears in that expression at all — `SymbolicMathTool` correctly falls
back to evaluating the expression directly. But `MathCheck`'s
re-substitution verification was never updated to recognize that same
shape: it substitutes the reported solution back in for a symbol that
isn't present, which SymPy silently treats as a no-op, so the "residual"
it computes is just the original expression's own unchanged value — a
number that's essentially never zero. **This made `MathCheck` fail on
every direct evaluation, regardless of whether the answer was correct.**

The scale of it, measured directly against the real data rather than
estimated: the fallback fired in **199 rounds across 183 traces**, and
**190 of those (95.5%) were incorrectly flagged as a math failure**. Of
the 33 traces whose *final* recorded failure included "math," **all 33
(100%) would pass with the fix** — meaning this dataset contains zero
confirmed genuine algebra errors caught by this check; every single one
was this false positive. **10 problems were reported
`unresolved_max_revisions`** — all 3 revisions burned, marked as failed —
**where math was the only thing "wrong,"** meaning those were very likely
solved correctly on the first attempt and the agent wasted its entire
revision budget chasing a bug that didn't exist. The damage compounds
downstream: semantic memory's per-fact confidence, which updates via
`success = not trace.checks_failed`, was punished for facts that had
nothing to do with the failure — `kin-001` and `grav-001` (both dragged
down to ~0.4-0.5 in the uploaded data) were hit 5 and 2 times respectively
by traces whose *only* failure was this phantom check. Every downstream
Stage 7 statistic that reads from `ProceduralMemory` or `ErrorMemory` —
`rederive_math`'s success rate, `algebra_error`'s recorded frequency — was
learning from a signal that was, in this dataset, never actually real.

**Fix:** `MathCheck._verify_solutions` now checks the same condition
`SymbolicMathTool` itself uses before falling back to direct evaluation
(non-`Eq` expression, `solve_for` absent from its free symbols) and skips
re-substitution verification entirely in that case, since there's no
equation constraint to check — the reported value simply *is* the
expression's value by definition. Verified directly against the exact
real failing case from the uploaded data (`Δp = m(v_f - v_i) = -12.75`,
correctly computed, previously always failed) and confirmed the fix
doesn't over-apply: a genuine equation where the solve-for variable *does*
appear (including one using the same variable names, to make sure the
skip is checking free-symbol presence and not just guessing from
variable names) still goes through full verification and still correctly
fails on a wrong solution.

**Bug 8 — synthesis had no retry on an empty LLM response, and the
correction strategy for it didn't change anything about the retry.**
Found the same way as Bug 7: analyzing a real accumulated `memory/`
directory (205 solved traces) rather than a single failing case. Of the
19 traces that ended up `unresolved_max_revisions`, **13 (68%) had a
completely empty `trace.initial_solution`** — not a wrong answer, an
empty string. `Logic Check` correctly caught this every time ("No initial
solution provided to evaluate"), and `error_taxonomy` correctly routed it
to `resynthesize` — but `_synthesize_solution` was a single, bare LLM
call with no retry logic at all, so `resynthesize` just called that same
un-retried function again, with nothing different about the second (or
third) attempt to make another empty response less likely.
`ProceduralMemory` shows the scale: `resynthesize` sat at a **flat 0%
success rate** across statistical-mechanics (0/14 uses), quantum-mechanics
(0/9), and five other domain combinations — not a verification bug like
Bug 7, but a real generation gap the verification layer was correctly,
repeatedly, and unproductively catching.

The fix already existed elsewhere in this codebase for the same failure
mode: `ProblemGenerator` (used by `curriculum_cli.py` /
`generate_problem_set_cli.py`) already retries on an empty response with a
lower temperature, built for exactly this "model returns empty" pattern
during problem generation — it just was never ported to the solving
path's synthesis step. **Fix:** `_synthesize_solution` now retries (once,
by default, matching `ProblemGenerator`'s own default) at a lower
temperature (0.1) on an empty or whitespace-only response, and raises a
clear `RuntimeError` if every attempt comes back empty, rather than
silently accepting `""` and letting three revision rounds discover that
slowly. Verified against the real data: retroactively applying the
generalized escalation check below to those same 19 unresolved traces,
**19/19** would now stop early instead of exhausting their revision
budget on a fix that was never going to work.

**Bug 9 — the escalation path (added as a Safety Rail, see below) was
scoped too narrowly to catch its own motivating case.** Built to stop a
trace from repeating a corrective action that had already failed —
originally only watched for `escalate_verification` specifically
repeating. Bug 8's data showed the far more common version of the same
shape happening on a different strategy entirely (`resynthesize`,
sitting at 0% success across several domains) without ever tripping it.
**Fix:** generalized to any strategy — if the one about to be applied was
already tried earlier in the same trace and every prior use failed to
resolve anything, escalate instead of repeating it; a strategy that
succeeded once and gets legitimately reused later for a new issue isn't
penalized. See "An escalation path distinct from `max_revisions`" above
for the full mechanism and the 19/19 retroactive result.

**If you have your own accumulated `memory/` directory from before this
fix, it's contaminated and worth discarding rather than continuing to
build on.** Given the 100% false-positive rate measured above, there's no
way to cleanly separate genuine signal from bug artifacts in
`error_memory.json`'s `algebra_error` entries, `procedural.json`'s
`rederive_math` success rates, or the confidence values in
`semantic_seed.json` for any fact that ever got associated with a math
failure — all of it should be treated as unreliable. `episodic.jsonl`'s
individual traces are still an accurate record of what actually happened
(the trace itself isn't wrong, just the check's verdict baked into it),
so it's fine to keep for reference, but Stage 7's policies would be
learning from corrupted history if pointed at the derived memory files
without resetting them first.

Let me know if you'd like to do that further testing round, revisit
anything in Stages 1-8, or take the project somewhere the original
roadmap didn't anticipate.
