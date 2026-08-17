"""
Autonomous Knowledge Growth (Stage 7) -- the last gap named in the
original response: "Every semantic fact and knowledge-graph edge in this
system was written by hand. There's no mechanism for the agent to
notice 'I've now solved 12 problems using a formula that isn't in
semantic memory at all' and propose adding it -- at low initial
confidence, with provenance noting it was self-derived rather than
seeded, refined the same way seeded facts already are via
record_outcome. This is the difference between a system with a fixed
knowledge base and one that actually grows its own."

The signal: self_correction/engine.py's own docstring notes trace.tool_calls
holds only the FINAL round's calls by the time solving finishes -- so the
symbolic_math `expression` string behind a cleanly-resolved trace's answer
is exactly "the derivation that actually passed verification," not a
stale intermediate attempt. When the same expression (whitespace-
normalized) recurs across many independently-solved traces, that's a
formula the agent keeps deriving from scratch instead of recalling --
worth proposing as its own semantic-memory entry.

Deliberately narrow scope, matching this project's other components that
act automatically only past a high bar (strategy_override.py,
verification_depth.py) -- and, unlike those, this one creates new
persistent knowledge rather than choosing among already-vetted options, so
the bar here is "propose," not "silently commit and move on":

  - Matching is a normalized STRING comparison of the tool-call
    expression, not symbolic equivalence checking. "Eq(m*g*h,
    0.5*m*v**2)" and "Eq(v**2, 2*g*h)" are mathematically the same
    physics but track as two separate candidates. Documented limitation,
    not a bug -- the literal repeated-pattern signal is still real value
    without full CAS-level equivalence.
  - The proposed "statement" is built directly and deterministically from
    the tool call itself, not LLM-paraphrased into confident-sounding
    prose nobody has vetted the wording of.
  - No attempt is made to detect whether a seeded fact's hand-written
    prose ("KE = 0.5*m*v^2") already covers the same physics as a
    self-derived expression signature ("Eq(m*g*h, 0.5*m*v**2)") -- the two
    vocabularies essentially never collide as strings, so this can't
    reliably tell. The accepted mitigation: a resulting near-duplicate
    entry is low-cost (a second, low-confidence description of the same
    physics isn't a correctness bug), while duplicate proposals from THIS
    mechanism re-running are prevented via the persisted registry below.
  - Initial confidence (INITIAL_CONFIDENCE) sits below every seeded fact
    in data/semantic_seed.json on purpose -- repeated successful *use* is
    real evidence, but weaker than whatever review a hand-seeded fact
    received. From there it's refined exactly like any other fact:
    SemanticStore.record_outcome, called by MemoryConsolidator on every
    future solve that retrieves it.

Two-step API, mirroring the "compute, then optionally act" split every
other Stage 7 module uses:
  - find_candidate_facts(): read-only, ranks candidates by observation
    count. Nothing here touches SemanticStore.
  - propose_and_add(): commits candidates that clear the bar and haven't
    been proposed before (tracked via ProposedFactsRegistry, so re-running
    this doesn't duplicate the same fact on every call).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from ..retrieval import SemanticStore
from ..trace import EpisodicMemory, Trace

# How many distinct, cleanly-resolved traces need to have used the exact
# same tool-call expression before it's trusted enough to become a
# semantic-memory candidate. The design doc's own example used "12
# problems"; 5 is used here for consistency with this project's other
# minimum-sample bars (StrategyOverridePolicy, escalation.py) rather than
# inventing a new number for the same kind of judgment call.
MIN_OBSERVATIONS_TO_PROPOSE = 5

# Confidence a self-derived fact starts at -- see module docstring.
INITIAL_CONFIDENCE = 0.3

# How many trace ids get embedded in the persisted provenance string
# before truncating with a "+N more" suffix -- the full list is still
# available in find_candidate_facts()'s return value / the scheduler's
# decision log; this only bounds what gets written into the semantic
# store entry itself.
MAX_PROVENANCE_TRACE_IDS = 5

_RESOLVED_STATUSES = {"passed_initial", "resolved_after_revision"}


def _normalize_expression(expression: str) -> str:
    """Whitespace-insensitive signature -- see module docstring's
    documented limitation on what this can and can't match."""
    return "".join(expression.split())


def _final_symbolic_math_calls(trace: Trace) -> List[Dict[str, Any]]:
    """
    trace.tool_calls holds only the FINAL round's calls by the time
    solving finishes (self_correction/engine.py overwrites it each
    revision -- see that module's docstring), so this is already "the
    tool calls behind the answer that actually passed," not a stale
    intermediate attempt from a since-corrected earlier round.
    """
    calls = []
    for tc in trace.tool_calls:
        if tc.tool != "symbolic_math":
            continue
        try:
            output = json.loads(tc.output)
        except json.JSONDecodeError:
            continue
        if "error" in output or "expression" not in output:
            continue
        calls.append(output)
    return calls


def find_candidate_facts(
    episodic_memory: EpisodicMemory,
    min_observations: int = MIN_OBSERVATIONS_TO_PROPOSE,
) -> List[Dict[str, Any]]:
    """
    Returns candidates ranked by observation count (most-repeated first),
    each:
        {signature, expression, solve_for, domain_tags, trace_ids,
         n_observations}
    Read-only -- does not touch SemanticStore. See propose_and_add for the
    step that actually commits any of these.
    """
    groups: Dict[str, Dict[str, Any]] = {}

    for trace in episodic_memory.read_all():
        if trace.resolution_status not in _RESOLVED_STATUSES:
            continue  # only genuinely verified-correct outcomes count
        for call in _final_symbolic_math_calls(trace):
            signature = _normalize_expression(call["expression"])
            group = groups.setdefault(
                signature,
                {
                    "signature": signature,
                    "expression": call["expression"],
                    "solve_for": call.get("solve_for"),
                    "domain_tags": set(),
                    "trace_ids": [],
                },
            )
            group["domain_tags"].update(trace.domain_tags)
            group["trace_ids"].append(trace.problem_id)

    candidates = []
    for group in groups.values():
        trace_ids = sorted(set(group["trace_ids"]))
        n_observations = len(trace_ids)
        if n_observations < min_observations:
            continue
        candidates.append(
            {
                "signature": group["signature"],
                "expression": group["expression"],
                "solve_for": group["solve_for"],
                "domain_tags": sorted(group["domain_tags"]),
                "trace_ids": trace_ids,
                "n_observations": n_observations,
            }
        )

    candidates.sort(key=lambda c: c["n_observations"], reverse=True)
    return candidates


class ProposedFactsRegistry:
    """
    Tiny persisted map of signatures already proposed by this mechanism,
    so re-running propose_and_add doesn't add the same self-derived fact
    twice. Deliberately separate from SemanticStore itself (which has no
    concept of "signature," only ids) rather than trying to detect
    duplicates by scanning existing entries' free-text statements.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", encoding="utf-8") as f:
                self.signature_to_entry_id: Dict[str, str] = json.load(f)
        else:
            self.signature_to_entry_id = {}
            self._persist()

    def already_proposed(self, signature: str) -> bool:
        return signature in self.signature_to_entry_id

    def record(self, signature: str, entry_id: str) -> None:
        self.signature_to_entry_id[signature] = entry_id
        self._persist()

    def _persist(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.signature_to_entry_id, f, indent=2)


def _entry_id_for(signature: str) -> str:
    # Short, stable, deterministic -- the same signature always maps to
    # the same id, so re-deriving it doesn't depend on the registry
    # having already seen it (useful if the registry file is ever lost
    # while the semantic store isn't -- a re-proposal collides on id
    # rather than silently duplicating under a new one).
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
    return f"derived-{digest}"


def propose_and_add(
    semantic_store: SemanticStore,
    registry: ProposedFactsRegistry,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Commits each candidate not already proposed before as a new,
    low-confidence, clearly self-labeled semantic-memory entry. Returns
    the newly-added entries (empty if every candidate had already been
    proposed in an earlier run).
    """
    added = []
    for candidate in candidates:
        if registry.already_proposed(candidate["signature"]):
            continue

        entry_id = _entry_id_for(candidate["signature"])
        statement = candidate["expression"]
        if candidate.get("solve_for"):
            statement += f", solved for {candidate['solve_for']}"

        trace_ids = candidate["trace_ids"]
        shown_ids = trace_ids[:MAX_PROVENANCE_TRACE_IDS]
        trace_id_note = ", ".join(shown_ids)
        if len(trace_ids) > MAX_PROVENANCE_TRACE_IDS:
            trace_id_note += f", +{len(trace_ids) - MAX_PROVENANCE_TRACE_IDS} more"

        conditions = (
            "Unverified: self-derived from repeated tool use across "
            f"{candidate['n_observations']} independently-solved problem(s); "
            "conditions of validity have not been characterized by a person."
        )
        provenance = (
            f"self_derived (proposed by knowledge_growth after "
            f"{candidate['n_observations']} successful uses; trace_ids=[{trace_id_note}])"
        )

        semantic_store.add(
            entry_id=entry_id,
            statement=statement,
            conditions=conditions,
            confidence=INITIAL_CONFIDENCE,
            provenance=provenance,
            tags=candidate["domain_tags"],
        )
        registry.record(candidate["signature"], entry_id)
        added.append(
            {
                "entry_id": entry_id,
                "signature": candidate["signature"],
                "statement": statement,
                "n_observations": candidate["n_observations"],
                "domain_tags": candidate["domain_tags"],
            }
        )
    return added
