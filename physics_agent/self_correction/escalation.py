"""
Escalation Signals (Safety Rails).

The design doc's ask: "An escalation path, not just a revision cap.
max_revisions already stops runaway correction loops -- but 'stop and mark
unresolved' and 'stop and ask a person' are different outcomes... at least
for cases like unresolved_max_revisions recurring heavily in one domain."

Two distinct signals, both surfaced here:

1. Per-problem escalations that already happened. self_correction/engine.py
   now sets resolution_status = "escalated_for_human_review" when a
   trace's confidence stayed low even after an independent verification
   attempt (see engine.py's docstring) -- this module just collects and
   groups those by domain so a person reviewing the system sees them as a
   set, not one at a time buried in individual traces.

2. Domains where unresolved_max_revisions recurs heavily -- the design
   doc's explicit example. A single unresolved trace is normal (some
   problems are genuinely hard); the same domain hitting the revision cap
   over and over is a pattern, and curriculum_signals.weak_areas() already
   surfaces exactly this count as a "practice more" signal. This module
   reuses the same count for a different purpose: past a threshold, it's
   not just something to practice, it's something to tell a person about.

Like check_value.py, pruning.py, and curriculum_signals.py before it: this
module flags, it does not act. Nothing here pauses the pipeline, blocks a
curriculum round, or notifies anyone directly -- see meta_report.py for
where these flags actually get surfaced to a person, and canary_cli.py's
exit-code convention for the shape a future automated response might take.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from ..trace import EpisodicMemory

# How many unresolved_max_revisions traces in one domain before it's a
# recurring pattern worth a person's attention, not just curriculum
# material. Matches pruning.py's MIN_USES_TO_FLAG in spirit: small enough
# to catch a real problem while the sample is still fresh, but above the
# range where "one or two hard problems" would look the same as a genuine
# systematic gap.
DEFAULT_MIN_RECURRING_UNRESOLVED = 5


def detect_escalations(
    episodic_memory: EpisodicMemory,
    min_recurring_unresolved: int = DEFAULT_MIN_RECURRING_UNRESOLVED,
) -> List[Dict[str, Any]]:
    escalations: List[Dict[str, Any]] = []

    escalated_traces = episodic_memory.query_by_resolution_status("escalated_for_human_review")
    by_domain: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for t in escalated_traces:
        by_domain[tuple(sorted(t.domain_tags))].append(t.problem_id)
    for domain_tags, problem_ids in by_domain.items():
        escalations.append(
            {
                "source": "escalated_for_human_review",
                "domain_tags": list(domain_tags),
                "problem_ids": problem_ids,
                "reason": (
                    f"{len(problem_ids)} problem(s) escalated after confidence stayed low "
                    "despite an independent verification attempt"
                ),
                "weight": len(problem_ids),
            }
        )

    unresolved = episodic_memory.query_by_resolution_status("unresolved_max_revisions")
    domain_problem_ids: Dict[str, List[str]] = defaultdict(list)
    for t in unresolved:
        for tag in t.domain_tags:
            domain_problem_ids[tag].append(t.problem_id)
    for tag, problem_ids in domain_problem_ids.items():
        if len(problem_ids) >= min_recurring_unresolved:
            escalations.append(
                {
                    "source": "recurring_unresolved",
                    "domain_tags": [tag],
                    "problem_ids": problem_ids,
                    "reason": (
                        f"{len(problem_ids)} problem(s) tagged '{tag}' hit the revision cap "
                        "without resolving -- recurring, not a one-off"
                    ),
                    "weight": len(problem_ids),
                }
            )

    escalations.sort(key=lambda e: e["weight"], reverse=True)
    return escalations
