"""
Check-Value Anomaly Detection (Stage 7 -> Safety Rails).

The design doc's own motivating case: "MathCheck's catch rate just went
from ~5% to ~35% -- that's exactly the kind of thing that should trigger a
pause-and-alert, not silently accumulate into memory." check_value.py
already computes catch rate; this module is what was missing -- comparing
that number against its OWN history rather than reporting it in isolation
every time. The real MathCheck bug (see math_check.py's docstring) had
this exact shape: 190 of 199 direct-evaluation traces got newly,
incorrectly flagged, which would show up here as MathCheck's catch rate
jumping sharply the moment the corrupted tool-output pattern started
appearing in episodic memory -- catchable long before someone happened to
inspect a trace by hand.

Deliberately narrow, matching check_value.py's own stated philosophy for
the same reason: this module flags a catch-rate anomaly. It does not
diagnose the cause (a real regression vs. a legitimate shift in which
problem types are being solved), silence a check, or block anything.
Escalation -- what happens when a flag like this keeps firing -- is a
separate concern (see self_correction/escalation.py).
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..trace import EpisodicMemory
from .check_value import compute_check_value_for_traces, traces_with_checks_run

# How many of the most recent traces count as "recent" for comparison.
# Matches curriculum's n_problems-style defaults: small enough to react to
# a fresh regression without waiting on a huge sample, large enough that a
# couple of unlucky problems don't look like a trend.
DEFAULT_RECENT_WINDOW = 20

# Baseline needs at least this many traces before the recent window for a
# comparison to mean anything -- an early-history comparison (5 baseline
# traces vs 20 recent) would be noise dressed up as a signal.
DEFAULT_MIN_BASELINE_N = 20

# Flag when a check's catch rate moves by at least this many percentage
# points between baseline and recent. Absolute, not relative: relative
# swings are wildly unstable near a 0% baseline (going from 1 failure in
# 20 to 2 failures in 20 traces is "100% relative increase" but not
# actually informative), while an instrument as coarse as a boolean pass/
# fail check moving by 15+ points in either direction is a large enough
# shift to be worth a human glance regardless of where it started.
DEFAULT_ABSOLUTE_THRESHOLD = 0.15


def detect_check_value_anomalies(
    episodic_memory: EpisodicMemory,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    min_baseline_n: int = DEFAULT_MIN_BASELINE_N,
    absolute_threshold: float = DEFAULT_ABSOLUTE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Returns:
        {
          "status": "ok" | "insufficient_data",
          "n_baseline": int, "n_recent": int,
          "flags": [
            {"check": str, "direction": "jump"|"collapse",
             "baseline_catch_rate": float, "recent_catch_rate": float,
             "delta": float, "reason": str},
            ...
          ]
        }

    "insufficient_data" (fewer than min_baseline_n traces exist before the
    most recent recent_window) returns an empty flags list rather than
    guessing from too little history -- consistent with weak_areas() and
    flag_declining_strategies() both requiring a minimum sample before
    flagging anything.

    Traces are taken in EpisodicMemory.read_all() order, which is write
    (i.e. chronological) order -- see EpisodicMemory.write, an append-only
    log -- so "the most recent recent_window traces" is simply the tail of
    that list; no separate timestamp sort is needed or performed.
    """
    traces = traces_with_checks_run(episodic_memory)

    if len(traces) < recent_window + min_baseline_n:
        return {
            "status": "insufficient_data",
            "n_baseline": max(0, len(traces) - recent_window),
            "n_recent": min(len(traces), recent_window),
            "flags": [],
        }

    baseline_traces = traces[:-recent_window]
    recent_traces = traces[-recent_window:]

    baseline_report = compute_check_value_for_traces(baseline_traces)
    recent_report = compute_check_value_for_traces(recent_traces)

    check_names = sorted(set(baseline_report) | set(recent_report))
    flags: List[Dict[str, Any]] = []

    for name in check_names:
        baseline_rate = baseline_report.get(name, {}).get("catch_rate", 0.0)
        recent_rate = recent_report.get(name, {}).get("catch_rate", 0.0)
        delta = recent_rate - baseline_rate

        if abs(delta) < absolute_threshold:
            continue

        direction = "jump" if delta > 0 else "collapse"
        flags.append(
            {
                "check": name,
                "direction": direction,
                "baseline_catch_rate": baseline_rate,
                "recent_catch_rate": recent_rate,
                "delta": delta,
                "reason": (
                    f"'{name}' catch rate {direction}ed from {baseline_rate:.1%} "
                    f"(over {len(baseline_traces)} prior traces) to {recent_rate:.1%} "
                    f"(over {len(recent_traces)} recent traces)"
                ),
            }
        )

    flags.sort(key=lambda f: abs(f["delta"]), reverse=True)

    return {
        "status": "ok",
        "n_baseline": len(baseline_traces),
        "n_recent": len(recent_traces),
        "flags": flags,
    }
