"""
Meta-Learning / Adaptive Adjustments (Stage 7).

This is the "outer loop" from the design doc: unlike Stages 1-6, which all
run once per problem, everything in this package reads *accumulated*
memory (episodic traces, procedural strategy stats, error signatures, the
knowledge graph) and either (a) computes a policy that changes future
per-problem behavior, or (b) produces a report meant for a human or a
later stage to act on.

What actually changes future behavior (wired into Stages 2/3/4 in cli.py):
  - tool_policy.ToolSelectionPolicy       -> filters ToolOrchestrator's
                                              offered tools per domain
  - verification_depth.VerificationDepthPolicy -> raises ConfidenceCheck's
                                              effective threshold per domain
  - strategy_override.StrategyOverridePolicy -> replaces error_taxonomy's
                                              fixed default corrective
                                              strategy with whatever
                                              procedural memory has found
                                              to actually work best for a
                                              given (domain, error_type),
                                              once there's enough data

What only reports, deliberately not acting automatically (see each
module's docstring for why):
  - check_value.compute_check_value_report
  - anomaly.detect_check_value_anomalies
  - pruning.flag_declining_strategies
  - curriculum_signals.weak_areas          (feeds a future Stage 8)

report.py ties the reporting half together into one call for a periodic
"how is the agent doing" review -- see physics_agent/meta_report.py for a
CLI entry point that prints it.
"""
