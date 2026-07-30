"""
Autonomous Curriculum (Stage 8).

Closes the loop the original design doc described: takes Stage 7's ranked
weak-area signals (curriculum_signals.weak_areas) and generates new
practice problems targeting them, solves those problems through the exact
same Stage 1-7 pipeline every other problem goes through, and measures
whether the practice actually moved the underlying metric that was
flagged -- honestly, including the possibility that it didn't.

  - problem_generator.py: ProblemGenerator, optionally grounded by a real
    literature_search result (never reproducing source text verbatim).
  - curriculum_runner.py: CurriculumRunner ties generation + solving +
    before/after measurement together; CurriculumLog persists round results.
  - benchmark.py: summarize() aggregates many rounds into an honest
    improved/regressed/unchanged breakdown per signal source.
"""
