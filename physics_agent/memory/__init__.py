"""
Memory Architecture (Stage 5).

Four memory types, matching the design doc:
  - Episodic  (physics_agent.trace.EpisodicMemory, built in Stage 1)
  - Semantic  (physics_agent.retrieval.SemanticStore, built in Stage 1,
               extended here with Stage 5's record_outcome)
  - Procedural (this package: procedural.py)
  - Error      (this package: error_memory.py)

consolidator.py is the write side that ties all four together after a
trace finishes Stage 4 -- the "Memory + Knowledge Update" box in the
architecture diagram, and the point where "solving" (Stages 1-4) hands off
to "learning" (this stage and beyond).
"""
