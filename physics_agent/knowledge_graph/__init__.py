"""
Knowledge Graph (Stage 6).

Nodes are physics facts, reusing SemanticStore's existing entries (same
ids, same confidence field Stage 5 already keeps updated via
record_outcome) rather than duplicating that data -- this package adds the
*relational* structure on top: typed edges between facts, plus the queries
the design doc calls for (validity checking, low-confidence clustering,
contradiction surfacing).
"""
