"""2-D geometry on batched arrays, written against the Array API namespace of its inputs (numpy or torch, any precision, any device).

Rotations, frame changes (world to an agent's own frame), angle wrapping, signed distances and closest points between discs,
axis-aligned rectangles and oriented rectangles, and ray casts against them. Everything is vectorised over leading batch axes
(candidates, scenarios, agents): no Python loops over them, so that one call serves a whole population (design doc §2).
"""
