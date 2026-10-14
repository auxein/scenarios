"""The penalty (spring-damper) contact model (design doc §2): forces and torques from overlaps, for every contact pair at once.

The pairs: disc-disc (agents among themselves, agents against circular obstacles), disc-rectangle (agents against rectangular
obstacles), disc-oriented box (agents against the box), oriented box-circle, oriented box-axis-aligned rectangle (the hardest one)
and everything against the walls. Penalty contacts are used because they vectorise cleanly; their stiffness bounds the time step.
"""
