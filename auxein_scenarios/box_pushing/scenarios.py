"""The scenario generator (design doc §4): seeded layouts as Auxein `ScenarioSet`s, with difficulty as a parameter.

A scenario is drawn from its seed: obstacles (rejected and redrawn until the arena is connected and the box can reach the target),
the box's pose, the target, and the agents' starting poses. Builds the mixed-difficulty training set and the held-out sets at full
difficulty, never seen in training.
"""
