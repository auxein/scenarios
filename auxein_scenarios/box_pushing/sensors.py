"""What each agent senses, local and in its own frame (design doc §3): the observation vector fed to the shared network.

The target (distance and bearing), range rays (normalised distance and what was hit), the box cue within the sensing range,
teammate sector densities and proprioception. The layout and the size of the observation are fixed here, whatever the team size,
so the same brain works for 20 or 100 agents.
"""
