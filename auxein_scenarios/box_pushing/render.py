"""Videos and charts (design doc §7), rendered offline from a run's recording, never during training.

Milestone champions (log-spaced generations, skipping unchanged champions) from `RunReader.best_up_to`, replayed on fixed showcase
scenarios to MP4 clips, a montage of them with the generation and a fitness curve, and the fitness-over-generations chart from
`RunReader.generation_history`.
"""
