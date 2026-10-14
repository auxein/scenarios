"""Training (design doc §6): builds the policy, the environment, the scenario sets and the evaluator from a config, and runs Auxein.

`GeneticAlgorithm` is the main strategy and `PycmaStrategy` the comparison; generation delivery, deterministic mode, recording on,
progress printed with `auxein.callbacks.print_progress`, and an optional early stop on the held-out success rate.
"""
