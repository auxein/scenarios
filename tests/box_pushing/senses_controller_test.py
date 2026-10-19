"""The senses-only controller: a feasibility check that the observation carries what the task needs."""

import numpy as np
from auxein.backend import Backend
from auxein.environments import BatchPolicy
from auxein.random import RandomStream

from auxein_scenarios.box_pushing import scenarios
from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.environment import BoxPushingWorld
from auxein_scenarios.box_pushing.scripted import SensesController
from tests.box_pushing.support import host

PARAMS = WorldParams()


def test_it_is_a_batch_policy_on_observations_alone() -> None:
    controller = SensesController()
    assert isinstance(controller, BatchPolicy)
    observations = np.random.default_rng(0).uniform(-1.0, 1.0, (2, 3, 5, BoxPushingWorld().observation_size))
    actions = np.asarray(controller.act_batch(observations))
    assert actions.shape == (2, 3, 5, 2) and np.all(np.abs(actions) <= 1.0)


def test_it_delivers_easy_scenarios_of_the_default_training_set(corner_backend: Backend) -> None:
    """The default training set's 8 easy scenarios (seed 1, 20 agents): it delivers 6 of them within the step limit on numpy
    float64. At least one is the bar: if a memoryless controller can deliver from the 43 inputs, evolution can too."""
    easy = scenarios.training_set(20, PARAMS, seed=1, mix=((scenarios.EASY, 8),))
    result = BoxPushingWorld(PARAMS).run_batch(
        SensesController(1, corner_backend), tuple(easy), RandomStream(np.random.SeedSequence(0), corner_backend)
    )
    delivered = host(corner_backend, result.measurements["delivered"])[0]
    assert delivered.sum() >= 1
    steps = host(corner_backend, result.measurements["steps"])[0]
    assert np.all(steps[delivered > 0.5] <= PARAMS.simulation.max_steps)
