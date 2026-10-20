"""The senses-only controller: a feasibility check that the observation carries what the task needs, and the baseline every
run is judged against, which the box's course leaves unchanged."""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest
from auxein.backend import Backend
from auxein.environments import BatchPolicy
from auxein.random import RandomStream

from auxein_scenarios.box_pushing import evaluation, scenarios, train
from auxein_scenarios.box_pushing.config import TrainingConfig, WorldParams
from auxein_scenarios.box_pushing.environment import BoxPushingWorld
from auxein_scenarios.box_pushing.scripted import SensesController
from auxein_scenarios.box_pushing.sensors import observation_layout, observe
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


# --- unchanged by the box's course: the baseline every run is judged against ---

PIN = json.loads((Path(__file__).parent / "data" / "controller_pin.json").read_text())
MACHINE = f"{sys.platform}-{platform.machine().lower().replace('amd64', 'x86_64').replace('aarch64', 'arm64')}"
NUMPY = Backend("numpy", "cpu", "float64")


def _states() -> tuple[list[object], object, TrainingConfig]:
    """The first 4 held-out layouts at the start and after 49 steps of the controller."""
    config = TrainingConfig()
    held = list(train.held_out_set(config))[:4]
    world = BoxPushingWorld(config.world, config.sensing, steps=50)
    layout = world.layout(held, NUMPY)
    states: list[object] = []
    controller = SensesController(1, NUMPY, world.sensing, config.world)
    world.run(
        controller.act_batch, layout, 1, hook=lambda t, state, _: states.append(state) if t in (0, 49) else None, stop_when_done=False
    )
    return states, layout, config


def _without_course(observation: np.ndarray) -> np.ndarray:
    course = observation_layout()["box_course"]
    return np.concatenate([observation[..., : course.start], observation[..., course.stop :]], axis=-1)


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(json.dumps([f"{v:.9g}" for v in values.reshape(-1)]).encode()).hexdigest()


def test_its_inputs_are_what_they_were() -> None:
    """Every part the controller reads is what the code before the course computed: the observation without `box_course` is the
    pinned one (to 9 digits; at step 0 on every platform, after 49 steps where it was pinned)."""
    states, layout, config = _states()
    for name, state in zip(("step 0", "step 49"), states, strict=True):
        if name == "step 49" and MACHINE != PIN["machine"]:
            continue  # 49 steps of contacts: the last bit depends on the platform
        observation = np.asarray(observe(state, layout, config.world, config.sensing), dtype=np.float64)  # type: ignore[arg-type]
        assert _digest(_without_course(observation)) == PIN["observation_sha256_9_digits"][name], name


@pytest.mark.skipif(MACHINE != PIN["machine"], reason="pinned bit for bit on macOS arm64: elsewhere a float's last bit differs")
@pytest.mark.parametrize("which", ["training", "held_out"])
def test_its_measurements_are_what_they_were(which: str) -> None:
    """On the default training set and on held-out seed 2, the controller's measurements are bit for bit those of the code
    before the course (numpy float64): its numbers, against which every run is judged, did not move."""
    config = TrainingConfig()
    chosen = train.training_set(config) if which == "training" else train.held_out_set(config)
    assert chosen.fingerprint == PIN[which]["fingerprint"]
    world = BoxPushingWorld(config.world, config.sensing)
    (measured,) = evaluation.measure(world, SensesController(1, NUMPY, world.sensing, config.world), list(chosen), NUMPY)
    rows = [m for m in measured if m is not None]
    assert len(rows) == len(measured), "no failed episode"
    assert sum(m["delivered"] > 0.5 for m in rows) == PIN[which]["delivered"]
    exact = hashlib.sha256(json.dumps([{k: repr(v) for k, v in sorted(m.items())} for m in rows]).encode()).hexdigest()
    assert exact == PIN[which]["measurements_sha256"]
