"""The environment (design doc §2.11): both of Auxein's paths, the layout cache, early stop, the measurements and failures."""

import math
from typing import Any

import numpy as np
import pytest
from auxein.backend import Array, Backend, backend_of
from auxein.core import Status
from auxein.environments import Scenario, ScenarioSet
from auxein.random import RandomStream

from auxein_scenarios.box_pushing import scenarios, scripted
from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.environment import MEASUREMENTS, ROLE, BoxPushingWorld, brain
from auxein_scenarios.box_pushing.scenarios import layout_params
from auxein_scenarios.box_pushing.world import initial_state, simulate
from tests.box_pushing.support import FACING_X, StatePolicy, ZeroPolicy, host, line_behind

PARAMS = WorldParams()
EASY = scenarios.training_set(20, PARAMS, seed=1, mix=((scenarios.EASY, 3), (scenarios.FULL, 1)))
POLICY = brain()


def stream(backend: Backend, seed: int = 0) -> RandomStream:
    return RandomStream(np.random.SeedSequence(seed), backend)


def genomes(backend: Backend, n: int, seed: int = 7) -> Array:
    return POLICY.space().sample_genomes(n, stream(backend, seed), backend)


def single(layout: dict[str, object]) -> ScenarioSet:
    return ScenarioSet.from_params([layout], seed=0, ids=["hand-built"])


def test_run_batch_returns_the_six_measurements_on_the_backend(corner_backend: Backend) -> None:
    world = BoxPushingWorld(steps=20)
    result = world.run_batch(POLICY.decode_batch(genomes(corner_backend, 3)), tuple(EASY), stream(corner_backend))
    assert set(result.measurements) == set(MEASUREMENTS) and not result.failures
    for name, array in result.measurements.items():
        assert tuple(array.shape) == (3, 4), name
        assert backend_of(array) == corner_backend, name
    values = {name: host(corner_backend, array) for name, array in result.measurements.items()}
    assert np.all(values["steps"] == 20) and np.all(values["delivered"] == 0.0)
    assert np.all((values["approach"] >= 0.0) & (values["approach"] <= 1.0))
    assert np.all((values["max_pushers"] >= 0.0) & (values["max_pushers"] <= 20.0))


def test_the_batched_and_per_episode_paths_agree(corner_backend: Backend) -> None:
    world = BoxPushingWorld(steps=60, backend=corner_backend)
    batch = genomes(corner_backend, 3)
    batched = world.run_batch(POLICY.decode_batch(batch), tuple(EASY), stream(corner_backend))
    on_host = corner_backend.to_numpy(batch)
    for row in range(3):
        agent = POLICY.decode(on_host[row])
        for column, scenario in enumerate(EASY):
            episode = world.run_episode({ROLE: agent}, scenario, stream(corner_backend))
            assert episode.status is Status.OK
            for name in MEASUREMENTS:
                expected = float(host(corner_backend, batched.measurements[name])[row, column])
                # the brain's products are grouped differently (one team against the whole batch), a few ulps apart;
                # float32 then carries them through a chaotic contact model, so its tolerance is wider
                assert episode.measurements[name] == pytest.approx(
                    expected, rel=1e-6 if corner_backend.precision == "float64" else 1e-2, abs=1e-6
                ), (row, column, name)


def test_the_layout_is_built_once_for_the_same_scenarios() -> None:
    backend = Backend("numpy", "cpu", "float64")
    world = BoxPushingWorld(steps=2)
    agents = POLICY.decode_batch(genomes(backend, 2))
    world.run_batch(agents, tuple(EASY), stream(backend))
    world.run_batch(agents, tuple(EASY), stream(backend, 1))
    assert world.layouts_built == 1
    # the same ids with other layouts are other scenarios: a training set of another seed
    other = scenarios.training_set(20, PARAMS, seed=2, mix=((scenarios.EASY, 3), (scenarios.FULL, 1)))
    assert other.ids == EASY.ids
    world.run_batch(agents, tuple(other), stream(backend))
    assert world.layouts_built == 2


def test_the_world_ignores_the_agents_stream() -> None:
    backend = Backend("numpy", "cpu", "float64")
    world = BoxPushingWorld(steps=30)
    agents = POLICY.decode_batch(genomes(backend, 2))
    a = world.run_batch(agents, tuple(EASY), stream(backend, 1))
    b = world.run_batch(agents, tuple(EASY), stream(backend, 2))
    for name in MEASUREMENTS:
        assert np.array_equal(host(backend, a.measurements[name]), host(backend, b.measurements[name]))


def test_measurements_of_the_push_controller_match_m1a(corner_backend: Backend) -> None:
    """The M1a push team, run through the environment's loop, delivers what `simulate` delivers, at the same step."""
    generated = scenarios.training_set(20, PARAMS, seed=11, mix=((scenarios.EASY, 6),))
    world = BoxPushingWorld()
    layout = world.layout(tuple(generated), corner_backend)
    reference = simulate(initial_state(layout, 1), layout, PARAMS, scripted.push(), PARAMS.simulation.max_steps)
    policy = StatePolicy(scripted.push(), layout, PARAMS)
    rollout = world.run(policy.act_batch, layout, 1, hook=policy.hook)
    got = {name: host(corner_backend, value)[0] for name, value in rollout.measurements.items()}
    done, steps = host(corner_backend, reference.done)[0] > 0.5, host(corner_backend, reference.steps)[0]
    assert done.any(), "the push team should deliver some easy scenarios"
    np.testing.assert_array_equal(got["delivered"], done)
    np.testing.assert_array_equal(got["steps"], steps)
    target = np.asarray([s.params["target"] for s in generated], dtype=np.float64)
    final = np.hypot(host(corner_backend, reference.box_x)[0] - target[:, 0], host(corner_backend, reference.box_y)[0] - target[:, 1])
    np.testing.assert_allclose(got["final_distance"], final, rtol=1e-5, atol=1e-5)
    assert np.all(got["final_distance"][done] <= PARAMS.simulation.target_radius)
    # max_pushers counts agents touching the box: a delivery needs k agents' thrust, but agents queued in a column push through
    # the ones in front (design doc §2.4), so as few as two may touch it
    assert np.all(got["max_pushers"][done] >= 1.0)


def test_early_stop_once_every_scenario_is_done() -> None:
    backend = Backend("numpy", "cpu", "float64")
    # five agents against a box 1.7 from the target: delivered within a few dozen steps
    near = layout_params([0.0, 0.0, FACING_X], [1.7, 0.0], line_behind(0.0, 5, PARAMS))
    world = BoxPushingWorld()
    layout = world.layout(tuple(single(near)), backend)
    calls: list[int] = []
    rollout = world.run(scripted_constant_policy(backend), layout, 1, hook=lambda t, state, senses: calls.append(t))
    steps = int(host(backend, rollout.measurements["steps"])[0, 0])
    assert host(backend, rollout.measurements["delivered"])[0, 0] == 1.0
    assert steps < 200 and calls[-1] == steps and len(calls) == steps + 1


def scripted_constant_policy(backend: Backend) -> Any:
    """Full thrust ahead for every agent, as an `act` function."""

    def act(observations: Array) -> Array:
        xp = backend.xp
        zero = xp.zeros(observations.shape[:-1], dtype=observations.dtype)
        return xp.stack([zero + 1.0, zero], axis=-1)

    return act


def test_max_pushers_counts_agents_touching_the_box(backend: Backend) -> None:
    team = line_behind(0.0, 3, PARAMS) + [[-5.0, 5.0, 0.0], [5.0, -5.0, 0.0]]
    layouts = single(layout_params([0.0, 0.0, FACING_X], [8.0, 0.0], team))
    world = BoxPushingWorld(steps=5)
    result = world.run_batch(ZeroPolicy(2, backend), tuple(layouts), stream(backend))
    np.testing.assert_array_equal(host(backend, result.measurements["max_pushers"]), [[3.0], [3.0]])


def test_approach_is_the_mean_of_the_closest_gaps(backend: Backend) -> None:
    # agents still (zero actions) at known gaps from the box's long face (x = −0.6 when it faces x): 0, 1, 2.5 and 7
    r = PARAMS.agent.radius
    face = -PARAMS.box.height / 2.0
    gaps = [0.0, 1.0, 2.5, 7.0]
    team = [[face - r - g, 0.0, 0.0] for g in gaps]
    layouts = single(layout_params([0.0, 0.0, FACING_X], [8.0, 0.0], team))
    result = BoxPushingWorld(steps=3).run_batch(ZeroPolicy(1, backend), tuple(layouts), stream(backend))
    expected = np.mean([1.0 - min(1.0, g / 5.0) for g in gaps])
    assert float(host(backend, result.measurements["approach"])[0, 0]) == pytest.approx(expected, abs=1e-5)


class NaNInScenario:
    """A batch policy whose actions are NaN in one scenario: that scenario's state turns non-finite at the first step."""

    def __init__(self, n: int, backend: Backend, scenario: int) -> None:
        self.n, self.backend, self.scenario = n, backend, scenario

    def act_batch(self, observations: Array) -> Array:
        xp = self.backend.xp
        s = int(observations.shape[1])
        poison = xp.asarray([math.nan if i == self.scenario else 0.0 for i in range(s)], dtype=observations.dtype)
        zero = xp.zeros(observations.shape[:-1], dtype=observations.dtype)
        actions = zero + poison[None, :, None]
        return xp.stack([actions, actions], axis=-1)


def test_a_non_finite_state_fails_the_episode_not_the_run(corner_backend: Backend) -> None:
    world = BoxPushingWorld(steps=10)
    result = world.run_batch(NaNInScenario(2, corner_backend, 1), tuple(EASY), stream(corner_backend))
    assert sorted(result.failures) == [(0, 1), (1, 1)]
    failure = result.failures[(0, 1)]
    assert failure.status is Status.FAILED and "step 1" in failure.error and repr(EASY[1].id) in failure.error
    for name, array in result.measurements.items():
        values = host(corner_backend, array)
        assert np.all(np.isfinite(values)), f"{name}: the failed scenario is frozen at its last finite state"
    # the other scenarios ran their full length
    np.testing.assert_array_equal(host(corner_backend, result.measurements["steps"])[:, [0, 2, 3]], 10.0)


def test_a_non_finite_episode_on_the_per_episode_path(corner_backend: Backend) -> None:
    class NaNAgent:
        def act(self, observation: Any) -> Any:
            return np.full((observation.shape[0], 2), np.nan)

    world = BoxPushingWorld(steps=5, backend=corner_backend)
    scenario: Scenario = EASY[0]
    episode = world.run_episode({ROLE: NaNAgent()}, scenario, stream(corner_backend))
    assert episode.status is Status.FAILED and episode.error is not None and "step 1" in episode.error


def test_run_batch_needs_to_know_the_batch() -> None:
    class Opaque:
        def act_batch(self, observations: Array) -> Array:
            return observations

    with pytest.raises(TypeError, match="backend"):
        BoxPushingWorld(steps=1).run_batch(Opaque(), tuple(EASY), stream(Backend()))
