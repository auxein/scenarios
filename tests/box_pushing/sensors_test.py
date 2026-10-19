"""The sensors against hand-built states: each part of the observation, its range, and batch invariance."""

import dataclasses
import math
from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from auxein.backend import Backend

from auxein_scenarios.box_pushing import scenarios, sensors
from auxein_scenarios.box_pushing.config import SensingParams, WorldParams
from auxein_scenarios.box_pushing.scenarios import layout_params
from auxein_scenarios.box_pushing.sensors import HIT_BOX, HIT_NOTHING, HIT_OBSTACLE, observation_layout
from auxein_scenarios.box_pushing.world import State, initial_state
from tests.box_pushing.support import host

PARAMS = WorldParams()
LAYOUT = observation_layout()


def senses(backend: Backend, layouts: Sequence[Mapping[str, object]], **state: object) -> tuple[sensors.Senses, State]:
    """What the agents of hand-built layouts sense, with some state fields overridden (given as nested lists or numbers)."""
    arrays = scenarios.layout_arrays(layouts, backend)
    start = initial_state(arrays, 1)
    changes = {
        name: backend.asarray(np.broadcast_to(np.asarray(value, dtype=np.float64), getattr(start, name).shape).copy())
        for name, value in state.items()
    }
    start = dataclasses.replace(start, **changes)
    return sensors.sense(start, arrays, PARAMS), start


def part(backend: Backend, sensed: sensors.Senses, name: str, agent: int = 0) -> np.ndarray:
    return host(backend, sensed.observation)[0, 0, agent, LAYOUT[name]]


def tolerance(backend: Backend) -> float:
    return 1e-9 if backend.precision == "float64" else 1e-5


def test_the_layout_is_defined_once_with_its_sizes() -> None:
    assert LAYOUT.size == 43
    assert {name: g.stop - g.start for name, g in LAYOUT.groups.items()} == {
        "target": 3,
        "rays": 24,
        "box": 6,
        "teammates": 8,
        "proprioception": 2,
    }
    starts = [p.start for p in LAYOUT.parts.values()]
    stops = [p.stop for p in LAYOUT.parts.values()]
    assert starts[0] == 0 and starts[1:] == stops[:-1], "the parts tile the vector in order"
    assert observation_layout(SensingParams(range=8.0)).size == 43, "the range scales distances, not the vector's size"


def test_target_distance_and_bearing_in_the_agents_frame(backend: Backend) -> None:
    layouts = [layout_params([-6.0, -6.0, 0.0], [5.0, 0.0], [[0.0, 0.0, 0.0], [0.0, 3.0, math.pi / 2], [0.0, -3.0, math.pi]])]
    sensed, _ = senses(backend, layouts)
    diagonal = PARAMS.arena.size * math.sqrt(2.0)
    np.testing.assert_allclose(part(backend, sensed, "target_distance", 0), [5.0 / diagonal], atol=tolerance(backend))
    np.testing.assert_allclose(part(backend, sensed, "target_bearing", 0), [0.0, 1.0], atol=tolerance(backend))
    # agent 1 at (0, 3) faces up: the target's world bearing atan2(−3, 5), less its heading of 90°
    bearing = math.atan2(-3.0, 5.0) - math.pi / 2
    np.testing.assert_allclose(part(backend, sensed, "target_bearing", 1), [math.sin(bearing), math.cos(bearing)], atol=tolerance(backend))
    # agent 2 at (0, −3) faces left: the world bearing atan2(3, 5), less 180°
    bearing = math.atan2(3.0, 5.0) - math.pi
    np.testing.assert_allclose(part(backend, sensed, "target_bearing", 2), [math.sin(bearing), math.cos(bearing)], atol=tolerance(backend))


def test_rotating_the_agent_rotates_the_bearing_back(backend: Backend) -> None:
    headings = np.linspace(-math.pi, math.pi, 9)
    agents = [[0.0, 2.0 * i - 8.0, float(h)] for i, h in enumerate(headings)]
    layouts = [layout_params([-7.0, 8.0, 0.0], [9.0, 0.0], agents)]
    sensed, _ = senses(backend, layouts)
    bearings = host(backend, sensed.observation)[0, 0, :, LAYOUT["target_bearing"]]
    world = np.arctan2(0.0 - np.asarray([a[1] for a in agents]), 9.0)
    np.testing.assert_allclose(bearings[:, 0], np.sin(world - headings), atol=tolerance(backend))
    np.testing.assert_allclose(bearings[:, 1], np.cos(world - headings), atol=tolerance(backend))


SCENE = layout_params([3.0, 0.0, 0.0], [-8.0, 8.0], [[6.0, 0.0, 0.0]], circles=[[6.0, 3.0, 1.0]])
"""An agent at (6, 0): the wall 4 to its right, a circle 2 above it, the box's face 2 to its left, nothing within 5 below."""


def test_rays_hit_a_wall_an_obstacle_and_the_box(backend: Backend) -> None:
    sensed, _ = senses(backend, [SCENE])
    distance, box = part(backend, sensed, "ray_distance"), part(backend, sensed, "ray_box")
    hit = host(backend, sensed.ray_hit)[0, 0, 0]
    # ray k points at 30k°: 0 right (the wall), 3 up (the circle), 6 left (the box), 9 down (nothing)
    np.testing.assert_allclose(distance[[0, 3, 6, 9]], [4.0 / 5.0, 2.0 / 5.0, 2.0 / 5.0, 1.0], atol=tolerance(backend))
    assert box[[0, 3, 6, 9]].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert hit[[0, 3, 6, 9]].tolist() == [HIT_OBSTACLE, HIT_OBSTACLE, HIT_BOX, HIT_NOTHING]
    ends = np.stack([host(backend, sensed.ray_x)[0, 0, 0], host(backend, sensed.ray_y)[0, 0, 0]], axis=-1)
    np.testing.assert_allclose(ends[[0, 3, 6, 9]], [[10.0, 0.0], [6.0, 2.0], [4.0, 0.0], [6.0, -5.0]], atol=1e-5)
    # turned by 90°, ray 0 points up, at the circle; ray 3 left, at the box
    turned, _ = senses(backend, [SCENE], agent_heading=math.pi / 2)
    np.testing.assert_allclose(part(backend, turned, "ray_distance")[[0, 3]], [0.4, 0.4], atol=tolerance(backend))
    assert part(backend, turned, "ray_box")[[0, 3]].tolist() == [0.0, 1.0]


def test_nothing_in_range_reads_one(backend: Backend) -> None:
    sensed, _ = senses(backend, [layout_params([-7.0, -7.0, 0.0], [7.0, 7.0], [[0.0, 0.0, 0.0]])])
    assert np.all(part(backend, sensed, "ray_distance") == 1.0) and np.all(part(backend, sensed, "ray_box") == 0.0)


def test_an_obstacle_hides_the_box_behind_it(backend: Backend) -> None:
    hidden = layout_params([1.0, 0.0, math.pi / 2], [-8.0, 8.0], [[-3.0, 0.0, 0.0]], circles=[[-1.0, 0.0, 0.5]])
    sensed, _ = senses(backend, [hidden])
    assert part(backend, sensed, "ray_box")[0] == 0.0
    np.testing.assert_allclose(part(backend, sensed, "ray_distance")[0], 1.5 / 5.0, atol=tolerance(backend))


def test_box_cue_in_range_and_zeros_beyond(backend: Backend) -> None:
    sensed, _ = senses(backend, [SCENE])
    np.testing.assert_allclose(part(backend, sensed, "box_flag"), [1.0])
    np.testing.assert_allclose(part(backend, sensed, "box_distance"), [3.0 / 5.0], atol=tolerance(backend))
    np.testing.assert_allclose(part(backend, sensed, "box_bearing"), [0.0, -1.0], atol=tolerance(backend))
    np.testing.assert_allclose(part(backend, sensed, "box_heading"), [0.0, 1.0], atol=tolerance(backend))
    far = layout_params([3.0, 0.0, 0.3], [-8.0, 8.0], [[8.5, -8.0, 1.0]])
    sensed, _ = senses(backend, [far])
    assert np.all(host(backend, sensed.observation)[0, 0, 0, LAYOUT.groups["box"]] == 0.0)


@pytest.mark.parametrize("theta", [0.0, 0.4, 1.3, -2.0])
def test_the_relative_heading_is_the_same_for_the_box_turned_half_a_turn(backend: Backend, theta: float) -> None:
    turned = [layout_params([3.0, 0.0, angle], [-8.0, 8.0], [[6.0, 0.0, 0.5]]) for angle in (theta, theta + math.pi)]
    sensed, _ = senses(backend, turned)
    heading = host(backend, sensed.observation)[0, :, 0, LAYOUT["box_heading"]]
    np.testing.assert_allclose(heading[0], heading[1], atol=tolerance(backend))
    relative = 2.0 * (theta - 0.5)
    np.testing.assert_allclose(heading[0], [math.sin(relative), math.cos(relative)], atol=tolerance(backend))


def test_teammate_sectors(backend: Backend) -> None:
    alone = layout_params([-7.0, -7.0, 0.0], [7.0, 7.0], [[0.0, 0.0, 0.0]])
    sensed, _ = senses(backend, [alone])
    assert np.all(part(backend, sensed, "teammates") == 0.0), "an agent never senses itself"
    # one teammate 0.5 ahead: x = 1 − 0.5/5 = 0.9 in sector 0, 1 − e^(−0.9) ≈ 0.59; one 2 units to the left: sector 2
    pair = layout_params([-7.0, -7.0, 0.0], [7.0, 7.0], [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, -6.0, 0.0]])
    sensed, _ = senses(backend, [pair])
    densities = part(backend, sensed, "teammates")
    expected = np.zeros(8)
    expected[0], expected[2] = 1.0 - math.exp(-0.9), 1.0 - math.exp(-0.6)
    np.testing.assert_allclose(densities, expected, atol=tolerance(backend))
    assert densities[0] == pytest.approx(0.6, abs=0.02)
    weights = host(backend, sensed.teammate_weight)[0, 0, 0]
    np.testing.assert_allclose(weights, [0.0, 0.9, 0.6, 0.0], atol=tolerance(backend))
    # a crowd of ten ahead saturates toward 1
    crowd = [[0.0, 0.0, 0.0]] + [[1.0 + 0.55 * (i // 2), 0.3 * (1 if i % 2 else -1), 0.0] for i in range(10)]
    sensed, _ = senses(backend, [layout_params([-7.0, -7.0, 0.0], [7.0, 7.0], crowd)])
    assert part(backend, sensed, "teammates")[0] > 0.99


def test_sectors_turn_with_the_agent(backend: Backend) -> None:
    pair = layout_params([-7.0, -7.0, 0.0], [7.0, 7.0], [[0.0, 0.0, math.pi / 2], [1.0, 0.0, 0.0]])
    sensed, _ = senses(backend, [pair])
    densities = part(backend, sensed, "teammates")
    assert densities.argmax() == 6 and np.count_nonzero(densities) == 1, "a teammate to the right of an agent facing up is at −90°"


def test_proprioception_is_normalised(backend: Backend) -> None:
    layouts = [layout_params([-7.0, -7.0, 0.0], [7.0, 7.0], [[0.0, 0.0, 0.0], [0.0, 3.0, math.pi / 2]])]
    sensed, _ = senses(backend, layouts, agent_vx=[[[0.75, 0.75]]], agent_vy=[[[0.0, 0.0]]], agent_turn=[[[math.pi / 2, -math.pi]]])
    np.testing.assert_allclose(part(backend, sensed, "speed", 0), [0.5], atol=tolerance(backend))
    # agent 1 faces up while moving right: sideways is not forward
    np.testing.assert_allclose(part(backend, sensed, "speed", 1), [0.0], atol=tolerance(backend))
    np.testing.assert_allclose(part(backend, sensed, "turn", 0), [0.5], atol=tolerance(backend))
    np.testing.assert_allclose(part(backend, sensed, "turn", 1), [-1.0], atol=tolerance(backend))


def test_every_value_within_its_range_on_random_states(backend: Backend) -> None:
    rng = np.random.default_rng(5)
    generated = scenarios.training_set(30, PARAMS, seed=3, mix=((scenarios.EASY, 2), (scenarios.FULL, 4)))
    arrays = scenarios.layout_arrays([s.params for s in generated], backend)
    start = initial_state(arrays, 3)
    shape = tuple(start.agent_x.shape)

    def draw(low: float, high: float, size: tuple[int, ...] = shape) -> object:
        return backend.asarray(rng.uniform(low, high, size))

    state = dataclasses.replace(
        start,
        agent_x=draw(-11.0, 11.0),
        agent_y=draw(-11.0, 11.0),
        agent_heading=draw(-10.0, 10.0),
        agent_vx=draw(-3.0, 3.0),
        agent_vy=draw(-3.0, 3.0),
        agent_turn=draw(-5.0, 5.0),
        box_x=draw(-9.0, 9.0, shape[:2]),
        box_y=draw(-9.0, 9.0, shape[:2]),
        box_angle=draw(-7.0, 7.0, shape[:2]),
    )
    observation = host(backend, sensors.observe(state, arrays, PARAMS))
    assert observation.shape == (*shape, 43) and np.all(np.isfinite(observation))
    for name, (low, high) in LAYOUT.ranges.items():
        values = observation[..., LAYOUT[name]]
        assert values.min() >= low and values.max() <= high, name
    assert set(np.unique(observation[..., LAYOUT["ray_box"]])) <= {0.0, 1.0}
    assert set(np.unique(observation[..., LAYOUT["box_flag"]])) <= {0.0, 1.0}


def test_an_observation_does_not_depend_on_the_batch_or_the_padding(backend: Backend) -> None:
    generated = scenarios.training_set(20, PARAMS, seed=4, mix=((scenarios.EASY, 2), (scenarios.FULL, 4)))
    layouts = [s.params for s in generated]
    together = host(
        backend,
        sensors.observe(initial_state(scenarios.layout_arrays(layouts, backend), 3), scenarios.layout_arrays(layouts, backend), PARAMS),
    )
    for i in (0, 3, 5):
        arrays = scenarios.layout_arrays([layouts[i]], backend, circles=8, rects=8)
        alone = host(backend, sensors.observe(initial_state(arrays, 1), arrays, PARAMS))
        if backend.name == "numpy":
            assert np.array_equal(alone[0, 0], together[2, i]), "bit for bit on numpy"
        else:
            np.testing.assert_allclose(alone[0, 0], together[2, i], atol=1e-6)
