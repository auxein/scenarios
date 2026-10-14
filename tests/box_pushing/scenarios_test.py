"""The scenario generator (design doc §4): seeds, difficulties, validity and rejection, the sets, JSON, fingerprints, and the
layout arrays the physics uses."""

import math
from typing import cast

import numpy as np
import pytest
from auxein.backend import Backend
from auxein.environments import ScenarioSet

from auxein_scenarios.box_pushing import scenarios
from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.scenarios import EASY, FULL, Difficulty, is_valid, layout_params

PARAMS = WorldParams()


@pytest.fixture(scope="module")
def training() -> ScenarioSet:
    return scenarios.training_set(20, PARAMS, seed=1)


@pytest.fixture(scope="module")
def held_out() -> ScenarioSet:
    return scenarios.held_out_set(20, PARAMS, seed=1)


def test_same_seed_same_layouts(training: ScenarioSet) -> None:
    again = scenarios.training_set(20, PARAMS, seed=1)
    assert again == training and again.fingerprint == training.fingerprint
    assert scenarios.training_set(20, PARAMS, seed=2).fingerprint != training.fingerprint


def test_the_default_training_mix(training: ScenarioSet) -> None:
    assert len(training) == 32
    difficulties = [s.params["difficulty"] for s in training]
    assert difficulties == ["easy"] * 8 + ["full"] * 24
    assert training.ids[0] == "train-0000"


def _surface_distance_to_box(x: float, y: float, box: list[float]) -> float:
    c, s = math.cos(box[2]), math.sin(box[2])
    lx, ly = c * (x - box[0]) + s * (y - box[1]), c * (y - box[1]) - s * (x - box[0])
    dx, dy = abs(lx) - PARAMS.box.width / 2, abs(ly) - PARAMS.box.height / 2
    return math.hypot(max(dx, 0.0), max(dy, 0.0)) + min(max(dx, dy), 0.0)


@pytest.mark.parametrize("difficulty", [EASY, FULL])
def test_difficulties_respect_their_ranges(training: ScenarioSet, held_out: ScenarioSet, difficulty: Difficulty) -> None:
    layouts = [s.params for s in [*training, *held_out] if s.params["difficulty"] == difficulty.name]
    assert layouts
    for layout in layouts:
        obstacles = len(cast("list[object]", layout["circles"])) + len(cast("list[object]", layout["rects"]))
        assert difficulty.obstacles[0] <= obstacles <= difficulty.obstacles[1]
        box, target = cast("list[float]", layout["box"]), cast("list[float]", layout["target"])
        distance = math.hypot(target[0] - box[0], target[1] - box[1])
        assert difficulty.box_target_distance[0] <= distance <= difficulty.box_target_distance[1]
        for circle in cast("list[list[float]]", layout["circles"]):
            assert difficulty.circle_radius[0] <= circle[2] <= difficulty.circle_radius[1]
        for rect in cast("list[list[float]]", layout["rects"]):
            assert difficulty.rect_side[0] <= 2 * rect[2] <= difficulty.rect_side[1]
            assert difficulty.rect_side[0] <= 2 * rect[3] <= difficulty.rect_side[1]
        agents = cast("list[list[float]]", layout["agents"])
        assert len(agents) == 20
        if difficulty.spawn_radius is not None:
            assert all(_surface_distance_to_box(a[0], a[1], box) <= difficulty.spawn_radius for a in agents)


def test_every_layout_is_valid(training: ScenarioSet, held_out: ScenarioSet) -> None:
    for scenario in [*training, *held_out]:
        assert is_valid(scenario.params, PARAMS), scenario.id


def test_validity_check_rejects_blocked_and_overlapping_layouts() -> None:
    team = [[-5.0, 0.0, 0.0], [-5.0, 1.0, 0.0]]
    open_layout = layout_params([-3.0, 0.0, 0.0], [5.0, 0.0], team)
    assert is_valid(open_layout, PARAMS)
    # a wall of rectangles across the arena between the box and the target
    wall = [[0.0, -6.0, 0.5, 4.0], [0.0, 2.0, 0.5, 4.0], [0.0, 8.0, 0.5, 2.0]]
    assert not is_valid(layout_params([-3.0, 0.0, 0.0], [5.0, 0.0], team, rects=wall), PARAMS)
    # a gap narrower than the box's circumscribed diameter blocks the box but not the agents
    gap = [[0.0, -5.6, 0.5, 4.5], [0.0, 5.6, 0.5, 4.5]]  # a 2.2-wide gap in the middle
    assert not is_valid(layout_params([-3.0, 0.0, 0.0], [5.0, 0.0], team, rects=gap), PARAMS)
    # agents overlapping each other, an obstacle, the box, or a wall
    assert not is_valid(layout_params([-3.0, 0.0, 0.0], [5.0, 0.0], [[-5.0, 0.0, 0.0], [-5.0, 0.3, 0.0]]), PARAMS)
    assert not is_valid(layout_params([-3.0, 0.0, 0.0], [5.0, 0.0], [[0.0, 6.0, 0.0]], circles=[[0.0, 6.0, 1.0]]), PARAMS)
    assert not is_valid(layout_params([-3.0, 0.0, 0.0], [5.0, 0.0], [[-3.0, 0.0, 0.0]]), PARAMS)
    assert not is_valid(layout_params([-3.0, 0.0, 0.0], [5.0, 0.0], [[9.9, 0.0, 0.0]]), PARAMS)
    # an agent walled in, away from the box
    pen = [[6.0, 6.0, 0.25, 1.5], [9.0, 6.0, 0.25, 1.5], [7.5, 4.75, 1.75, 0.25], [7.5, 7.25, 1.75, 0.25]]
    assert not is_valid(layout_params([-3.0, 0.0, 0.0], [5.0, -5.0], [[7.5, 6.0, 0.0]], rects=pen), PARAMS)


def test_the_generator_rejects_and_redraws_invalid_layouts() -> None:
    """With many large obstacles, some draws block the box: they are redrawn, and what comes out is valid."""
    crowded = Difficulty(
        "crowded", obstacles=(10, 12), box_target_distance=(8.0, 12.0), spawn_radius=None, circle_radius=(1.0, 2.0), rect_side=(2.0, 6.0)
    )
    generated = scenarios.training_set(10, PARAMS, seed=3, mix=((crowded, 12),))
    attempts = [cast("int", s.params["attempts"]) for s in generated]
    assert max(attempts) > 1, "no draw was rejected: the test needs a denser difficulty"
    for scenario in generated:
        assert is_valid(scenario.params, PARAMS)


def test_impossible_difficulties_fail_clearly() -> None:
    impossible = Difficulty("impossible", obstacles=(0, 0), box_target_distance=(30.0, 40.0), spawn_radius=None)
    with pytest.raises(scenarios.LayoutError, match="no valid 'impossible' layout"):
        scenarios.generate_layout(np.random.default_rng(0), impossible, 5, PARAMS, max_attempts=20)


def test_training_and_held_out_sets_are_disjoint(training: ScenarioSet, held_out: ScenarioSet) -> None:
    assert len(held_out) == 64 and {s.params["difficulty"] for s in held_out} == {"full"}
    assert held_out.ids[0] == "held-0000"
    scenarios.check_disjoint(training, held_out)  # even with the same seed: their streams differ
    with pytest.raises(ValueError, match="same layout"):
        scenarios.check_disjoint(training, training)


def test_params_round_trip_through_json(training: ScenarioSet, tmp_path: object) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "training.json"
    training.save(path)
    loaded = ScenarioSet.load(path)
    assert loaded == training and loaded.fingerprint == training.fingerprint
    assert loaded[3].params == training[3].params


def test_the_fingerprint_changes_when_a_layout_does(training: ScenarioSet) -> None:
    params = [dict(s.params) for s in training]
    same = ScenarioSet.from_params(params, seed=1, ids=list(training.ids))
    assert same.fingerprint == training.fingerprint
    moved = dict(params[5])
    moved["target"] = [cast("list[float]", moved["target"])[0] + 0.01, cast("list[float]", moved["target"])[1]]
    changed = ScenarioSet.from_params([*params[:5], moved, *params[6:]], seed=1, ids=list(training.ids))
    assert changed.fingerprint != training.fingerprint


def test_layout_arrays(backend: Backend, training: ScenarioSet) -> None:
    layouts = [s.params for s in training]
    arrays = scenarios.layout_arrays(layouts, backend)
    s = len(layouts)
    most_circles = max(len(cast("list[object]", p["circles"])) for p in layouts)
    assert tuple(arrays.circle_x.shape) == (1, s, most_circles)
    assert tuple(arrays.agent_x.shape) == (1, s, 20)
    assert tuple(arrays.target_x.shape) == (1, s)
    assert arrays.scenarios == s and arrays.agents == 20
    valid = np.asarray(backend.to_numpy(arrays.circle_valid))[0]
    assert valid.sum() == sum(len(cast("list[object]", p["circles"])) for p in layouts)
    padded = scenarios.layout_arrays(layouts, backend, circles=10, rects=10)
    assert tuple(padded.rect_x.shape) == (1, s, 10)
    with pytest.raises(ValueError, match="less than a layout has"):
        scenarios.layout_arrays(layouts, backend, circles=0)
    with pytest.raises(ValueError, match="same team size"):
        scenarios.layout_arrays([layouts[0], layout_params([0, 0, 0], [5, 0], [[1, 1, 0]])], backend)
    x = np.asarray(backend.to_numpy(arrays.agent_x))[0, 2]
    np.testing.assert_allclose(x, [a[0] for a in cast("list[list[float]]", layouts[2]["agents"])], rtol=1e-6)
