"""The physics core, on the corner backends: the friction threshold, rotation, tunnelling, conservation,
stability, delivery, determinism and batching."""

import dataclasses
import math
from typing import Literal

import numpy as np
import pytest
from auxein.backend import Backend

from auxein_scenarios.box_pushing import scenarios, scripted
from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.scenarios import layout_params
from auxein_scenarios.box_pushing.world import State, initial_state, simulate, step
from tests.box_pushing.support import FACING_X, host, line_behind, run

FAR = [8.0, 8.0]  # a target nobody reaches


def box_moved(backend: Backend, trajectory_x: np.ndarray, end: State) -> float:
    return float(np.max(np.abs(trajectory_x[-1] - trajectory_x[0])))


# --- static friction: k - 1 agents cannot move the box, k can --------------------------------------------------------------


@pytest.mark.parametrize("k", [4, 3, 2])
def test_friction_threshold(corner_backend: Backend, k: int) -> None:
    params = WorldParams().replace(friction={"k": k})
    parked = [[-9.0, 9.0, math.pi]]  # the batch needs equal teams: the weak team's extra agent drives into a far wall
    weak = layout_params([0.0, 0.0, FACING_X], FAR, line_behind(0.0, k - 1, params) + parked)
    strong = layout_params([0.0, 0.0, FACING_X], FAR, line_behind(0.0, k, params))
    end, trajectory, _ = run(corner_backend, [weak, strong], scripted.constant(1.0), 200, params)
    moved = np.abs(trajectory.box_x[-1, 0] - trajectory.box_x[0, 0])
    # k - 1 agents: at most the small jolt of their first contact (the contact springs overshoot), then at rest
    assert moved[0] < 0.03, f"{k - 1} agents moved the box {moved[0]:.3f}"
    assert bool(trajectory.box_at_rest[-1, 0, 0])
    assert bool(np.all(trajectory.box_at_rest[100:, 0, 0]))
    # k agents push it steadily
    assert moved[1] > 1.0, f"{k} agents moved the box only {moved[1]:.3f}"
    assert not bool(trajectory.box_at_rest[-1, 0, 1])


def test_k_minus_one_agents_cannot_keep_a_moving_box_moving(corner_backend: Backend) -> None:
    params = WorldParams()
    team = line_behind(0.0, params.friction.k - 1, params)
    layout = scenarios.layout_arrays([layout_params([0.0, 0.0, FACING_X], FAR, team)], corner_backend)
    start = initial_state(layout, 1)
    xp = corner_backend.xp
    moving = dataclasses.replace(
        start,
        box_vx=xp.ones_like(start.box_vx) * 0.5,
        box_at_rest=start.box_at_rest & False,
        agent_vx=xp.ones_like(start.agent_vx) * 0.5,
    )
    end = simulate(moving, layout, params, scripted.constant(1.0), 300, stop_when_done=False)
    assert bool(host(corner_backend, end.box_at_rest)[0, 0])
    assert float(host(corner_backend, end.box_x)[0, 0]) < 1.0  # it slid a little and stopped


# --- rotation: off-centre pushes turn the box ----------------------------------------------------------------------------


def test_off_centre_push_rotates_the_box_the_right_way(corner_backend: Backend) -> None:
    params = WorldParams()
    r = params.agent.radius
    count = params.friction.k + 1
    # k + 1 agents on the upper half of the face (two lines, pushing through each other): the push is above the centre
    ys = [0.75, 0.25, 0.75, 0.25, 0.75][:count]
    face = -params.box.height / 2.0 - r
    upper = [[face - (i // 2) * (2 * r + 0.02), y, 0.0] for i, y in enumerate(ys)]
    centred = [
        [a[0], -a[1] if i % 2 else a[1], 0.0]
        for i, a in enumerate(
            [
                [face, 0.26, 0.0],
                [face, 0.26, 0.0],
                [face - 2 * r - 0.02, 0.26, 0.0],
                [face - 2 * r - 0.02, 0.26, 0.0],
                [face - 4 * r - 0.04, 0.0, 0.0],
            ]
        )
    ]
    _, trajectory, _ = run(
        corner_backend,
        [layout_params([0.0, 0.0, FACING_X], FAR, upper), layout_params([0.0, 0.0, FACING_X], FAR, centred)],
        scripted.constant(1.0),
        100,
        params,
    )
    turned = trajectory.box_angle[-1, 0] - trajectory.box_angle[0, 0]
    # a push along +x above the centre has torque r × f = -y F < 0: the box turns clockwise
    assert turned[0] < -0.05
    assert abs(turned[1]) < (1e-9 if corner_backend.precision == "float64" else 1e-4)
    assert trajectory.box_x[-1, 0, 1] - trajectory.box_x[0, 0, 1] > 0.3  # the centred push slides it straight


# --- no tunnelling --------------------------------------------------------------------------------------------------------


def test_no_tunnelling_through_walls_at_full_speed(corner_backend: Backend) -> None:
    params = WorldParams()
    angles = np.linspace(-math.pi, math.pi, 16, endpoint=False) + math.pi / 16
    team = [[1.5 * math.cos(a), 1.5 * math.sin(a), a] for a in angles]
    layout = scenarios.layout_arrays([layout_params([0.0, -8.5, 0.0], FAR, team)], corner_backend)
    start = initial_state(layout, 1)
    speed = params.agent.max_speed
    start = dataclasses.replace(
        start, agent_vx=speed * corner_backend.xp.cos(start.agent_heading), agent_vy=speed * corner_backend.xp.sin(start.agent_heading)
    )
    end_state = simulate(start, layout, params, scripted.constant(1.0), 300, stop_when_done=False)
    half, r = params.arena.half, params.agent.radius
    x, y = host(corner_backend, end_state.agent_x), host(corner_backend, end_state.agent_y)
    assert np.all(np.abs(x) < half) and np.all(np.abs(y) < half)
    assert np.all(np.maximum(np.abs(x), np.abs(y)) > half - 2 * r)  # they are all pressed against a wall
    assert np.all(np.maximum(np.abs(x), np.abs(y)) < half - r + 0.05)  # with a small overlap


def test_no_tunnelling_through_the_thinnest_obstacle(corner_backend: Backend) -> None:
    """A column of agents and a team pushing the box, all at full thrust into a rectangle 0.5 thick (the thinnest allowed)."""
    params = WorldParams()
    wall = [[0.0, 0.0, 0.25, 4.0]]  # x in [-0.25, 0.25]
    column = [[-0.6 - i * 0.55, 1.0, 0.0] for i in range(10)] + [[-0.6, -2.0 - 0.6 * j, 0.0] for j in range(3)]
    agents_only = layout_params([-6.0, 7.0, 0.0], FAR, column, rects=wall)
    box_team = layout_params(
        [-1.0, 0.0, FACING_X],
        FAR,
        line_behind(-1.0, 4, params)
        + [[a[0] - 0.55 * (i + 1), a[1], 0.0] for i in range(2) for a in line_behind(-1.0, 4, params)]
        + [[-6.0, -6.0, 0.0]],
        rects=wall,
    )
    end, trajectory, _ = run(corner_backend, [agents_only, box_team][:1], scripted.constant(1.0), 300, params)
    assert np.all(trajectory.agent_x[:, 0, 0] < 0.0), "an agent's centre crossed the middle of the obstacle"
    end, trajectory, _ = run(corner_backend, [box_team], scripted.constant(1.0), 300, params)
    assert np.all(trajectory.box_x[:, 0, 0] < -0.25), "the box's centre reached the obstacle"
    assert np.all(trajectory.agent_x[:, 0, 0, :12] < 0.0)


# --- conservation ---------------------------------------------------------------------------------------------------------


def test_collisions_conserve_momentum_and_never_add_energy(corner_backend: Backend) -> None:
    """With thrust, drag and lateral damping off, agents collide freely: momentum is conserved, kinetic energy never grows."""
    params = WorldParams().replace(agent={"max_speed": math.inf, "lateral_damping": 0.0})
    rng = np.random.default_rng(5)
    team = [
        [float(x), float(y), float(h)]
        for x, y, h in zip(rng.uniform(-2, 2, 12), rng.uniform(-2, 2, 12), rng.uniform(-math.pi, math.pi, 12), strict=True)
    ]
    # spread them so that none starts overlapping
    team = [[a[0] * 2.0, a[1] * 2.0, a[2]] for a in team]
    layout = scenarios.layout_arrays([layout_params([0.0, -8.5, 0.0], FAR, team)], corner_backend)
    start = initial_state(layout, 1)
    xp = corner_backend.xp
    start = dataclasses.replace(start, agent_vx=1.0 * xp.cos(start.agent_heading), agent_vy=1.0 * xp.sin(start.agent_heading))
    p0 = host(corner_backend, start.agent_vx).sum(), host(corner_backend, start.agent_vy).sum()
    energy = [0.5 * float((host(corner_backend, start.agent_vx) ** 2 + host(corner_backend, start.agent_vy) ** 2).sum())]
    contacts_seen = False
    state = start
    for _ in range(60):  # a few seconds: collisions among themselves, before anyone reaches a wall
        state = step(state, xp.zeros((*state.agent_x.shape, 2), dtype=state.agent_x.dtype), layout, params)
        vx, vy = host(corner_backend, state.agent_vx), host(corner_backend, state.agent_vy)
        energy.append(0.5 * float((vx**2 + vy**2).sum()))
        contacts_seen = contacts_seen or energy[-1] < energy[0] - 1e-6
        x, y = host(corner_backend, state.agent_x), host(corner_backend, state.agent_y)
        assert np.all(np.maximum(np.abs(x), np.abs(y)) < params.arena.half - params.agent.radius), "an agent reached a wall"
    tolerance = 1e-9 if corner_backend.precision == "float64" else 1e-4
    vx, vy = host(corner_backend, state.agent_vx), host(corner_backend, state.agent_vy)
    assert abs(vx.sum() - p0[0]) < tolerance and abs(vy.sum() - p0[1]) < tolerance
    assert contacts_seen, "no collision happened: the test is not testing anything"
    # outside contacts the energy is constant; through a contact the damper only removes it
    assert max(energy) <= energy[0] * (1 + tolerance)
    assert energy[-1] <= energy[0]


# --- stability ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["numpy", "torch"])
def test_long_crowded_run_with_random_actions_stays_finite_in_float32(name: Literal["numpy", "torch"]) -> None:
    if name == "torch":
        pytest.importorskip("torch")
    backend = Backend(name, "cpu", "float32")
    params = WorldParams()
    rng = np.random.default_rng(6)
    team: list[list[float]] = []
    while len(team) < 100:  # a crowd in a ring around the box
        a, rho = rng.uniform(-math.pi, math.pi), rng.uniform(2.0, 7.0)
        x, y = rho * math.cos(a), rho * math.sin(a)
        if all(math.hypot(x - b[0], y - b[1]) > 0.55 for b in team):
            team.append([x, y, float(rng.uniform(-math.pi, math.pi))])
    layout = scenarios.layout_arrays([layout_params([0.0, 0.0, 0.3], FAR, team)], backend)
    noise = rng.uniform(-1.0, 1.0, (2000, 1, 1, 100, 2))
    towards_box = scripted.go_to(0.0, 0.0)
    t = [0]

    def crowd(state: State, layout_: object, params_: WorldParams) -> object:
        actions = towards_box(state, layout, params_)
        mixed = backend.xp.clip(actions + backend.asarray(noise[t[0]]), -1.0, 1.0)
        t[0] += 1
        return mixed

    end = simulate(initial_state(layout, 1), layout, params, crowd, 2000, stop_when_done=False)  # type: ignore[arg-type]
    for field in dataclasses.fields(end):
        values = host(backend, getattr(end, field.name))
        assert np.all(np.isfinite(values)), f"{field.name} is not finite"
    x, y = host(backend, end.agent_x), host(backend, end.agent_y)
    assert np.all(np.abs(x) < params.arena.half) and np.all(np.abs(y) < params.arena.half)


# --- delivery -------------------------------------------------------------------------------------------------------------


def test_delivery_freezes_the_scenario_while_others_continue(corner_backend: Backend) -> None:
    params = WorldParams()
    close = layout_params([0.0, 0.0, FACING_X], [2.0, 0.0], line_behind(0.0, 8, params, spread=0.75))
    far = layout_params([0.0, 0.0, FACING_X], [9.0, 0.0], line_behind(0.0, 8, params, spread=0.75))
    end, trajectory, _ = run(corner_backend, [close, far], scripted.push(), 300, params)
    done = trajectory.done[:, 0]
    assert bool(done[-1, 0]) and not bool(done[-1, 1])
    first = int(np.argmax(done[:, 0]))
    assert first > 0
    # from the delivery on, every recorded field of the delivered scenario is unchanged; the other keeps moving
    for name in ("agent_x", "agent_y", "agent_heading", "box_x", "box_y", "box_angle"):
        values = getattr(trajectory, name)[first:, 0, 0]
        assert np.array_equal(values, np.broadcast_to(values[0], values.shape)), f"{name} changed after delivery"
    assert trajectory.box_x[-1, 0, 1] != trajectory.box_x[first, 0, 1]
    steps = host(corner_backend, end.steps)[0]
    assert steps[0] == first and steps[1] == 300
    distance = math.hypot(trajectory.box_x[first, 0, 0] - 2.0, trajectory.box_y[first, 0, 0])
    assert distance <= params.simulation.target_radius


def test_simulate_stops_early_when_everything_is_done(corner_backend: Backend) -> None:
    params = WorldParams()
    close = layout_params([0.0, 0.0, FACING_X], [1.0, 0.0], line_behind(0.0, 8, params, spread=0.75))
    end, trajectory, _ = run(corner_backend, [close], scripted.push(), 300, params, stop_when_done=True)
    assert int(trajectory.step[-1]) < 300 and bool(trajectory.done[-1, 0, 0])


# --- determinism and batching ---------------------------------------------------------------------------------------------


def _generated(agents: int = 12) -> list[dict[str, object]]:
    mixed = scenarios.training_set(agents, seed=21, mix=((scenarios.EASY, 2), (scenarios.FULL, 2)))
    return [dict(s.params) for s in mixed]


def test_same_inputs_give_identical_trajectories(corner_backend: Backend) -> None:
    layouts = _generated()
    _, first, _ = run(corner_backend, layouts, scripted.push(), 150, n=2)
    _, second, _ = run(corner_backend, layouts, scripted.push(), 150, n=2)
    for field in dataclasses.fields(first):
        assert np.array_equal(getattr(first, field.name), getattr(second, field.name)), field.name


def test_a_scenario_gets_the_same_trajectory_alone_or_in_a_batch(corner_backend: Backend) -> None:
    layouts = _generated()
    steps = 120
    _, batch, _ = run(corner_backend, layouts, scripted.push(), steps, n=3)
    for scenario in range(len(layouts)):
        _, alone, _ = run(
            corner_backend, [layouts[scenario]], scripted.push(), steps, n=1, circles=batch_slots(layouts)[0], rects=batch_slots(layouts)[1]
        )
        for candidate in range(3):
            for name in ("agent_x", "agent_y", "box_x", "box_y", "box_angle"):
                got, want = getattr(batch, name)[:, candidate, scenario], getattr(alone, name)[:, 0, 0]
                if corner_backend.name == "numpy":
                    assert np.array_equal(got, want), f"{name}, scenario {scenario}, candidate {candidate}"
                else:
                    np.testing.assert_allclose(got, want, atol=1e-3, err_msg=f"{name}, scenario {scenario}, candidate {candidate}")


def batch_slots(layouts: list[dict[str, object]]) -> tuple[int, int]:
    return max(len(p["circles"]) for p in layouts), max(len(p["rects"]) for p in layouts)  # type: ignore[arg-type]


def test_padding_obstacles_changes_nothing(corner_backend: Backend) -> None:
    layouts = _generated()
    _, six, _ = run(corner_backend, layouts, scripted.push(), 120, circles=6, rects=6)
    _, ten, _ = run(corner_backend, layouts, scripted.push(), 120, circles=10, rects=10)
    for field in dataclasses.fields(six):
        if corner_backend.name == "numpy":
            assert np.array_equal(getattr(six, field.name), getattr(ten, field.name)), field.name
        else:
            np.testing.assert_allclose(getattr(six, field.name), getattr(ten, field.name), atol=1e-4, err_msg=field.name)
