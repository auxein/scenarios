"""The demo clips of M1a: short scenes, driven by scripted controllers, that show whether the world behaves right.

Each demo is a `Demo`: its layouts (in the scenarios' params format, hand-built or generated), the controller, the number of
steps and a title saying what to look for. `run` simulates it on a backend, records the trajectory and renders the clip, so a
demo shows exactly what the physics computes.

| Demo | Look for |
|---|---|
| `threshold` | `k − 1` agents push and the box does not move; one more joins and it moves |
| `rotation` | a push near one end turns the box while it slides; a centred push does not |
| `obstacles` | the box pushed against a circle, a rectangle and into a corner slides along them and does not pass through |
| `crowd` | 40 agents sent to one point jostle; nobody overlaps anybody for long; nothing explodes |
| `walls` | agents at full speed into the walls and the corners: no tunnelling |
| `delivery` | the scripted push team delivers the box in an easy and a full scenario; the scenario freezes on delivery |
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from auxein.backend import Backend

from auxein_scenarios.box_pushing import scenarios, scripted
from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.render import render_clip
from auxein_scenarios.box_pushing.world import Controller, TrajectoryRecorder, initial_state, simulate


def layout(*args: Any, **kwargs: Any) -> dict[str, object]:
    return scenarios.layout_params(*args, difficulty="demo", **kwargs)


__all__ = ["DEMOS", "Demo", "build", "run"]


@dataclass(frozen=True)
class Demo:
    """A scene to simulate and render: one panel per layout, all driven by `controller` for `steps` steps."""

    name: str
    title: str
    subtitle: str
    layouts: list[dict[str, object]]
    labels: list[str]
    controller: Controller
    steps: int
    columns: int | None = None
    stop_when_done: bool = True
    notes: list[str] = field(default_factory=list[str])


def _behind(box_x: float, box_y: float, count: int, rng: np.random.Generator, depth: float = 2.5, spread: float = 2.0) -> list[list[float]]:
    """`count` agents scattered behind (to the left of) a box, facing +x, without overlapping."""
    placed: list[list[float]] = []
    while len(placed) < count:
        x, y = box_x - 1.2 - float(rng.uniform(0.0, depth)), box_y + float(rng.uniform(-spread, spread))
        if all(math.hypot(x - a[0], y - a[1]) > 0.6 for a in placed):
            placed.append([x, y, 0.0])
    return placed


def _threshold(params: WorldParams, agents: int, backend: Backend) -> Demo:
    k, r = params.friction.k, params.agent.radius
    face = -2.0 - params.box.height / 2.0 - r  # agents' centres touching the box's back face (the box is turned to face +x)
    slots = np.linspace(-(params.box.width / 2 - r), params.box.width / 2 - r, k)
    team = [[face, float(y), 0.0] for y in slots[:-1]] + [[face - 4.0, float(slots[-1]), 0.0]]
    return Demo(
        "threshold",
        f"Static friction: {k - 1} agents cannot move the box, {k} can",
        f"{k - 1} agents push from the start: the box stays grey (at rest). The {k}th arrives: the box turns orange and moves.",
        [layout([-2.0, 0.0, math.pi / 2], [8.0, 0.0], team)],
        [f"k = {k}"],
        scripted.constant(1.0),
        300,
    )


def _rotation(params: WorldParams, agents: int, backend: Backend) -> Demo:
    r, half_width = params.agent.radius, params.box.width / 2
    face = -4.0 - params.box.height / 2.0 - r
    end = [half_width - r, half_width - 3 * r]  # two lines near the box's top end
    off = [
        [face, end[0], 0.0],
        [face, end[1], 0.0],
        [face - 2 * r - 0.05, end[0], 0.0],
        [face - 2 * r - 0.05, end[1], 0.0],
        [face - 4 * r - 0.1, end[0], 0.0],
    ]
    mid = [r + 0.02, -(r + 0.02)]
    centred = [
        [face, mid[0], 0.0],
        [face, mid[1], 0.0],
        [face - 2 * r - 0.05, mid[0], 0.0],
        [face - 2 * r - 0.05, mid[1], 0.0],
        [face - 4 * r - 0.1, 0.0, 0.0],
    ]
    box = [-4.0, 0.0, math.pi / 2]
    return Demo(
        "rotation",
        f"Torque: {params.friction.k + 1} agents pushing off-centre turn the box",
        "Left: the push is near one end, so the box turns while it slides. Right: the same push, centred: it slides straight.",
        [layout(box, [8.0, 0.0], off), layout(box, [8.0, 0.0], centred)],
        ["pushed near one end", "pushed at the centre"],
        scripted.constant(1.0),
        260,
    )


def _obstacles(params: WorldParams, agents: int, backend: Backend) -> Demo:
    rng = np.random.default_rng(3)
    team = 12
    circle = layout([-5.0, 0.7, math.pi / 2], [7.0, 0.0], _behind(-5.0, 0.7, team, rng), circles=[[0.0, 0.0, 1.2]])
    rect = layout([-5.0, 1.2, math.pi / 2], [7.0, 0.0], _behind(-5.0, 1.2, team, rng), rects=[[0.0, 0.0, 0.25, 2.5]])
    corner_box = [5.0, 5.0, math.pi / 4]
    corner_team = [[a[0] - 0.0, a[1], a[2]] for a in _behind(5.0, 5.0, team, rng)]
    corner = layout(corner_box, [-8.0, -8.0], corner_team)
    goals_x = np.asarray([7.0, 7.0, 12.0])[None, :]
    goals_y = np.asarray([0.0, 0.0, 12.0])[None, :]
    return Demo(
        "obstacles",
        "Contacts: the box against a circle, a rectangle and a corner",
        "The push team drives the box into each: it slides along them or stops, and never passes through.",
        [circle, rect, corner],
        ["against a circle", "against a thin rectangle", "into a corner"],
        scripted.push(backend.asarray(goals_x), backend.asarray(goals_y)),
        400,
        stop_when_done=False,
    )


def _crowd(params: WorldParams, agents: int, backend: Backend) -> Demo:
    rng = np.random.default_rng(4)
    count = agents if agents > 0 else 40
    angles = rng.uniform(-math.pi, math.pi, count)
    radii = rng.uniform(5.0, 8.5, count)
    team = [
        [float(rho * math.cos(a)), float(rho * math.sin(a)), float(rng.uniform(-math.pi, math.pi))]
        for rho, a in zip(radii, angles, strict=True)
    ]
    scene = layout([8.0, -8.5, 0.0], [-8.0, 8.0], team)
    return Demo(
        "crowd",
        f"Crowd: {count} agents sent to the same point",
        "They jostle at the centre and settle into a packed disc: nobody overlaps anybody for long, nothing explodes.",
        [scene],
        [f"{count} agents to (0, 0)"],
        scripted.go_to(0.0, 0.0),
        400,
    )


def _walls(params: WorldParams, agents: int, backend: Backend) -> Demo:
    count = agents if agents > 0 else 16
    angles = np.linspace(-math.pi, math.pi, count, endpoint=False) + math.pi / count
    team = [[float(1.5 * math.cos(a)), float(1.5 * math.sin(a)), float(a)] for a in angles]
    scene = layout([-6.0, -8.8, 0.0], [6.0, 8.0], team)
    return Demo(
        "walls",
        f"Walls: {count} agents at full speed into the walls and corners",
        f"They reach top speed ({params.agent.max_speed:g} units/s) and stay pressed against the walls: no tunnelling.",
        [scene],
        ["full thrust, straight out"],
        scripted.constant(1.0),
        300,
    )


def _delivery(params: WorldParams, agents: int, backend: Backend) -> Demo:
    count = agents if agents > 0 else 20
    chosen: list[dict[str, object]] = []
    labels: list[str] = []
    notes: list[str] = []
    for difficulty, seed in ((scenarios.EASY, 11), (scenarios.FULL, 12)):
        candidates = scenarios.training_set(count, params, seed=seed, mix=((difficulty, 16),))
        layouts = [dict(s.params) for s in candidates]
        arrays = scenarios.layout_arrays(layouts, backend)
        end = simulate(initial_state(arrays, 1), arrays, params, scripted.push(), params.simulation.max_steps)
        done: list[bool] = backend.to_numpy(end.done)[0].tolist()
        steps: list[int] = backend.to_numpy(end.steps)[0].tolist()
        if any(done):
            best = min((i for i, d in enumerate(done) if d), key=lambda i: steps[i])
            label = f"{difficulty.name} {candidates[best].id}"
        else:
            best = 0
            label = f"{difficulty.name} {candidates[best].id} (not delivered)"
            notes.append(f"no {difficulty.name} scenario of seed {seed} was delivered by the push team")
        chosen.append(layouts[best])
        labels.append(label)
    return Demo(
        "delivery",
        f"Delivery: the scripted push team ({count} agents) delivers the box",
        "The box's centre enters the target zone: the box turns green and the scenario freezes, while the other goes on.",
        chosen,
        labels,
        scripted.push(),
        params.simulation.max_steps,
        notes=notes,
    )


DEMOS: Mapping[str, Callable[[WorldParams, int, Backend], Demo]] = {
    "threshold": _threshold,
    "rotation": _rotation,
    "obstacles": _obstacles,
    "crowd": _crowd,
    "walls": _walls,
    "delivery": _delivery,
}
"""The demos by name, in the order `demo all` renders them."""


def build(name: str, params: WorldParams | None = None, *, agents: int = 0, backend: Backend | None = None) -> Demo:
    """The demo `name`; `agents` sets the team size of the demos that have a variable one (`crowd`, `walls`, `delivery`), 0 for
    their defaults."""
    if name not in DEMOS:
        raise ValueError(f"unknown demo {name!r}: use one of {sorted(DEMOS)}")
    return DEMOS[name](params or WorldParams(), agents, backend or Backend("numpy", "cpu", "float64"))


def run(
    demo: Demo,
    out: str | Path,
    params: WorldParams | None = None,
    *,
    backend: Backend | None = None,
    fps: float = 30.0,
    speedup: float = 2.0,
) -> Path:
    """Simulates `demo` on `backend`, records every step and renders `<out>/<name>.mp4`."""
    world_params = params or WorldParams()
    on = backend or Backend("numpy", "cpu", "float64")
    arrays = scenarios.layout_arrays(demo.layouts, on)
    recorder = TrajectoryRecorder()
    simulate(initial_state(arrays, 1), arrays, world_params, demo.controller, demo.steps, hook=recorder, stop_when_done=demo.stop_when_done)
    return render_clip(
        recorder.trajectory(),
        demo.layouts,
        world_params,
        Path(out) / f"{demo.name}.mp4",
        labels=demo.labels,
        title=demo.title,
        subtitle=demo.subtitle,
        fps=fps,
        speedup=speedup,
        columns=demo.columns,
    )
