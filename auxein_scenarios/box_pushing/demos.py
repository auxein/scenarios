"""The demo clips: short scenes that show whether the world behaves right (M1a), and what agents sense (M1b).

Each demo is a `Demo`: its layouts (in the scenarios' params format, hand-built or generated), a scripted controller (true
state) or a batch policy (senses, through the environment's step loop), the number of steps and a title saying what to look
for. `run` simulates it on a backend, records the trajectory and renders the clip, so a demo shows exactly what the physics
computes.

| Demo | Look for |
|---|---|
| `threshold` | `k − 1` agents push and the box does not move; one more joins and it moves |
| `rotation` | a push near one end turns the box while it slides; a centred push does not |
| `obstacles` | pushed at an angle: the box slides over a circle, stops against a thin rectangle, wedges into a corner |
| `crowd` | 40 agents sent to one point jostle; nobody overlaps anybody for long; nothing explodes |
| `walls` | agents at full speed into the walls and the corners: no tunnelling |
| `delivery` | the scripted push team delivers the box in an easy and a full scenario; the scenario freezes on delivery |
| `sensing` | one agent turns on the spot near the box, an obstacle and moving teammates: its sensing overlay changes everywhere |

Two more clips watch **policies acting on their own senses** through the environment's step loop (M1b): `brains` (random
`MLPPolicy` brains, the GA's first generation, one panel per brain) and `senses` (the senses-only controller of `scripted.py`
on generated scenarios). Both can draw the sensing overlay.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from auxein.backend import Backend
from auxein.environments import BatchPolicy, ScenarioSet
from auxein.random import RandomStream

from auxein_scenarios.box_pushing import scenarios, scripted
from auxein_scenarios.box_pushing.config import SensingParams, WorldParams
from auxein_scenarios.box_pushing.environment import BoxPushingWorld, brain
from auxein_scenarios.box_pushing.render import render_clip
from auxein_scenarios.box_pushing.world import Controller, TrajectoryRecorder, initial_state, simulate


def layout(*args: Any, **kwargs: Any) -> dict[str, object]:
    return scenarios.layout_params(*args, difficulty="demo", **kwargs)


__all__ = ["DEMOS", "Demo", "Watched", "build", "run", "watch_brains", "watch_senses"]


@dataclass(frozen=True)
class Demo:
    """A scene to simulate and render: one panel per layout, all driven for `steps` steps by `controller` (from the true state,
    through `world.simulate`) or by `policy` (from the senses, through the environment's step loop, with the sensing overlay for
    the agents in `watch`)."""

    name: str
    title: str
    subtitle: str
    layouts: list[dict[str, object]]
    labels: list[str]
    controller: Controller | None
    steps: int
    columns: int | None = None
    stop_when_done: bool = True
    notes: list[str] = field(default_factory=list[str])
    policy: BatchPolicy | None = None
    watch: tuple[int, ...] = ()


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
        f"{k - 1} agents push from the start: the box stays dark, only outlined (at rest). "
        f"The {k}th arrives: it lights up white and moves.",
        [layout([-2.0, 0.0, math.pi / 2], [8.0, 0.0], team)],
        [f"{k} agents needed"],
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
    circle = layout([-5.0, 2.2, math.pi / 2], [7.0, 2.2], _behind(-5.0, 2.2, team, rng), circles=[[0.0, 0.0, 1.2]])
    rect = layout([-5.0, 0.5, math.pi / 2], [6.0, 4.5], _behind(-5.0, 0.5, team, rng), rects=[[0.0, 0.0, 0.25, 2.5]])
    corner_box = [5.0, 5.0, math.pi / 4]
    corner = layout(corner_box, [-8.0, -8.0], _behind(5.0, 5.0, team, rng))
    goals_x = np.asarray([7.0, 6.0, 12.0])[None, :]
    goals_y = np.asarray([2.2, 4.5, 12.0])[None, :]
    return Demo(
        "obstacles",
        "Contacts: the box against a circle, a rectangle and a corner",
        "The push team drives the box into each at an angle: it slides over the circle and on to the target, stops against "
        "the thin rectangle's end, and wedges into the corner. It never passes through.",
        [circle, rect, corner],
        ["against a circle", "against a thin rectangle", "into a corner"],
        scripted.push(backend.asarray(goals_x), backend.asarray(goals_y)),
        400,
        columns=3,
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
    scene = layout([8.0, -8.5, 0.0], [-7.5, -7.5], team)
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
        "The box's centre enters the target zone: the zone fills, the box dims to silver and the scenario freezes, "
        "while the other goes on.",
        chosen,
        labels,
        scripted.push(),
        params.simulation.max_steps,
        notes=notes,
    )


def _sensing(params: WorldParams, agents: int, backend: Backend) -> Demo:
    """One agent turning slowly on the spot beside the box, between a circle and a rectangle, with teammates driving circles
    around it: as it turns, its rays sweep the box and the obstacles, the box cue's bearing turns, and the teammates move
    from sector to sector."""
    watched = [-1.8, -1.6, 0.0]
    teammates = [[-3.5, -3.8, 0.0], [0.5, -4.2, math.pi], [-4.4, 0.8, -math.pi / 2], [0.2, 2.4, math.pi / 2]]
    scene = layout(
        [0.0, 0.0, 0.4],
        [7.0, 6.5],
        [watched, *teammates],
        circles=[[-5.0, -1.8, 1.0]],
        rects=[[0.6, -5.0, 1.5, 0.5]],
    )
    actions = [[0.0, 0.3], [0.45, 0.6], [0.45, -0.6], [0.45, 0.6], [0.4, -0.5]]
    return Demo(
        "sensing",
        "Sensing: what one agent senses as it turns",
        "The bright agent turns on the spot: its rays sweep the box (bright hits) and the obstacles and walls (dim hits), the "
        "line to the box shows the box cue, and the teammates it senses are joined to it, brighter when closer.",
        [scene],
        ["agent 0, watched"],
        None,
        240,
        policy=scripted.ConstantPolicy(actions, 1, backend),
        watch=(0,),
        stop_when_done=False,
    )


DEMOS: Mapping[str, Callable[[WorldParams, int, Backend], Demo]] = {
    "threshold": _threshold,
    "rotation": _rotation,
    "obstacles": _obstacles,
    "crowd": _crowd,
    "walls": _walls,
    "delivery": _delivery,
    "sensing": _sensing,
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
    senses = None
    if demo.policy is not None:
        world = BoxPushingWorld(world_params, steps=demo.steps, backend=on)
        demo_set = ScenarioSet.from_params(demo.layouts, ids=[f"{demo.name}-{i}" for i in range(len(demo.layouts))])
        trajectory, senses, _ = world.watch(demo.policy, tuple(demo_set), agents=demo.watch, stop_when_done=demo.stop_when_done)
    elif demo.controller is not None:
        arrays = scenarios.layout_arrays(demo.layouts, on)
        recorder = TrajectoryRecorder()
        simulate(
            initial_state(arrays, 1), arrays, world_params, demo.controller, demo.steps, hook=recorder, stop_when_done=demo.stop_when_done
        )
        trajectory = recorder.trajectory()
    else:
        raise ValueError(f"demo {demo.name!r} has neither a controller nor a policy")
    return render_clip(
        trajectory,
        demo.layouts,
        world_params,
        Path(out) / f"{demo.name}.mp4",
        labels=demo.labels,
        senses=senses,
        kicker=f"Demo · {demo.name}",
        title=demo.title,
        subtitle=demo.subtitle,
        fps=fps,
        speedup=speedup,
        columns=demo.columns,
    )


# --- policies acting on their own senses (M1b) ---


@dataclass(frozen=True)
class Watched:
    """A rendered clip of a policy and what it measured: `measurements[name]` is the `(n, s)` host array of each measurement."""

    path: Path
    labels: list[str]
    measurements: dict[str, np.ndarray]


def _measured(world: BoxPushingWorld, rollout_measurements: Mapping[str, object], backend: Backend) -> dict[str, np.ndarray]:
    return {name: np.asarray(backend.to_numpy(value), dtype=np.float64) for name, value in rollout_measurements.items()}


def watch_brains(
    out: str | Path,
    *,
    count: int = 6,
    seed: int = 1,
    difficulty: scenarios.Difficulty = scenarios.EASY,
    agents: int = 20,
    overlay: bool = False,
    params: WorldParams | None = None,
    sensing: SensingParams | None = None,
    backend: Backend | None = None,
    fps: float = 30.0,
    speedup: float = 2.0,
) -> Watched:
    """**The clueless start**: `count` random brains sampled from the brain's genome space (`MLPPolicy.space()`, as the GA's
    first generation is), each driving the whole team on the same generated scenario, one panel per brain. With `overlay`, the
    sensing overlay of agent 0 in every panel."""
    world_params, on = params or WorldParams(), backend or Backend("numpy", "cpu", "float64")
    world = BoxPushingWorld(world_params, sensing)
    policy = brain(world.sensing)
    genomes = policy.space().sample_genomes(count, RandomStream(np.random.SeedSequence(seed), on), on)
    scenario_set = scenarios.training_set(agents, world_params, seed=seed, mix=((difficulty, 1),))
    trajectory, senses, rollout = world.watch(policy.decode_batch(genomes), tuple(scenario_set), agents=(0,) if overlay else ())
    labels = [f"brain {i}" for i in range(count)]
    path = render_clip(
        trajectory,
        [dict(scenario_set[0].params)],
        world_params,
        Path(out) / f"brains-{difficulty.name}-seed{seed}{'-overlay' if overlay else ''}.mp4",
        cells=[(i, 0) for i in range(count)],
        labels=labels,
        senses=senses,
        sensing=world.sensing,
        kicker="Brains · the first generation",
        title=f"{count} random brains on one {difficulty.name} scenario",
        subtitle="Each panel is one random network driving all "
        f"{agents} agents, as the genetic algorithm's first generation will: do they wander, spin, crowd, find the box?",
        fps=fps,
        speedup=speedup,
        columns=min(count, 3),
    )
    return Watched(path, labels, _measured(world, rollout.measurements, on))


def watch_senses(
    out: str | Path,
    *,
    count: int = 4,
    seed: int = 1,
    difficulty: scenarios.Difficulty = scenarios.EASY,
    agents: int = 20,
    overlay: bool = False,
    params: WorldParams | None = None,
    sensing: SensingParams | None = None,
    backend: Backend | None = None,
    fps: float = 30.0,
    speedup: float = 2.0,
) -> Watched:
    """The senses-only controller (`scripted.SensesController`, design doc §10.1) on `count` generated scenarios, one panel per
    scenario. With `overlay`, the sensing overlay of agent 0 in every panel."""
    world_params, on = params or WorldParams(), backend or Backend("numpy", "cpu", "float64")
    world = BoxPushingWorld(world_params, sensing)
    controller = scripted.SensesController(1, on, world.sensing, world_params)
    scenario_set = scenarios.training_set(agents, world_params, seed=seed, mix=((difficulty, count),))
    trajectory, senses, rollout = world.watch(controller, tuple(scenario_set), agents=(0,) if overlay else ())
    labels = list(scenario_set.ids)
    path = render_clip(
        trajectory,
        [dict(s.params) for s in scenario_set],
        world_params,
        Path(out) / f"senses-{difficulty.name}-seed{seed}{'-overlay' if overlay else ''}.mp4",
        labels=labels,
        senses=senses,
        sensing=world.sensing,
        kicker="Senses only · a feasibility check",
        title=f"A hand-written controller that sees only the {world.observation_size} inputs",
        subtitle="No true state: each agent pushes when it is behind the box, steers to a ray that sees it, "
        "and otherwise orbits the target to search. If this delivers, evolution can.",
        fps=fps,
        speedup=speedup,
        columns=min(count, 2) if count <= 4 else 3,
    )
    return Watched(path, labels, _measured(world, rollout.measurements, on))
