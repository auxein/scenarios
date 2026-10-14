"""The scenario generator (design doc §4): seeded layouts as Auxein `ScenarioSet`s, with difficulty as a parameter.

A scenario is drawn from its seed: obstacles, the box's pose, the target, and the agents' starting poses. **The whole layout is
stored in the scenario's `params`** (JSON), so the world has no randomness of its own, the set's fingerprint covers the actual
layouts, and a set can be saved and inspected. The params of a scenario:

```
{"difficulty": "easy", "attempts": 1,
 "box": [x, y, angle], "target": [x, y],
 "circles": [[x, y, radius], ...], "rects": [[x, y, half_width, half_height], ...],
 "agents": [[x, y, heading], ...]}
```

Generation runs once per set, on the host, with numpy and plain loops (design doc §2.2): it is not per step. A layout is
**valid** if the box can reach the target and every agent can reach the box, checked by flood fills on a grid of
`GRID` units: for the box with the obstacles and walls inflated by the box's circumscribed radius (so the check holds whatever
the box's orientation, and is conservative), for the agents inflated by an agent's radius. An invalid draw is **rejected and the
whole layout redrawn** from the same stream; `attempts` records how many draws it took.

`layout_arrays` turns a set's layouts into the padded `(1, s, ...)` arrays of `world.Layout`, on a given backend.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
from auxein.backend import Backend
from auxein.environments import ScenarioSet
from auxein.random import RunSeed

from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.world import Layout

__all__ = [
    "EASY",
    "FULL",
    "GRID",
    "Difficulty",
    "LayoutError",
    "check_disjoint",
    "generate_layout",
    "held_out_set",
    "is_valid",
    "layout_arrays",
    "layout_params",
    "training_set",
]

Floats = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]

GRID = 0.25
"""The cell size of the validity check's grid (units)."""

CLEARANCE = 0.3
"""The gap kept between a new obstacle and the box's starting footprint, and between it and the target zone."""

AGENT_GAP = 0.05
"""The gap kept around each agent at spawn: from obstacles, the box, the walls and each other."""


class LayoutError(RuntimeError):
    """No valid layout was found within the generator's attempts: the difficulty asks for something the arena cannot hold."""


@dataclass(frozen=True)
class Difficulty:
    """A difficulty level of the generator (design doc §4.2).

    - `obstacles`: the inclusive range of the number of obstacles, each a circle or an axis-aligned rectangle with equal odds.
    - `box_target_distance`: the range of the distance from the box's centre to the target's.
    - `spawn_radius`: agents start within this distance of the box's surface; `None` scatters them across the arena.
    - `circle_radius`, `rect_side`: the ranges of the obstacles' sizes. No side is thinner than an agent's diameter (design doc
      §2.2), which keeps agents and the box from passing through one.
    """

    name: str
    obstacles: tuple[int, int]
    box_target_distance: tuple[float, float]
    spawn_radius: float | None
    circle_radius: tuple[float, float] = (0.5, 1.5)
    rect_side: tuple[float, float] = (0.5, 4.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "obstacles": list(self.obstacles),
            "box_target_distance": list(self.box_target_distance),
            "spawn_radius": self.spawn_radius,
            "circle_radius": list(self.circle_radius),
            "rect_side": list(self.rect_side),
        }


EASY = Difficulty("easy", obstacles=(0, 2), box_target_distance=(3.0, 5.0), spawn_radius=3.0)
FULL = Difficulty("full", obstacles=(2, 6), box_target_distance=(6.0, 12.0), spawn_radius=None)


def _at(value: float) -> Floats:
    return np.asarray(value, dtype=np.float64)


def _box_radius(params: WorldParams) -> float:
    return math.hypot(params.box.width, params.box.height) / 2.0


def _circle_surface_distance(x: Floats, y: Floats, circle: Sequence[float]) -> Floats:
    return np.hypot(x - circle[0], y - circle[1]) - circle[2]


def _rect_surface_distance(x: Floats, y: Floats, rect: Sequence[float]) -> Floats:
    """Signed distance from an axis-aligned rectangle's surface (negative inside)."""
    dx, dy = np.abs(x - rect[0]) - rect[2], np.abs(y - rect[1]) - rect[3]
    outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    return outside + np.minimum(np.maximum(dx, dy), 0.0)


def _box_surface_distance(x: Floats, y: Floats, box: Sequence[float], params: WorldParams) -> Floats:
    c, s = math.cos(box[2]), math.sin(box[2])
    lx, ly = c * (x - box[0]) + s * (y - box[1]), c * (y - box[1]) - s * (x - box[0])
    return _rect_surface_distance(lx, ly, (0.0, 0.0, params.box.width / 2.0, params.box.height / 2.0))


def _obstacle_distance(x: Floats, y: Floats, layout: Mapping[str, object]) -> Floats:
    """The distance from the nearest obstacle's surface (infinite without obstacles)."""
    distance: Floats = np.full(x.shape, np.inf)
    for circle in cast("list[list[float]]", layout["circles"]):
        distance = np.minimum(distance, _circle_surface_distance(x, y, circle))
    for rect in cast("list[list[float]]", layout["rects"]):
        distance = np.minimum(distance, _rect_surface_distance(x, y, rect))
    return distance


def _grid(params: WorldParams) -> tuple[Floats, Floats]:
    half = params.arena.half
    centres = np.arange(-half + GRID / 2.0, half, GRID, dtype=np.float64)
    gx: Floats = centres[:, None] + np.zeros((1, centres.shape[0]))
    return gx, gx.T.copy()


def _cell(x: float, y: float, params: WorldParams) -> tuple[int, int]:
    count = round(params.arena.size / GRID)
    i = min(max(int((x + params.arena.half) / GRID), 0), count - 1)
    j = min(max(int((y + params.arena.half) / GRID), 0), count - 1)
    return i, j


def _flood(free: Mask, seeds: Mask) -> Mask:
    """The cells of `free` connected (4-neighbours) to the free cells of `seeds`, by repeated dilation."""
    reached = seeds & free
    while True:
        grown = reached.copy()
        grown[1:, :] |= reached[:-1, :]
        grown[:-1, :] |= reached[1:, :]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= free
        if bool((grown == reached).all()):
            return reached
        reached = grown


def _free(layout: Mapping[str, object], params: WorldParams, inflation: float) -> Mask:
    gx, gy = _grid(params)
    inside = (np.abs(gx) <= params.arena.half - inflation) & (np.abs(gy) <= params.arena.half - inflation)
    return inside & (_obstacle_distance(gx, gy, layout) > inflation)


def _box_reaches_target(layout: Mapping[str, object], params: WorldParams) -> bool:
    free = _free(layout, params, _box_radius(params))
    box, target = cast("list[float]", layout["box"]), cast("list[float]", layout["target"])
    seeds = np.zeros_like(free)
    seeds[_cell(box[0], box[1], params)] = True
    return bool(_flood(free, seeds)[_cell(target[0], target[1], params)])


def _agent_reachable(layout: Mapping[str, object], params: WorldParams) -> Mask:
    """The cells from which an agent can reach the box: free for an agent (obstacles, walls and the box inflated by its radius)
    and connected to the cells next to the box."""
    gx, gy = _grid(params)
    r = params.agent.radius
    box = cast("list[float]", layout["box"])
    box_distance = _box_surface_distance(gx, gy, box, params)
    free = _free(layout, params, r) & (box_distance > r)
    return _flood(free, box_distance < r + 2 * GRID)


def is_valid(layout: Mapping[str, object], params: WorldParams) -> bool:
    """The validity check (design doc §2.2): the box can reach the target, and every agent starts in free space connected to the
    box, not overlapping an obstacle, the box, a wall or another agent."""
    if not _box_reaches_target(layout, params):
        return False
    agents = np.asarray(layout["agents"], dtype=np.float64).reshape(-1, 3)
    if agents.shape[0] == 0:
        return True
    r, half = params.agent.radius, params.arena.half
    x, y = agents[:, 0], agents[:, 1]
    if bool((np.abs(x) > half - r).any() or (np.abs(y) > half - r).any()):
        return False
    box = cast("list[float]", layout["box"])
    if bool((_obstacle_distance(x, y, layout) < r).any() or (_box_surface_distance(x, y, box, params) < r).any()):
        return False
    pairs = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]) + np.eye(len(x)) * 4 * r
    if bool((pairs < 2 * r).any()):
        return False
    reachable = _agent_reachable(layout, params)
    return all(bool(reachable[_cell(float(ax), float(ay), params)]) for ax, ay in zip(x, y, strict=True))


def _draw_obstacles(
    rng: np.random.Generator, difficulty: Difficulty, box: list[float], target: list[float], params: WorldParams
) -> tuple[list[list[float]], list[list[float]]] | None:
    half, keep_box = params.arena.half, _box_radius(params) + CLEARANCE
    keep_target = max(params.simulation.target_radius, _box_radius(params)) + CLEARANCE
    circles: list[list[float]] = []
    rects: list[list[float]] = []
    count = int(rng.integers(difficulty.obstacles[0], difficulty.obstacles[1] + 1))
    for _ in range(count):
        for _ in range(100):
            if rng.random() < 0.5:
                radius = float(rng.uniform(*difficulty.circle_radius))
                x, y = (float(v) for v in rng.uniform(-half + radius, half - radius, size=2))
                candidate = {"circles": [[x, y, radius]], "rects": []}
            else:
                hx, hy = (float(v) / 2.0 for v in rng.uniform(*difficulty.rect_side, size=2))
                x, y = float(rng.uniform(-half + hx, half - hx)), float(rng.uniform(-half + hy, half - hy))
                candidate = {"circles": [], "rects": [[x, y, hx, hy]]}
            near_box = _obstacle_distance(_at(box[0]), _at(box[1]), candidate) < keep_box
            near_target = _obstacle_distance(_at(target[0]), _at(target[1]), candidate) < keep_target
            if not (near_box or near_target):
                circles += candidate["circles"]
                rects += candidate["rects"]
                break
        else:
            return None
    return circles, rects


def _draw_agents(
    rng: np.random.Generator, difficulty: Difficulty, layout: Mapping[str, object], agents: int, params: WorldParams
) -> list[list[float]] | None:
    half, r = params.arena.half, params.agent.radius
    box = cast("list[float]", layout["box"])
    reachable = _agent_reachable(layout, params)
    placed: list[list[float]] = []
    for _ in range(agents):
        for _ in range(1000):
            if difficulty.spawn_radius is None:
                x, y = (float(v) for v in rng.uniform(-half + r + AGENT_GAP, half - r - AGENT_GAP, size=2))
            else:
                reach = _box_radius(params) + difficulty.spawn_radius
                angle, distance = float(rng.uniform(-math.pi, math.pi)), reach * math.sqrt(float(rng.random()))
                x, y = box[0] + distance * math.cos(angle), box[1] + distance * math.sin(angle)
                if max(abs(x), abs(y)) > half - r - AGENT_GAP:
                    continue
                if float(_box_surface_distance(_at(x), _at(y), box, params)) > difficulty.spawn_radius:
                    continue
            px, py = _at(x), _at(y)
            if float(_obstacle_distance(px, py, layout)) < r + AGENT_GAP:
                continue
            if float(_box_surface_distance(px, py, box, params)) < r + AGENT_GAP:
                continue
            if any(math.hypot(x - a[0], y - a[1]) < 2 * r + AGENT_GAP for a in placed):
                continue
            if not reachable[_cell(x, y, params)]:
                continue
            placed.append([x, y, float(rng.uniform(-math.pi, math.pi))])
            break
        else:
            return None
    return placed


def generate_layout(
    rng: np.random.Generator, difficulty: Difficulty, agents: int, params: WorldParams, *, max_attempts: int = 200
) -> dict[str, object]:
    """One layout (the scenario's params) drawn from `rng`: redrawn whole until it is valid, at most `max_attempts` times."""
    half, margin = params.arena.half, _box_radius(params) + 0.5
    if half - margin <= -half + margin:
        raise LayoutError("the arena is too small for the box")
    for attempt in range(1, max_attempts + 1):
        box = [float(v) for v in rng.uniform(-half + margin, half - margin, size=2)] + [float(rng.uniform(0.0, math.pi))]
        distance = float(rng.uniform(*difficulty.box_target_distance))
        angle = float(rng.uniform(-math.pi, math.pi))
        target = [box[0] + distance * math.cos(angle), box[1] + distance * math.sin(angle)]
        if max(abs(target[0]), abs(target[1])) > half - margin:
            continue
        obstacles = _draw_obstacles(rng, difficulty, box, target, params)
        if obstacles is None:
            continue
        layout: dict[str, object] = {
            "difficulty": difficulty.name,
            "box": box,
            "target": target,
            "circles": obstacles[0],
            "rects": obstacles[1],
            "agents": [],
        }
        if not _box_reaches_target(layout, params):
            continue
        placed = _draw_agents(rng, difficulty, layout, agents, params)
        if placed is None:
            continue
        layout["agents"] = placed
        layout["attempts"] = attempt
        return layout
    raise LayoutError(f"no valid {difficulty.name!r} layout for {agents} agents in {max_attempts} attempts")


def layout_params(
    box: Sequence[float],
    target: Sequence[float],
    agents: Sequence[Sequence[float]],
    circles: Sequence[Sequence[float]] = (),
    rects: Sequence[Sequence[float]] = (),
    difficulty: str = "custom",
) -> dict[str, object]:
    """A hand-built layout in the params format above: the box `[x, y, angle]`, the target `[x, y]`, agents `[x, y, heading]`,
    circles `[x, y, radius]` and rectangles `[x, y, half_width, half_height]`. Not checked: see `is_valid`."""
    return {
        "difficulty": difficulty,
        "box": [float(v) for v in box],
        "target": [float(v) for v in target],
        "circles": [[float(v) for v in c] for c in circles],
        "rects": [[float(v) for v in r] for r in rects],
        "agents": [[float(v) for v in a] for a in agents],
    }


def _set(name: str, mix: Sequence[tuple[Difficulty, int]], agents: int, params: WorldParams, seed: int, prefix: str) -> ScenarioSet:
    run = RunSeed(seed)
    layouts: list[Mapping[str, object]] = []
    for difficulty, count in mix:
        for _ in range(count):
            index = len(layouts)
            rng = np.random.default_rng(run.sequence(name, index))
            layouts.append(generate_layout(rng, difficulty, agents, params))
    return ScenarioSet.from_params(layouts, seed=seed, ids=[f"{prefix}-{i:04d}" for i in range(len(layouts))])


def training_set(
    agents: int,
    params: WorldParams | None = None,
    *,
    seed: int = 1,
    mix: Sequence[tuple[Difficulty, int]] = ((EASY, 8), (FULL, 24)),
) -> ScenarioSet:
    """The training set (design doc §4.2): by default 8 easy and 24 full scenarios, in that order, for a team of `agents`."""
    return _set("box-pushing-training", mix, agents, params or WorldParams(), seed, "train")


def held_out_set(
    agents: int, params: WorldParams | None = None, *, seed: int = 2, count: int = 64, difficulty: Difficulty = FULL
) -> ScenarioSet:
    """A held-out set (design doc §4.3): `count` scenarios at full difficulty, never used in training.

    Its layouts come from a stream of its own (another name than the training set's), so the two cannot share a layout even
    with the same seed; `check_disjoint` verifies it."""
    return _set("box-pushing-held-out", ((difficulty, count),), agents, params or WorldParams(), seed, "held")


def check_disjoint(*sets: ScenarioSet) -> None:
    """Raises `ValueError` if a layout appears more than once across the sets (or within one)."""
    seen: dict[str, str] = {}
    for scenarios in sets:
        for scenario in scenarios:
            key = repr(sorted((k, repr(v)) for k, v in scenario.params.items() if k not in ("attempts", "difficulty")))
            if key in seen:
                raise ValueError(f"scenarios {seen[key]!r} and {scenario.id!r} have the same layout")
            seen[key] = scenario.id


def layout_arrays(
    layouts: Sequence[Mapping[str, object]], backend: Backend, *, circles: int | None = None, rects: int | None = None
) -> Layout:
    """The `(1, s, ...)` arrays of `world.Layout` for a sequence of scenario params (or a `ScenarioSet`'s scenarios' params), on
    `backend`. Obstacles are padded to `circles` and `rects` slots (by default the most any layout has), with validity masks;
    every layout must have the same number of agents."""
    if not layouts:
        raise ValueError("no layouts")
    teams = {len(cast("list[object]", p["agents"])) for p in layouts}
    if len(teams) != 1:
        raise ValueError(f"the layouts of one batch must have the same team size, got {sorted(teams)}")
    most_circles = max(len(cast("list[object]", p["circles"])) for p in layouts)
    most_rects = max(len(cast("list[object]", p["rects"])) for p in layouts)
    circles = most_circles if circles is None else circles
    rects = most_rects if rects is None else rects
    if circles < most_circles or rects < most_rects:
        raise ValueError(f"padding to {circles} circles and {rects} rectangles is less than a layout has ({most_circles}, {most_rects})")
    s = len(layouts)
    circle = np.zeros((s, circles, 4))  # x, y, radius, valid
    rect = np.zeros((s, rects, 5))  # x, y, hx, hy, valid
    for i, p in enumerate(layouts):
        for j, values in enumerate(cast("list[list[float]]", p["circles"])):
            circle[i, j] = [*values, 1.0]
        for j, values in enumerate(cast("list[list[float]]", p["rects"])):
            rect[i, j] = [*values, 1.0]
    box = np.asarray([cast("list[float]", p["box"]) for p in layouts], dtype=np.float64)
    target = np.asarray([cast("list[float]", p["target"]) for p in layouts], dtype=np.float64)
    agents = np.asarray([cast("list[list[float]]", p["agents"]) for p in layouts], dtype=np.float64).reshape(s, -1, 3)

    def put(values: Floats) -> object:
        return backend.asarray(np.ascontiguousarray(values)[None])

    return Layout(
        circle_x=put(circle[..., 0]),
        circle_y=put(circle[..., 1]),
        circle_r=put(circle[..., 2]),
        circle_valid=put(circle[..., 3]),
        rect_x=put(rect[..., 0]),
        rect_y=put(rect[..., 1]),
        rect_hx=put(rect[..., 2]),
        rect_hy=put(rect[..., 3]),
        rect_valid=put(rect[..., 4]),
        target_x=put(target[:, 0]),
        target_y=put(target[:, 1]),
        box_x=put(box[:, 0]),
        box_y=put(box[:, 1]),
        box_angle=put(box[:, 2]),
        agent_x=put(agents[..., 0]),
        agent_y=put(agents[..., 1]),
        agent_heading=put(agents[..., 2]),
    )
