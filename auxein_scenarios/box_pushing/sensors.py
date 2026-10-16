"""What each agent senses, local and in its own frame (design doc §3): the observation vector fed to the shared network.

**The observation layout is defined here, once** (`observation_layout`): the named parts of the vector, their slices and sizes,
derived from the sensing parameters (`config.SensingParams`), and the range of values each part takes. Everything else (the
environment, the senses-only controller, the overlay, the smoke test) reads it from here. With the defaults it has 43 values:

| Part | Values | Range |
|---|---|---|
| `target_distance` | the distance to the target, over the arena's diagonal | `[0, 1]` |
| `target_bearing` | the bearing to the target, `(sin, cos)` | `[-1, 1]` |
| `ray_distance` | per ray: the distance to the nearest wall, obstacle or box, over the ray range (1: nothing in range) | `[0, 1]` |
| `ray_box` | per ray: 1 if what it hit first is the box, else 0 | `{0, 1}` |
| `box_flag` | 1 if the box's centre is within the sensing range `r`, else 0 | `{0, 1}` |
| `box_distance` | the distance to the box's centre, over `r` (0 when out of range) | `[0, 1)` |
| `box_bearing` | the bearing to the box's centre, `(sin, cos)` (0 when out of range) | `[-1, 1]` |
| `box_heading` | the box's heading relative to the agent's, `(sin 2θ, cos 2θ)` (0 when out of range) | `[-1, 1]` |
| `teammates` | per sector: `1 − e^(−x)`, `x = Σ (1 − d / r)` over the teammates within `r` in it | `[0, 1)` |
| `speed` | the forward speed, over the maximum speed (clipped) | `[-1, 1]` |
| `turn` | the turning rate, over the maximum turning rate (clipped) | `[-1, 1]` |

Bearings are angles from the agent's heading, counter-clockwise positive. Ray `k` points at `2πk / rays` from the heading (ray 0
straight ahead); sector `k` is centred on `2πk / sectors` (sector 0 straight ahead). Rays start at the agent's centre, ignore
teammates (a crowd must not blind an agent to the box), and report 0 from inside a shape (an agent pressed into an obstacle).

**Batched like the physics.** `sense` computes every agent of every candidate and scenario at once, `(n, s, N, size)`, in the
namespace of the state's arrays, with no Python loops over candidates, scenarios or agents: the loops are over the few
obstacle slots (ray casts, `geometry.ray_*`) and the few teammate sectors (so that the `(n, s, N, N)` pair arrays are not
multiplied by the number of sectors in memory). **Pairwise distances are computed once per step** and serve both the teammate
weights and their sectors. Padded obstacle slots never hit, and the minimum over shapes is exact, so an agent's observation does
not depend on the padding or on the rest of the batch.

`sense` also returns the intermediate quantities the sensing overlay draws (ray end points and what they hit, the teammates
counted and their sectors); `observe` returns the observation alone.
"""

import math
from dataclasses import dataclass
from functools import cache

import numpy as np
from auxein.backend import Array

from auxein_scenarios.box_pushing.config import SensingParams, WorldParams
from auxein_scenarios.box_pushing.world import Layout, State
from auxein_scenarios.common.geometry import ray_aabb, ray_circle, ray_inside_aabb, ray_obb, to_frame

__all__ = ["HIT_BOX", "HIT_NOTHING", "HIT_OBSTACLE", "ObservationLayout", "Senses", "observation_layout", "observe", "sense"]

HIT_NOTHING, HIT_OBSTACLE, HIT_BOX = 0, 1, 2
"""What a ray hit first, in `Senses.ray_hit`: nothing within range, a wall or an obstacle, the box."""


@dataclass(frozen=True)
class ObservationLayout:
    """The named parts of the observation vector, as slices of its last axis (see the module docstring)."""

    sensing: SensingParams
    parts: dict[str, slice]
    """Every part, in order."""
    ranges: dict[str, tuple[float, float]]
    """The closed interval of values each part takes (`box_distance` and `teammates` never reach their upper bound)."""

    @property
    def size(self) -> int:
        return max(part.stop for part in self.parts.values())

    @property
    def groups(self) -> dict[str, slice]:
        """The five groups of design doc §3: `target`, `rays`, `box`, `teammates`, `proprioception`."""
        p = self.parts
        return {
            "target": slice(p["target_distance"].start, p["target_bearing"].stop),
            "rays": slice(p["ray_distance"].start, p["ray_box"].stop),
            "box": slice(p["box_flag"].start, p["box_heading"].stop),
            "teammates": p["teammates"],
            "proprioception": slice(p["speed"].start, p["turn"].stop),
        }

    def __getitem__(self, name: str) -> slice:
        return self.parts[name]


@cache
def observation_layout(sensing: SensingParams | None = None) -> ObservationLayout:
    """The observation layout for these sensing parameters (the defaults: 43 values)."""
    sensing = sensing or SensingParams()
    sizes = {
        "target_distance": (1, (0.0, 1.0)),
        "target_bearing": (2, (-1.0, 1.0)),
        "ray_distance": (sensing.rays, (0.0, 1.0)),
        "ray_box": (sensing.rays, (0.0, 1.0)),
        "box_flag": (1, (0.0, 1.0)),
        "box_distance": (1, (0.0, 1.0)),
        "box_bearing": (2, (-1.0, 1.0)),
        "box_heading": (2, (-1.0, 1.0)),
        "teammates": (sensing.sectors, (0.0, 1.0)),
        "speed": (1, (-1.0, 1.0)),
        "turn": (1, (-1.0, 1.0)),
    }
    parts: dict[str, slice] = {}
    start = 0
    for name, (size, _) in sizes.items():
        parts[name] = slice(start, start + size)
        start += size
    return ObservationLayout(sensing, parts, {name: interval for name, (_, interval) in sizes.items()})


@dataclass(frozen=True)
class Senses:
    """What the agents sensed in one state: the observation, and the intermediate quantities the overlay draws.

    Shapes: `n` candidates, `s` scenarios, `N` agents, `R` rays, `S` sectors."""

    observation: Array
    """`(n, s, N, size)`, laid out as `observation_layout(sensing)`."""
    ray_x: Array
    """`(n, s, N, R)`: where each ray stopped (what it hit, or its full reach)."""
    ray_y: Array
    ray_hit: Array
    """`(n, s, N, R)`, integer: `HIT_NOTHING`, `HIT_OBSTACLE` (a wall or an obstacle) or `HIT_BOX`."""
    teammate_weight: Array
    """`(n, s, N, N)`: `1 − d / r` for each teammate `j` agent `i` senses (within `r`), else 0; 0 on the diagonal."""
    teammate_sector: Array
    """`(n, s, N, N)`: the sector of teammate `j` around agent `i` (as a float index; meaningful where the weight is above 0)."""
    sector_sum: Array
    """`(n, s, N, S)`: the closeness-weighted count per sector, `x`, before squashing."""


def _ray_directions(sensing: SensingParams) -> tuple[np.ndarray, np.ndarray]:
    angles = 2.0 * math.pi * np.arange(sensing.rays) / sensing.rays
    return np.cos(angles), np.sin(angles)


def sense(state: State, layout: Layout, params: WorldParams, sensing: SensingParams | None = None) -> Senses:
    """What every agent senses in `state` (design doc §3), with the intermediate quantities (see `Senses`)."""
    sensing = sensing or SensingParams()
    backend = state.backend
    xp = backend.xp
    floats = state.agent_x.dtype
    x, y, heading = state.agent_x, state.agent_y, state.agent_heading
    c, s = xp.cos(heading), xp.sin(heading)
    r, reach = sensing.range, sensing.reach

    # the target, in the agent's frame
    lx, ly = to_frame(layout.target_x[..., None], layout.target_y[..., None], x, y, c, s)
    target_distance = xp.sqrt(lx * lx + ly * ly)
    there = target_distance > 0.0
    safe = xp.where(there, target_distance, 1.0)
    target = [
        xp.clip(target_distance / (params.arena.size * math.sqrt(2.0)), 0.0, 1.0),
        xp.where(there, ly / safe, 0.0),
        xp.where(there, lx / safe, 1.0),
    ]

    # range rays: walls, obstacles slot by slot, then the box; the nearest wins
    ck, sk = (backend.asarray(v) for v in _ray_directions(sensing))
    dx, dy = c[..., None] * ck - s[..., None] * sk, s[..., None] * ck + c[..., None] * sk  # (n, s, N, R)
    ox, oy = x[..., None], y[..., None]
    nearest = ray_inside_aabb(ox, oy, dx, dy, params.arena.half, reach)
    for slot in range(int(layout.circle_x.shape[-1])):
        cx, cy, cr, valid = (a[..., slot, None, None] for a in (layout.circle_x, layout.circle_y, layout.circle_r, layout.circle_valid))
        nearest = xp.minimum(nearest, xp.where(valid > 0.0, ray_circle(ox, oy, dx, dy, cx, cy, cr, reach), reach))
    for slot in range(int(layout.rect_x.shape[-1])):
        rx, ry, hx, hy, valid = (
            a[..., slot, None, None] for a in (layout.rect_x, layout.rect_y, layout.rect_hx, layout.rect_hy, layout.rect_valid)
        )
        nearest = xp.minimum(nearest, xp.where(valid > 0.0, ray_aabb(ox, oy, dx, dy, rx, ry, hx, hy, reach), reach))
    bx, by = state.box_x[..., None, None], state.box_y[..., None, None]
    bc, bs = xp.cos(state.box_angle)[..., None, None], xp.sin(state.box_angle)[..., None, None]
    to_box = ray_obb(ox, oy, dx, dy, bx, by, bc, bs, params.box.width / 2.0, params.box.height / 2.0, reach)
    hits_box = to_box < nearest
    distance = xp.minimum(nearest, to_box)
    # HIT_OBSTACLE for anything within reach, one more for the box (which is within reach whenever it is hit first)
    ray_hit = xp.astype(distance < reach, xp.int32) + xp.astype(hits_box, xp.int32)
    rays = [distance / reach, xp.astype(hits_box, floats)]

    # the box cue: within r of the box's centre
    blx, bly = to_frame(state.box_x[..., None], state.box_y[..., None], x, y, c, s)
    box_distance = xp.sqrt(blx * blx + bly * bly)
    on = box_distance < r
    seen = on & (box_distance > 0.0)
    safe = xp.where(seen, box_distance, 1.0)
    relative = 2.0 * (state.box_angle[..., None] - heading)
    box = [
        xp.astype(on, floats),
        xp.where(on, box_distance / r, 0.0),
        xp.where(seen, bly / safe, 0.0),
        xp.where(seen, blx / safe, 0.0),
        xp.where(on, xp.sin(relative), 0.0),
        xp.where(on, xp.cos(relative), 0.0),
    ]

    # teammates: the pairwise distances once, then closeness weights summed per sector
    agents = int(x.shape[-1])
    px, py = x[..., None, :] - x[..., :, None], y[..., None, :] - y[..., :, None]  # teammate j relative to agent i
    apart = xp.sqrt(px * px + py * py)
    index = xp.arange(agents, device=backend.device)
    other = index[:, None] != index[None, :]
    weight = xp.where(other & (apart < r), 1.0 - apart / r, 0.0)
    ci, si = c[..., :, None], s[..., :, None]
    bearing = xp.atan2(ci * py - si * px, ci * px + si * py)
    width = 2.0 * math.pi / sensing.sectors
    sector = xp.floor(bearing / width + 0.5) % sensing.sectors
    sums = xp.stack([xp.sum(xp.where(sector == k, weight, 0.0), axis=-1) for k in range(sensing.sectors)], axis=-1)
    teammates = 1.0 - xp.exp(-sums)

    # proprioception
    a = params.agent
    forward = state.agent_vx * c + state.agent_vy * s
    speed = 0.0 * forward if math.isinf(a.max_speed) else xp.clip(forward / a.max_speed, -1.0, 1.0)
    proprioception = [speed, xp.clip(state.agent_turn / a.max_turn_rate, -1.0, 1.0)]

    columns = [*(v[..., None] for v in target), *rays, *(v[..., None] for v in box), teammates, *(v[..., None] for v in proprioception)]
    observation = xp.concat(columns, axis=-1)
    return Senses(
        observation=observation,
        ray_x=ox + distance * dx,
        ray_y=oy + distance * dy,
        ray_hit=ray_hit,
        teammate_weight=weight,
        teammate_sector=sector,
        sector_sum=sums,
    )


def observe(state: State, layout: Layout, params: WorldParams, sensing: SensingParams | None = None) -> Array:
    """The observations of every agent in `state`: `(n, s, N, size)` (see the module docstring for the layout)."""
    return sense(state, layout, params, sensing).observation
