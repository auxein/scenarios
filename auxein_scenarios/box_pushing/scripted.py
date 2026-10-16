"""Scripted controllers: test and demo tools that drive the physics from the **true state**, with no sensing and no network.

Each is a batched `world.Controller`: the state, the layout and the parameters to actions `(n, s, N, 2)` in `[-1, 1]` (thrust,
turning), computed for the whole batch at once like the physics, with no Python loops over candidates, scenarios or agents.
They exist to exercise the world (the physics tests, the demo clips, the timing), not to solve the task: `push` is naive by
design and ignores obstacles.

- `idle`: zero actions.
- `constant(thrust, turn)`: the same actions every step (per agent if given as arrays).
- `go_to(x, y)`: each agent to its own point, then hold there.
- `push(goal_x, goal_y)`: agents line up in assigned columns on the box's far side from the goal, then drive toward the goal,
  steering to stay on their line.

And one that does **not** read the true state (M1b):

- `SensesController`: a policy-like object (`act_batch(observations)`, like a brain) that sees **only the observation vector**
  (design doc §3), a hand-written feasibility check for design doc §10.1.
"""

import math

from auxein.backend import Array, Backend, backend_of

from auxein_scenarios.box_pushing.config import SensingParams, WorldParams
from auxein_scenarios.box_pushing.sensors import observation_layout
from auxein_scenarios.box_pushing.world import Controller, Layout, State
from auxein_scenarios.common.geometry import wrap_angle

__all__ = ["SensesController", "constant", "go_to", "idle", "push", "steer"]

TURN_GAIN = 3.0
"""Turning action per radian of heading error (saturating at ±1, i.e. full turning rate, beyond a third of a radian)."""

ARRIVE_GAIN = 1.5
"""Thrust per unit of distance when arriving at a point, so that an agent slows down as it gets close."""

BRAKE_GAIN = 1.0
"""Thrust against the forward speed, per unit of speed, which damps the arrival."""


def idle(state: State, layout: Layout, params: WorldParams) -> Array:
    """Zero actions: agents coast to a stop under their drag."""
    xp = backend_of(state.agent_x).xp
    return xp.zeros((*state.agent_x.shape, 2), dtype=state.agent_x.dtype, device=backend_of(state.agent_x).device)


def constant(thrust: Array | float, turn: Array | float = 0.0) -> Controller:
    """The same actions every step: `thrust` and `turn` broadcast against the team's `(n, s, N)` (numbers or arrays)."""

    def control(state: State, layout: Layout, params: WorldParams) -> Array:
        xp = backend_of(state.agent_x).xp
        zero = 0.0 * state.agent_x
        return xp.stack([zero + thrust, zero + turn], axis=-1)

    return control


def steer(state: State, aim_x: Array, aim_y: Array, *, arrive: bool) -> Array:
    """Actions that take each agent toward its aim point `(aim_x, aim_y)` (broadcast against `(n, s, N)`).

    The agent turns toward the point and thrusts forward only as far as it is facing it (`cos` of the heading error, so it turns
    on the spot when the point is behind it). With `arrive`, thrust falls with the distance and brakes against the forward speed,
    so it stops at the point; without, it drives at full thrust (pushing)."""
    xp = backend_of(state.agent_x).xp
    dx, dy = aim_x - state.agent_x, aim_y - state.agent_y
    error = wrap_angle(xp.atan2(dy, dx) - state.agent_heading)
    facing = xp.clip(xp.cos(error), 0.0, None)
    if arrive:
        forward = state.agent_vx * xp.cos(state.agent_heading) + state.agent_vy * xp.sin(state.agent_heading)
        drive = ARRIVE_GAIN * xp.sqrt(dx * dx + dy * dy) - BRAKE_GAIN * forward
    else:
        drive = 1.0 + 0.0 * dx
    thrust = xp.clip(drive, -1.0, 1.0) * facing
    turn = xp.clip(TURN_GAIN * error, -1.0, 1.0)
    return xp.stack([thrust, turn], axis=-1)


def go_to(x: Array | float, y: Array | float) -> Controller:
    """Each agent to its own point `(x, y)` (broadcast against `(n, s, N)`), then hold there."""

    def control(state: State, layout: Layout, params: WorldParams) -> Array:
        return steer(state, 0.0 * state.agent_x + x, 0.0 * state.agent_y + y, arrive=True)

    return control


def push(goal_x: Array | None = None, goal_y: Array | None = None, *, spacing: float = 0.0, lead: float = 3.0) -> Controller:
    """The naive push team: lines behind the box, then a straight push toward the goal (the layout's target by default).

    The goal direction `g` runs from the box's centre to the goal. Agent `i` is assigned a **column**, a line parallel to `g`
    across the box's back face (`i mod columns`, with as many columns as fit across the box's longer side, agents `spacing` apart:
    with the defaults, 4 across the 2-unit side, the `k` the default friction needs). An agent behind the
    box and on its line pushes: it drives at full thrust toward a point `lead` units ahead on its line, which keeps it on the
    line as the box moves and turns. Agents of one column queue on the line and push through the ones in front. Any other agent
    goes to the head of its line, around the box if the box is in its way. Lines follow the box, so a box pushed off-centre that
    turns is pushed back on course."""

    def control(state: State, layout: Layout, params: WorldParams) -> Array:
        xp = backend_of(state.agent_x).xp
        r, agents = params.agent.radius, state.agent_x.shape[-1]
        hx, hy = params.box.width / 2.0, params.box.height / 2.0
        gx_goal = layout.target_x if goal_x is None else goal_x
        gy_goal = layout.target_y if goal_y is None else goal_y
        bx, by = state.box_x[..., None], state.box_y[..., None]
        gx, gy = gx_goal - state.box_x, gy_goal - state.box_y
        norm = xp.sqrt(gx * gx + gy * gy)
        safe = xp.where(norm > 0.0, norm, 1.0)
        gx, gy = (xp.where(norm > 0.0, gx / safe, 1.0))[..., None], (xp.where(norm > 0.0, gy / safe, 0.0))[..., None]
        px, py = -gy, gx  # across the push
        c, s = xp.cos(state.box_angle)[..., None], xp.sin(state.box_angle)[..., None]
        depth = hx * xp.abs(gx * c + gy * s) + hy * xp.abs(gy * c - gx * s)  # the box's extent along g
        width = hx * xp.abs(px * c + py * s) + hy * xp.abs(py * c - px * s)  # ... and across it

        columns = max(1, math.floor(2.0 * max(hx, hy) / (2.0 * r + spacing)))
        index = xp.arange(agents, device=backend_of(state.agent_x).device)
        column = xp.astype(index % columns, state.agent_x.dtype)
        across = (column - (columns - 1) / 2.0) / max((columns - 1) / 2.0, 1.0) * xp.clip(width - r, 0.0, None)
        head_x = bx - gx * (depth + r + spacing) + px * across
        head_y = by - gy * (depth + r + spacing) + py * across

        # where the agent is relative to the box: along g (positive: ahead) and across it
        ax, ay = state.agent_x - bx, state.agent_y - by
        along, side = ax * gx + ay * gy, ax * px + ay * py
        pushing = (along < -depth) & (xp.abs(side - across) < 0.35)
        sign = xp.where(side >= 0.0, 1.0, -1.0)
        clear = width + r + 0.6
        ahead = along > -(depth + r)
        in_way = ahead & (xp.abs(side) < width + r + 0.2)
        beside_x, beside_y = bx + px * sign * clear + gx * along, by + py * sign * clear + gy * along
        corner_x, corner_y = bx + px * sign * clear - gx * (depth + r + 0.6), by + py * sign * clear - gy * (depth + r + 0.6)
        way_x = xp.where(in_way, beside_x, xp.where(ahead, corner_x, head_x))
        way_y = xp.where(in_way, beside_y, xp.where(ahead, corner_y, head_y))

        go = steer(state, way_x, way_y, arrive=False)
        drive = steer(state, bx + px * across + gx * lead, by + py * across + gy * lead, arrive=False)
        return xp.where(pushing[..., None], drive, go)

    return control


class SensesController:
    """A hand-written controller that sees **only what a brain sees**: the observation vector, read through
    `sensors.observation_layout`, and nothing of the true state. A **feasibility check for design doc §10.1**, not a solution:
    if a memoryless controller can deliver some easy scenarios from these senses, the senses carry what the task needs and
    evolution can find it too; if it can do nothing, the senses are missing something.

    Like a brain, it is a batch policy (`act_batch(observations) -> actions`, `(n, s, N, size)` to `(n, s, N, 2)`), memoryless
    and the same for every agent, so agents act differently only because they sense differently. Every agent, in its own frame:

    - **Box in range** (the cue is on): it rebuilds the box's position and the target's from their distances and bearings, and
      the box's extent along the push from its relative heading (`(sin 2θ, cos 2θ)` is enough: the box is symmetric under a
      half turn). Behind the box with respect to the target, it **pushes**: full thrust toward a point ahead of the box on the
      line to the target. Otherwise it goes to the box's far side from the target, around the box if the box is in its way.
    - **Box out of range, but a ray sees it**: it turns toward that ray and drives.
    - **Nothing seen**: it **searches** by orbiting the target at `ORBIT` units, which passes within sensing range of any box the
      generator places 6 to 12 units from the target (design doc §4.2). It uses the target's distance and bearing, which every
      agent always has. Teammates are not used: a first version turned searching agents toward their teammates, and agents far
      from the box trapped each other in small spinning clusters.
    - **Obstacles**: a wall or an obstacle close ahead (not the box) makes it slow down and turn toward the freer side.

    `n` and `backend` say which batch it acts for, as `BoxPushingWorld.run_batch` asks of a policy; `sensing` and `params` must
    be the world's (for the layout, and to undo the normalisation of distances)."""

    LEAD = 3.0
    """How far ahead of the box, on the line to the target, a pusher aims."""
    STANDOFF = 0.6
    """How far behind the box's back face an agent gathers before pushing."""
    CLEARANCE = 0.8
    """How wide an agent goes around the box."""
    ORBIT = 9.0
    """The radius (units) of the search orbit around the target: the middle of the full difficulty's box–target distances."""
    ORBIT_GAIN = 0.5
    """How sharply a searching agent turns in (or out) per unit off its orbit, in radians."""
    AVOID = 0.18
    """A ray shorter than this (normalised) straight ahead, not on the box, means an obstacle to steer away from."""

    def __init__(
        self, n: int = 1, backend: Backend | None = None, sensing: SensingParams | None = None, params: WorldParams | None = None
    ) -> None:
        self.n = n
        self.backend = backend or Backend("numpy", "cpu", "float64")
        self.sensing = sensing or SensingParams()
        self.params = params or WorldParams()
        self.layout = observation_layout(self.sensing)

    def __repr__(self) -> str:
        return f"SensesController(n={self.n})"

    def act_batch(self, observations: Array) -> Array:
        xp = backend_of(observations).xp
        o, part = observations, self.layout
        sensing, p = self.sensing, self.params
        r, rays = sensing.range, sensing.rays

        def one(name: str, k: int = 0) -> Array:
            return o[..., part[name].start + k]

        # the target and the box, in the agent's frame (x ahead, y to the left)
        target_distance = one("target_distance") * p.arena.size * math.sqrt(2.0)
        tx, ty = target_distance * one("target_bearing", 1), target_distance * one("target_bearing", 0)
        on = one("box_flag") > 0.5
        box_distance = one("box_distance") * r
        bx, by = box_distance * one("box_bearing", 1), box_distance * one("box_bearing", 0)
        gx, gy = tx - bx, ty - by  # from the box to the target
        norm = xp.sqrt(gx * gx + gy * gy)
        safe = xp.where(norm > 0.0, norm, 1.0)
        gx, gy = xp.where(norm > 0.0, gx / safe, 1.0), xp.where(norm > 0.0, gy / safe, 0.0)
        px, py = -gy, gx  # across the push
        theta = 0.5 * xp.atan2(one("box_heading", 0), one("box_heading", 1))  # the box's axis, up to a half turn
        ux, uy = xp.cos(theta), xp.sin(theta)
        hx, hy = p.box.width / 2.0, p.box.height / 2.0
        depth = hx * xp.abs(gx * ux + gy * uy) + hy * xp.abs(gx * uy - gy * ux)
        width = hx * xp.abs(px * ux + py * uy) + hy * xp.abs(px * uy - py * ux)

        # where the agent is relative to the box: along the push (negative: behind) and across it
        along, side = -(bx * gx + by * gy), -(bx * px + by * py)
        radius = p.agent.radius
        pushing = (along < -depth) & (xp.abs(side) < width - 0.5 * radius)
        sign = xp.where(side >= 0.0, 1.0, -1.0)
        clear = width + radius + self.CLEARANCE
        ahead = along > -(depth + radius)
        in_way = ahead & (xp.abs(side) < clear)
        back = depth + radius + self.STANDOFF
        lane = xp.clip(side, -(width - radius), width - radius)  # behind the box, the agent's line across its back face
        # waypoints, relative to the box's centre (the box is at (bx, by) from the agent)
        wx = xp.where(in_way, px * sign * clear + gx * along, xp.where(ahead, px * sign * clear - gx * back, -gx * back + px * lane))
        wy = xp.where(in_way, py * sign * clear + gy * along, xp.where(ahead, py * sign * clear - gy * back, -gy * back + py * lane))
        aim_x = xp.where(pushing, bx + gx * self.LEAD, bx + wx)
        aim_y = xp.where(pushing, by + gy * self.LEAD, by + wy)

        # the box seen only by a ray: toward the nearest ray on it
        angles = 2.0 * math.pi * xp.arange(rays, dtype=o.dtype, device=self.backend.device) / rays
        ray_distance, ray_box = o[..., part["ray_distance"]], o[..., part["ray_box"]] > 0.5
        nearest_on_box = xp.argmin(xp.where(ray_box, ray_distance, 2.0), axis=-1)
        sees_box = xp.any(ray_box, axis=-1)
        ray_angle = xp.take(angles, xp.reshape(nearest_on_box, (-1,)), axis=0)
        ray_angle = xp.reshape(ray_angle, nearest_on_box.shape)

        # searching: orbit the target at ORBIT units, counter-clockwise, which passes within sensing range of any box that starts
        # 6 to 12 units from the target (design doc §4.2); beyond the orbit the agent heads in, inside it out
        bearing = xp.atan2(one("target_bearing", 0), one("target_bearing", 1))
        drift = xp.clip(self.ORBIT_GAIN * (target_distance - self.ORBIT), -math.pi / 2, math.pi / 2)
        search = bearing + math.pi / 2 - drift

        error = xp.where(on, xp.atan2(aim_y, aim_x), xp.where(sees_box, ray_angle, search))
        error = xp.atan2(xp.sin(error), xp.cos(error))
        turn = xp.clip(TURN_GAIN * error, -1.0, 1.0)
        thrust = xp.where(on & pushing, 1.0, xp.clip(xp.cos(error), 0.0, None))

        # obstacles close ahead (not the box): slow down and turn toward the freer side
        front = ray_distance[..., 0]
        left, right = ray_distance[..., 1], ray_distance[..., rays - 1]
        blocked = (front < self.AVOID) & ~ray_box[..., 0] & ~(on & pushing)
        turn = xp.where(blocked, xp.where(left >= right, 1.0, -1.0), turn)
        thrust = xp.where(blocked, 0.2, thrust)
        return xp.stack([thrust, turn], axis=-1)
