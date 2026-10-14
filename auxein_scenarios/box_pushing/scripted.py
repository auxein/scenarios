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
"""

import math

from auxein.backend import Array, backend_of

from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.world import Controller, Layout, State
from auxein_scenarios.common.geometry import wrap_angle

__all__ = ["constant", "go_to", "idle", "push", "steer"]

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


def push(goal_x: Array | None = None, goal_y: Array | None = None, *, spacing: float = 0.05, lead: float = 3.0) -> Controller:
    """The naive push team: lines behind the box, then a straight push toward the goal (the layout's target by default).

    The goal direction `g` runs from the box's centre to the goal. Agent `i` is assigned a **column**, a line parallel to `g`
    across the box's back face (`i mod columns`, with as many columns as fit across the box's longer side). An agent behind the
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
