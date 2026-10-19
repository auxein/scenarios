"""The physics core: the state of every world of a batch, and the step that advances them all at once.

This is a **pure, batched physics core**, separate from Auxein's environment interface: `step(state, actions, layout, params)`
returns the next `State` of every candidate × scenario × agent at once, from actions `(n, s, N, 2)` in `[-1, 1]` (thrust and
turning). The environment (`environment.py`) wraps it, adding sensing and measurements; nothing here depends on sensors or networks,
so the physics is tested and watched with scripted controllers that read the true state (`scripted.py`).

- **Arrays.** Everything is written against the Array API namespace of the state's arrays (numpy or torch, float64 or float32,
  any device), with no Python loops over candidates, scenarios or agents: the loops are over substeps, over the (few) padded
  obstacle slots and over the four walls (`contacts.py`). No array is modified in place; finished scenarios are frozen with
  `where`.
- **Shapes.** A batch has `n` candidates and `s` scenarios, each with a team of `N` agents. The state holds `(n, s)` arrays for
  the box and `(n, s, N)` for the agents. The `Layout` (obstacles, target, starting poses) holds `(1, s, ...)` arrays, which
  broadcast over the candidates without a copy, or `(n, s, ...)`.
- **Integration.** Each episode step applies the actions for `substeps` physics substeps of `h = dt / substeps` with
  semi-implicit Euler: forces at the current positions, velocities updated first (drag, lateral damping and the turning lag
  implicitly, so they are stable at any rate), then positions from the new velocities.
- **The box's friction.** At rest, the box stays at rest while the net force on it is below `F_s` and the net
  torque below `τ_s`. Moving, kinetic friction removes `F_k h / M` of its speed and `τ_k h / I` of its spin per substep, never
  reversing either (both are clamped at zero), and the spin also decays at the angular damping rate; it comes to rest again
  when it is slow, barely turning and no longer pushed beyond the static thresholds.
- **Delivery**: at the end of a step, a scenario whose box centre is within the target radius is done. A done
  scenario keeps its shape in the arrays and **stops changing**: its state is frozen by `where`, and its step count stops.
- **Determinism.** For given inputs, backend, precision and device, the step gives the same numbers every time; and an entry's
  numbers do not depend on the rest of the batch or on how many obstacle slots the batch is padded to (`contacts.py`).
"""

from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np
from auxein.backend import Array, Backend, HostArray, backend_of

from auxein_scenarios.box_pushing import contacts
from auxein_scenarios.box_pushing.config import WorldParams

__all__ = [
    "Controller",
    "Hook",
    "Layout",
    "State",
    "Trajectory",
    "TrajectoryRecorder",
    "initial_state",
    "simulate",
    "step",
]


@dataclass(frozen=True)
class Layout:
    """What a scenario fixes for the whole episode, as arrays: `(1, s, ...)` (shared by all candidates) or `(n, s, ...)`.

    Obstacles are padded to a fixed number of slots per batch (`O_c` circles, `O_r` rectangles), with `*_valid` masks (1 for a
    real obstacle, 0 for padding, in the float dtype so that it multiplies forces). The starting poses are used by
    `initial_state` only. Built from scenarios by `scenarios.layout_arrays`."""

    circle_x: Array  # (..., O_c)
    circle_y: Array
    circle_r: Array
    circle_valid: Array
    rect_x: Array  # (..., O_r), centres and half-sides of axis-aligned rectangles
    rect_y: Array
    rect_hx: Array
    rect_hy: Array
    rect_valid: Array
    target_x: Array  # (...)
    target_y: Array
    box_x: Array  # (...) the box's starting pose
    box_y: Array
    box_angle: Array
    agent_x: Array  # (..., N) the agents' starting poses
    agent_y: Array
    agent_heading: Array

    @property
    def scenarios(self) -> int:
        return int(self.target_x.shape[-1])

    @property
    def agents(self) -> int:
        return int(self.agent_x.shape[-1])


@dataclass(frozen=True)
class State:
    """The state of every world of a batch: `(n, s, N)` arrays for the agents, `(n, s)` for the box and the episode."""

    agent_x: Array
    agent_y: Array
    agent_vx: Array
    agent_vy: Array
    agent_heading: Array
    agent_turn: Array
    """The agents' turning rates (rad/s), which lag behind the commanded rate."""
    box_x: Array
    box_y: Array
    box_angle: Array
    box_vx: Array
    box_vy: Array
    box_spin: Array
    box_at_rest: Array
    """Boolean: the box is held by static friction."""
    done: Array
    """Boolean: the box was delivered; the scenario is frozen."""
    steps: Array
    """Integer: the steps simulated before the scenario was done (it stops counting on delivery)."""

    @property
    def backend(self) -> Backend:
        return backend_of(self.box_x)


Controller = Callable[[State, Layout, WorldParams], Array]
"""A batched controller: the state (and the layout and parameters) to actions `(n, s, N, 2)` in `[-1, 1]`."""

Hook = Callable[[int, State], None]
"""Called by `simulate` with the step number and the state (after that step; step 0 is the initial state)."""


def initial_state(layout: Layout, n: int) -> State:
    """The state at the start of an episode for `n` candidates: the layout's starting poses, everything at rest.

    The starting poses are broadcast over the candidates, which is a view, not a copy, where the namespace allows; the first
    step makes new arrays anyway."""
    xp = backend_of(layout.box_x).xp
    s, agents = layout.scenarios, layout.agents
    team = (n, s, agents)
    zeros = xp.zeros(team, dtype=layout.agent_x.dtype, device=backend_of(layout.agent_x).device)
    box_zeros = zeros[..., 0]
    return State(
        agent_x=xp.broadcast_to(layout.agent_x, team),
        agent_y=xp.broadcast_to(layout.agent_y, team),
        agent_vx=zeros,
        agent_vy=zeros,
        agent_heading=xp.broadcast_to(layout.agent_heading, team),
        agent_turn=zeros,
        box_x=xp.broadcast_to(layout.box_x, (n, s)),
        box_y=xp.broadcast_to(layout.box_y, (n, s)),
        box_angle=xp.broadcast_to(layout.box_angle, (n, s)),
        box_vx=box_zeros,
        box_vy=box_zeros,
        box_spin=box_zeros,
        box_at_rest=box_zeros == 0.0,
        done=box_zeros != 0.0,
        steps=xp.astype(box_zeros, xp.int64),
    )


def step(state: State, actions: Array, layout: Layout, params: WorldParams) -> State:
    """One episode step for the whole batch: the actions `(n, s, N, 2)` (clipped to `[-1, 1]`) applied for every substep, then
    delivery. Done scenarios are left unchanged."""
    xp = backend_of(state.box_x).xp
    a = params.agent
    actions = xp.clip(actions, -1.0, 1.0)
    thrust_command, turn_command = actions[..., 0], actions[..., 1]
    thrust = xp.where(thrust_command >= 0.0, thrust_command, thrust_command * a.reverse_factor) * a.max_thrust
    turn_target = turn_command * a.max_turn_rate
    frozen = state.done
    for _ in range(params.simulation.substeps):
        state = _substep(state, thrust, turn_target, layout, params, frozen)
    radius = params.simulation.target_radius
    dx, dy = state.box_x - layout.target_x, state.box_y - layout.target_y
    delivered = dx * dx + dy * dy <= radius * radius
    return _replace(state, done=frozen | delivered, steps=state.steps + xp.astype(~frozen, state.steps.dtype))


def _substep(state: State, thrust: Array, turn_target: Array, layout: Layout, params: WorldParams, frozen: Array) -> State:
    xp = backend_of(state.box_x).xp
    a, b, f, c, d = params.agent, params.box, params.friction, params.contact, params.derived
    h, half = params.simulation.h, params.arena.half
    x, y, vx, vy = state.agent_x, state.agent_y, state.agent_vx, state.agent_vy
    box = contacts.BoxKinematics(
        x=state.box_x,
        y=state.box_y,
        c=xp.cos(state.box_angle),
        s=xp.sin(state.box_angle),
        vx=state.box_vx,
        vy=state.box_vy,
        spin=state.box_spin,
        hx=b.width / 2.0,
        hy=b.height / 2.0,
    )

    # forces on the agents: thrust along the heading, then every contact
    ux, uy = xp.cos(state.agent_heading), xp.sin(state.agent_heading)
    fx, fy = thrust * ux, thrust * uy
    k, r = c.stiffness, a.radius
    for gx, gy in (
        contacts.agent_agent(x, y, vx, vy, r, k, d.agent_damping),
        contacts.agent_circles(x, y, vx, vy, r, *_circles(layout), k, d.agent_static_damping),
        contacts.agent_rects(x, y, vx, vy, r, *_rects(layout), k, d.agent_static_damping),
        contacts.agent_walls(x, y, vx, vy, r, half, k, d.agent_static_damping),
    ):
        fx, fy = fx + gx, fy + gy
    gx, gy, box_fx, box_fy, box_torque = contacts.agent_box(x, y, vx, vy, r, box, k, d.agent_box_damping)
    fx, fy = fx + gx, fy + gy

    # forces on the box: the agents' reactions, then obstacles and walls (stiffer: the box is heavier)
    kb, cb = c.box_stiffness, d.box_static_damping
    for gx, gy, gt in (
        contacts.box_circles(box, *_circles(layout), kb, cb),
        contacts.box_rects(box, *_rects(layout), kb, cb),
        contacts.box_walls(box, half, kb, cb),
    ):
        box_fx, box_fy, box_torque = box_fx + gx, box_fy + gy, box_torque + gt

    # agents: semi-implicit Euler; drag along the heading and lateral damping across it, both implicit
    vx1, vy1 = vx + fx * (h / a.mass), vy + fy * (h / a.mass)
    along, across = vx1 * ux + vy1 * uy, vy1 * ux - vx1 * uy
    along = along / (1.0 + d.linear_drag * h / a.mass)
    across = across / (1.0 + (d.linear_drag / a.mass + a.lateral_damping) * h)
    vx1, vy1 = along * ux - across * uy, along * uy + across * ux
    lag = h / a.turn_time_constant
    turn1 = (state.agent_turn + turn_target * lag) / (1.0 + lag)
    heading1, x1, y1 = state.agent_heading + turn1 * h, x + vx1 * h, y + vy1 * h

    # the box: static friction holds it below the thresholds; kinetic friction and angular damping once it moves
    breaks = (box_fx * box_fx + box_fy * box_fy >= d.static_force**2) | (xp.abs(box_torque) >= d.static_torque)
    bvx, bvy = state.box_vx + box_fx * (h / b.mass), state.box_vy + box_fy * (h / b.mass)
    spin = state.box_spin + box_torque * (h / d.box_inertia)
    speed = xp.sqrt(bvx * bvx + bvy * bvy)
    moving_speed = speed > 0.0
    keep = xp.where(moving_speed, xp.clip(1.0 - d.kinetic_force * h / (b.mass * xp.where(moving_speed, speed, 1.0)), 0.0, None), 0.0)
    bvx, bvy = bvx * keep, bvy * keep
    # the kinetic torque takes τ_k h / I off the spin, never reversing it; then the viscous angular damping
    turning = xp.abs(spin)
    spin = xp.where(spin >= 0.0, 1.0, -1.0) * xp.clip(turning - d.kinetic_torque * h / d.box_inertia, 0.0, None)
    spin = spin / (1.0 + f.angular_damping * h)
    settled = (speed * keep < f.rest_speed) & (xp.abs(spin) < f.rest_spin) & ~breaks
    at_rest = xp.where(state.box_at_rest, ~breaks, settled)
    bvx, bvy, spin = (xp.where(at_rest, 0.0, v) for v in (bvx, bvy, spin))

    team = frozen[..., None]
    return _replace(
        state,
        agent_x=xp.where(team, x, x1),
        agent_y=xp.where(team, y, y1),
        agent_vx=xp.where(team, vx, vx1),
        agent_vy=xp.where(team, vy, vy1),
        agent_heading=xp.where(team, state.agent_heading, heading1),
        agent_turn=xp.where(team, state.agent_turn, turn1),
        box_x=xp.where(frozen, state.box_x, state.box_x + bvx * h),
        box_y=xp.where(frozen, state.box_y, state.box_y + bvy * h),
        box_angle=xp.where(frozen, state.box_angle, state.box_angle + spin * h),
        box_vx=xp.where(frozen, state.box_vx, bvx),
        box_vy=xp.where(frozen, state.box_vy, bvy),
        box_spin=xp.where(frozen, state.box_spin, spin),
        box_at_rest=xp.where(frozen, state.box_at_rest, at_rest),
    )


def _circles(layout: Layout) -> tuple[Array, Array, Array, Array]:
    return layout.circle_x, layout.circle_y, layout.circle_r, layout.circle_valid


def _rects(layout: Layout) -> tuple[Array, Array, Array, Array, Array]:
    return layout.rect_x, layout.rect_y, layout.rect_hx, layout.rect_hy, layout.rect_valid


def _replace(state: State, **changes: Array) -> State:
    values = {f.name: changes.get(f.name, getattr(state, f.name)) for f in fields(state)}
    return State(**values)


def simulate(
    state: State,
    layout: Layout,
    params: WorldParams,
    controller: Controller,
    steps: int,
    *,
    hook: Hook | None = None,
    every: int = 1,
    stop_when_done: bool = True,
) -> State:
    """Runs `steps` episode steps (the only Python loop is over time), calling `controller` for the actions of each.

    `hook`, if given, is called with the initial state (step 0), after every `every`-th step, and after the last step run.
    With `stop_when_done`, the run ends early once every scenario of every candidate is done (one host read per step)."""
    if every < 1:
        raise ValueError(f"every must be at least 1, got {every}")
    xp = backend_of(state.box_x).xp
    if hook is not None:
        hook(0, state)
    for t in range(1, steps + 1):
        state = step(state, controller(state, layout, params), layout, params)
        last = t == steps or (stop_when_done and bool(xp.all(state.done)))
        if hook is not None and (t % every == 0 or last):
            hook(t, state)
        if last:
            break
    return state


@dataclass(frozen=True)
class Trajectory:
    """The states of a simulation, recorded on the host: each field of `State` stacked over the recorded steps, `(T, n, s, ...)`,
    with `step` the episode step of each record."""

    step: HostArray
    agent_x: HostArray
    agent_y: HostArray
    agent_heading: HostArray
    box_x: HostArray
    box_y: HostArray
    box_angle: HostArray
    box_at_rest: HostArray
    done: HostArray

    @property
    def records(self) -> int:
        return int(self.step.shape[0])


class TrajectoryRecorder:
    """A `Hook` that copies the states it is given to host numpy arrays, for rendering: the renderer draws
    exactly what the physics computed. Use `trajectory()` once the simulation is over."""

    _FIELDS = ("agent_x", "agent_y", "agent_heading", "box_x", "box_y", "box_angle", "box_at_rest", "done")

    def __init__(self) -> None:
        self._steps: list[int] = []
        self._records: dict[str, list[HostArray]] = {name: [] for name in self._FIELDS}

    def __call__(self, t: int, state: State) -> None:
        backend = state.backend
        self._steps.append(t)
        for name in self._FIELDS:
            self._records[name].append(backend.to_numpy(getattr(state, name)))

    def trajectory(self) -> Trajectory:
        if not self._steps:
            raise ValueError("nothing was recorded")
        stacked = {name: np.stack(values) for name, values in self._records.items()}  # pyright: ignore[reportUnknownMemberType]
        return Trajectory(step=np.asarray(self._steps), **stacked)
