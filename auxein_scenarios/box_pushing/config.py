"""The world's parameters (design doc §2): frozen dataclasses with the design's defaults, and the quantities derived from them.

Everything the physics needs is here, so that a world is fully described by one `WorldParams` value that can be written to JSON
(`to_dict` / `from_dict`) next to a run. **Derived quantities** (the friction thresholds, the box's inertia, the agents' drag,
the contact damping) are computed in one place, `WorldParams.derived`, and the **stability** of the penalty contacts against
the substep (design doc §2.6) is checked when the parameters are built, so that an unstable combination is a clear error before
anything runs rather than a NaN a thousand steps into a run.

TOML loading of the training configuration comes with training (M2); it will build these.
"""

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

__all__ = [
    "AgentParams",
    "ArenaParams",
    "BoxParams",
    "ContactParams",
    "Derived",
    "FrictionParams",
    "STABILITY_LIMIT",
    "SimulationParams",
    "WorldParams",
]

STABILITY_LIMIT = 1.0
"""The largest `ω h` allowed for a contact spring, with `ω = √(k / m)` and `h` the substep.

Semi-implicit Euler is stable for a single spring up to `ω h < 2`. Half of that leaves room for crowds, where a body pressed by
several neighbours feels the sum of their springs (a stiffer effective spring), and for the damping."""


@dataclass(frozen=True)
class ArenaParams:
    """The walled square: `size` is its side, centred on the origin."""

    size: float = 20.0

    @property
    def half(self) -> float:
        return self.size / 2.0


@dataclass(frozen=True)
class BoxParams:
    """The box: a rectangle `width × height` (along its own x and y axes) of uniform `mass`."""

    width: float = 2.0
    height: float = 1.2
    mass: float = 4.0


@dataclass(frozen=True)
class FrictionParams:
    """The box's friction with the floor (design doc §2.4).

    - `k`: the number of agents needed to move the box. The static threshold is `F_s = (k − margin) · F_a`, with `F_a` an
      agent's maximum thrust, so `k − 1` agents pushing in line cannot move it and `k` can.
    - `margin`: how far below `k` agents the threshold sits, in agents' thrusts (`0 < margin < 1`).
    - `torque_radius_factor`: the static torque threshold is `τ_s = F_s · ρ` with `ρ = (width + height) · factor`, about the
      mean distance of the box's footprint from its centre.
    - `kinetic_ratio` (`μ`): kinetic friction is `F_k = μ · F_s`, and the kinetic torque `τ_k = μ · τ_s`. It must be large
      enough that `k − 1` agents cannot keep a moving box moving, `μ (k − margin) > k − 1`, or a team that got the box moving
      once (with an impact, or a `k`th agent that leaves) would need one agent fewer from then on.
    - `angular_damping`: the rate (1/s) at which a moving box's spin decays, on top of the kinetic torque: `−rate · I · ω`.
      Without the kinetic torque, a moving box would keep turning under any torque, however small, and `k − 1` agents pushing
      off-centre could keep it turning (and drifting) forever once an impact had set it moving.
    - `rest_speed`, `rest_spin`: below both, and with the applied force and torque below the static thresholds, a moving box
      comes to rest again.
    """

    k: int = 4
    margin: float = 0.5
    torque_radius_factor: float = 1.0 / 6.0
    kinetic_ratio: float = 0.9
    angular_damping: float = 2.0
    rest_speed: float = 0.01
    rest_spin: float = 0.01


@dataclass(frozen=True)
class AgentParams:
    """The agents: discs with differential drive (design doc §2.5).

    - `max_thrust` (`F_a`): the forward force at action 1; reverse is `reverse_factor` of it.
    - `max_speed`: the terminal speed under full thrust, which sets the linear drag `F_a / max_speed` (`math.inf` turns drag off).
    - `max_turn_rate` (rad/s) at action ±1, reached through a first-order lag of time constant `turn_time_constant`.
    - `lateral_damping` (1/s, per unit mass): how strongly the velocity across the heading is damped, as wheels resist sliding.
    """

    radius: float = 0.25
    mass: float = 1.0
    max_thrust: float = 2.0
    reverse_factor: float = 0.5
    max_speed: float = 1.5
    max_turn_rate: float = math.pi
    turn_time_constant: float = 0.1
    lateral_damping: float = 20.0


@dataclass(frozen=True)
class ContactParams:
    """Penalty contacts (design doc §2.6): a spring `stiffness` and a `damping_ratio` (of critical damping) per contact.

    `stiffness` is for every contact involving an agent; `box_stiffness` for the box against walls and obstacles. The box is
    heavier, so the same stiffness would let a pushing crowd press it far into an obstacle; a stiffer spring keeps its overlap
    small at the same oscillation frequency.
    """

    stiffness: float = 200.0
    box_stiffness: float = 2000.0
    damping_ratio: float = 0.3


@dataclass(frozen=True)
class SimulationParams:
    """Time: `dt` per episode step (the policy acts once per step), `substeps` physics steps per episode step, the step limit,
    and the target zone's radius (delivery: the box's centre within it, design doc §4.4)."""

    dt: float = 0.05
    substeps: int = 2
    max_steps: int = 600
    target_radius: float = 1.5

    @property
    def h(self) -> float:
        """The physics substep."""
        return self.dt / self.substeps


@dataclass(frozen=True)
class Derived:
    """Quantities computed from the parameters, in one place (`WorldParams.derived`)."""

    static_force: float
    """`F_s`: the force the box needs to break loose."""
    static_torque: float
    """`τ_s`: the torque the box needs to break loose."""
    kinetic_force: float
    """`F_k`: the friction force on a moving box."""
    kinetic_torque: float
    """`τ_k`: the friction torque on a turning box."""
    box_inertia: float
    """The box's moment of inertia about its centre, `M (w² + h²) / 12`."""
    linear_drag: float
    """The agents' linear drag coefficient, `F_a / max_speed` (0 when the speed is unlimited)."""
    agent_damping: float
    """The damping coefficient of a contact between two agents (reduced mass `m / 2`)."""
    agent_static_damping: float
    """... between an agent and a wall or an obstacle (mass `m`)."""
    agent_box_damping: float
    """... between an agent and the box (reduced mass `m M / (m + M)`)."""
    box_static_damping: float
    """... between the box and a wall or an obstacle (mass `M`)."""
    stability: float
    """The largest `ω h` over the contact types, at most `STABILITY_LIMIT`."""


def _damping(ratio: float, stiffness: float, mass: float) -> float:
    return 2.0 * ratio * math.sqrt(stiffness * mass)


@dataclass(frozen=True)
class WorldParams:
    """Every parameter of the world, with the design doc's defaults. Built, it is valid: see `derived`."""

    arena: ArenaParams = field(default_factory=ArenaParams)
    box: BoxParams = field(default_factory=BoxParams)
    friction: FrictionParams = field(default_factory=FrictionParams)
    agent: AgentParams = field(default_factory=AgentParams)
    contact: ContactParams = field(default_factory=ContactParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(self, "derived", self._derive())

    derived: Derived = field(init=False, repr=False, compare=False)

    def _validate(self) -> None:
        f, a, b, c, sim = self.friction, self.agent, self.box, self.contact, self.simulation
        positive = {
            "arena.size": self.arena.size,
            "box.width": b.width,
            "box.height": b.height,
            "box.mass": b.mass,
            "agent.radius": a.radius,
            "agent.mass": a.mass,
            "agent.max_thrust": a.max_thrust,
            "agent.max_speed": a.max_speed,
            "agent.max_turn_rate": a.max_turn_rate,
            "agent.turn_time_constant": a.turn_time_constant,
            "contact.stiffness": c.stiffness,
            "contact.box_stiffness": c.box_stiffness,
            "simulation.dt": sim.dt,
            "simulation.target_radius": sim.target_radius,
            "friction.torque_radius_factor": f.torque_radius_factor,
        }
        for name, value in positive.items():
            if not value > 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if f.k < 1:
            raise ValueError(f"friction.k must be at least 1, got {f.k}")
        if not 0 < f.margin < 1:
            raise ValueError(f"friction.margin must be in (0, 1), got {f.margin}")
        if not 0 < f.kinetic_ratio <= 1:
            raise ValueError(f"friction.kinetic_ratio must be in (0, 1], got {f.kinetic_ratio}")
        if f.kinetic_ratio * (f.k - f.margin) <= f.k - 1:
            needed = (f.k - 1) / (f.k - f.margin)
            raise ValueError(
                f"friction.kinetic_ratio {f.kinetic_ratio} lets {f.k - 1} agents keep a moving box moving "
                f"(kinetic friction {f.kinetic_ratio * (f.k - f.margin):.3g} F_a is not above {f.k - 1} F_a): use more than {needed:.4g}"
            )
        if f.angular_damping < 0 or a.lateral_damping < 0 or c.damping_ratio < 0:
            raise ValueError("damping rates and ratios must be non-negative")
        if not 0 <= a.reverse_factor <= 1:
            raise ValueError(f"agent.reverse_factor must be in [0, 1], got {a.reverse_factor}")
        if sim.substeps < 1 or sim.max_steps < 1:
            raise ValueError(f"simulation.substeps and max_steps must be at least 1, got {sim.substeps} and {sim.max_steps}")
        if self.arena.size <= 4 * max(b.width, b.height):
            raise ValueError(f"the arena ({self.arena.size}) is too small for the box ({b.width} × {b.height})")

    def _derive(self) -> Derived:
        f, a, b, c, h = self.friction, self.agent, self.box, self.contact, self.simulation.h
        static_force = (f.k - f.margin) * a.max_thrust
        agent_pair_mass = a.mass / 2.0
        agent_box_mass = a.mass * b.mass / (a.mass + b.mass)
        frequencies = {
            "agent–agent": math.sqrt(c.stiffness / agent_pair_mass),
            "agent–wall or obstacle": math.sqrt(c.stiffness / a.mass),
            "agent–box": math.sqrt(c.stiffness / agent_box_mass),
            "box–wall or obstacle": math.sqrt(c.box_stiffness / b.mass),
        }
        worst = max(frequencies, key=lambda name: frequencies[name])
        stability = frequencies[worst] * h
        if stability > STABILITY_LIMIT:
            largest = STABILITY_LIMIT / frequencies[worst]
            raise ValueError(
                f"the {worst} contact is unstable: ω h = {stability:.3g} (ω = {frequencies[worst]:.3g} rad/s, substep h = {h:.3g} s) "
                f"exceeds {STABILITY_LIMIT}. Use a substep of at most {largest:.3g} s (more substeps or a smaller dt) or softer contacts."
            )
        return Derived(
            static_force=static_force,
            static_torque=static_force * (b.width + b.height) * f.torque_radius_factor,
            kinetic_force=f.kinetic_ratio * static_force,
            kinetic_torque=f.kinetic_ratio * static_force * (b.width + b.height) * f.torque_radius_factor,
            box_inertia=b.mass * (b.width**2 + b.height**2) / 12.0,
            linear_drag=0.0 if math.isinf(a.max_speed) else a.max_thrust / a.max_speed,
            agent_damping=_damping(c.damping_ratio, c.stiffness, agent_pair_mass),
            agent_static_damping=_damping(c.damping_ratio, c.stiffness, a.mass),
            agent_box_damping=_damping(c.damping_ratio, c.stiffness, agent_box_mass),
            box_static_damping=_damping(c.damping_ratio, c.box_stiffness, b.mass),
            stability=stability,
        )

    def replace(self, **sections: Any) -> "WorldParams":
        """A copy with some sections replaced, each given as a mapping of the fields to change, e.g.
        `params.replace(friction={"k": 3}, agent={"max_speed": math.inf})`. The copy is validated like any new one."""
        known = {f.name for f in fields(self) if f.init}
        values: dict[str, Any] = {}
        for name in known:
            current = getattr(self, name)
            change = sections.pop(name, None)
            values[name] = current if change is None else type(current)(**{**asdict(current), **dict(change)})
        if sections:
            raise ValueError(f"unknown sections {sorted(sections)}: use {sorted(known)}")
        return WorldParams(**values)

    def to_dict(self) -> dict[str, dict[str, object]]:
        """The parameters as a JSON-ready dict of sections (the derived quantities are not included: they are recomputed).
        An unlimited `max_speed` is written as the string `"inf"`, since JSON has no infinity."""
        out: dict[str, dict[str, object]] = {}
        for section in fields(self):
            if section.init:
                values: dict[str, object] = asdict(getattr(self, section.name))
                out[section.name] = {k: ("inf" if isinstance(v, float) and math.isinf(v) else v) for k, v in values.items()}
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldParams":
        """The inverse of `to_dict`; missing sections and fields take their defaults, unknown ones are an error."""
        return cls().replace(**{name: {k: _float_or_inf(v) for k, v in section.items()} for name, section in data.items()})


def _float_or_inf(value: Any) -> Any:
    return math.inf if value == "inf" else value
