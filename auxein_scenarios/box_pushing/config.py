"""The world's parameters (design doc §2): frozen dataclasses with the design's defaults, and the quantities derived from them.

Everything the physics needs is here, so that a world is fully described by one `WorldParams` value that can be written to JSON
(`to_dict` / `from_dict`) next to a run. **Derived quantities** (the friction thresholds, the box's inertia, the agents' drag,
the contact damping) are computed in one place, `WorldParams.derived`, and the **stability** of the penalty contacts against
the substep (design doc §2.6) is checked when the parameters are built, so that an unstable combination is a clear error before
anything runs rather than a NaN a thousand steps into a run.

`SensingParams` (M1b) describes what each agent senses (design doc §3): the number of rays and of teammate sectors and the
sensing range. It sits beside `WorldParams` rather than inside it, because the physics never reads it: the same world can be
sensed differently, and the observation's size depends on it alone (`sensors.observation_layout`).

**The training configuration** (M2a, design doc §6): `TrainingConfig`, the whole definition of a run, loaded from TOML
(`load_config`, `parse_config`) and written back with every default filled in (`dump_config`). Every field has the design
doc's default, so a config lists only what it changes; unknown keys and bad values are errors that name the key.

```toml
name = "pilot"                       # the run directory's prefix (default: the file's name)
[world.simulation]                   # any field of WorldParams, by section
max_steps = 900
[sensing]                            # SensingParams
[team]
agents = 40
[brain]
hidden = [32]
scale = 1.0                          # MLPPolicy.space(scale)
[scenarios]
easy = 4                             # the training set: 4 easy and 12 full, fixed for the run
full = 12
seed = 1
held_out = 64                        # the held-out set (for after the run)
held_out_seed = 2
[fitness]                            # fitness.Weights
[strategy]                           # GeneticAlgorithm
population = 100
offspring = 100
[budget]
generations = 400
[run]
seed = 1
backend = "torch"                    # torch or numpy
device = "mps"                       # mps, cuda or cpu
precision = "float32"
checkpoint_every = 10                # generations
```
"""

import json
import math
import tomllib
import types
import typing
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from auxein_scenarios.box_pushing.fitness import Weights

__all__ = [
    "AgentParams",
    "ArenaParams",
    "BoxParams",
    "ContactParams",
    "Derived",
    "FrictionParams",
    "STABILITY_LIMIT",
    "BrainConfig",
    "BudgetConfig",
    "ConfigError",
    "LEGACY_GENERATOR",
    "RunConfig",
    "ScenariosConfig",
    "SensingParams",
    "SimulationParams",
    "StrategyConfig",
    "TeamConfig",
    "TrainingConfig",
    "config_differences",
    "dump_config",
    "load_config",
    "parse_config",
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
    and the target zone's radius (delivery: the box's centre within it, design doc §4.4).

    The step limit is 900 steps (45 s), raised at M2a from 600: four agents move the box at about 0.3 units/s, so 600 steps
    left a team little time to find the box and push it up to 12 units (design doc §10.2)."""

    dt: float = 0.05
    substeps: int = 2
    max_steps: int = 900
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


@dataclass(frozen=True)
class SensingParams:
    """What each agent senses (design doc §3), with the design's defaults.

    - `rays`: range rays, evenly spaced around the agent, the first straight ahead.
    - `range` (`r`): the sensing range. The box cue is on within it (the box's centre closer than `r`), teammates are counted
      within it, and distances are normalised by it.
    - `ray_range`: how far the rays reach; `None` (the default) is `range`. A longer ray range lets rays see the box beyond the
      cue, one of the levers of design doc §10.1.
    - `sectors`: teammate sectors around the agent, the first centred straight ahead.

    Sensing runs once per episode step, on the state at the start of the step (not per physics substep).
    """

    rays: int = 12
    range: float = 5.0
    ray_range: float | None = None
    sectors: int = 8

    def __post_init__(self) -> None:
        if self.rays < 1 or self.sectors < 1:
            raise ValueError(f"rays and sectors must be at least 1, got {self.rays} and {self.sectors}")
        if not self.range > 0 or (self.ray_range is not None and not self.ray_range > 0):
            raise ValueError(f"range and ray_range must be positive, got {self.range} and {self.ray_range}")

    @property
    def reach(self) -> float:
        """The rays' range: `ray_range`, or `range` when it is not set."""
        return self.range if self.ray_range is None else self.ray_range

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SensingParams":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown sensing parameters {sorted(unknown)}: use {sorted(known)}")
        return cls(**data)


# --- the training configuration (M2a) ---


class ConfigError(ValueError):
    """A training configuration that cannot be used: an unknown key, a value of the wrong type or out of range."""


@dataclass(frozen=True)
class TeamConfig:
    """The team: `agents` per scenario (design doc §4.4)."""

    agents: int = 40

    def __post_init__(self) -> None:
        if self.agents < 1:
            raise ValueError(f"team.agents must be at least 1, got {self.agents}")


@dataclass(frozen=True)
class BrainConfig:
    """The shared brain (design doc §3.6): the widths of the hidden layers, and the `scale` of its genome space
    (`MLPPolicy.space(scale)`: weights within `±scale / √fan_in`, biases within `±scale`)."""

    hidden: tuple[int, ...] = (32,)
    scale: float = 1.0

    def __post_init__(self) -> None:
        if any(w < 1 for w in self.hidden):
            raise ValueError(f"brain.hidden widths must be at least 1, got {list(self.hidden)}")
        if not (math.isfinite(self.scale) and self.scale > 0):
            raise ValueError(f"brain.scale must be positive, got {self.scale}")


@dataclass(frozen=True)
class ScenariosConfig:
    """The scenario sets (design doc §4.2–4.3): the training set, `easy` then `full` scenarios from `seed`, **fixed for the whole
    run** (parents and children are judged on the same scenarios); the held-out set, `held_out` full scenarios from
    `held_out_seed`, used only after the run.

    `generator` is the version of the scenario generator (`scenarios.GENERATORS`): 2, the default, keeps the box clear of
    walls and obstacles; 1 is the generator before M2b. **A config file without it means 1** (`parse_config`), unlike every
    other missing key, which takes today's default: runs recorded before M2b have no `generator` in their `config.toml`, and
    must rebuild exactly the scenarios they were trained and judged on."""

    easy: int = 4
    full: int = 28
    seed: int = 1
    held_out: int = 64
    held_out_seed: int = 2
    generator: int = 2

    def __post_init__(self) -> None:
        if self.easy < 0 or self.full < 0 or self.easy + self.full < 1:
            raise ValueError(f"scenarios.easy and scenarios.full must not be negative and not both 0, got {self.easy} and {self.full}")
        if self.held_out < 1:
            raise ValueError(f"scenarios.held_out must be at least 1, got {self.held_out}")
        if self.seed < 0 or self.held_out_seed < 0:
            raise ValueError("scenarios.seed and scenarios.held_out_seed must not be negative")
        if self.generator not in (1, 2):
            raise ValueError(f"scenarios.generator must be 1 (before M2b) or 2 (box clearance), got {self.generator}")


@dataclass(frozen=True)
class StrategyConfig:
    """Auxein's `GeneticAlgorithm` (design doc §6): `population` parents and `offspring` children per generation, plus
    selection, tournaments of `tournament_size`, intermediate recombination with `crossover_probability`, and self-adaptive
    mutation starting at `initial_step` (relative to the box's width)."""

    population: int = 100
    offspring: int = 100
    crossover_probability: float = 1.0
    tournament_size: int = 2
    initial_step: float = 0.1

    def __post_init__(self) -> None:
        if self.population < 2:
            raise ValueError(f"strategy.population must be at least 2, got {self.population}")
        if self.offspring < 1:
            raise ValueError(f"strategy.offspring must be at least 1, got {self.offspring}")
        if not 0.0 <= self.crossover_probability <= 1.0:
            raise ValueError(f"strategy.crossover_probability must be in [0, 1], got {self.crossover_probability}")
        if self.tournament_size < 1:
            raise ValueError(f"strategy.tournament_size must be at least 1, got {self.tournament_size}")
        if not self.initial_step > 0:
            raise ValueError(f"strategy.initial_step must be positive, got {self.initial_step}")


@dataclass(frozen=True)
class BudgetConfig:
    """How long the run is, in generations: the initial population, then `generations − 1` generations of children. The only
    thing a resume may change."""

    generations: int = 400

    def __post_init__(self) -> None:
        if self.generations < 1:
            raise ValueError(f"budget.generations must be at least 1, got {self.generations}")


BACKENDS = ("numpy", "torch")
PRECISIONS = ("float32", "float64")


@dataclass(frozen=True)
class RunConfig:
    """The run: its `seed`, where it computes (torch float32 on Metal by default; design doc §6), and how often it checkpoints
    (`checkpoint_every` generations)."""

    seed: int = 1
    backend: str = "torch"
    device: str = "mps"
    precision: str = "float32"
    checkpoint_every: int = 10

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"run.seed must not be negative, got {self.seed}")
        if self.backend not in BACKENDS:
            raise ValueError(f"run.backend must be one of {list(BACKENDS)}, got {self.backend!r}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"run.precision must be one of {list(PRECISIONS)}, got {self.precision!r}")
        if self.backend == "numpy" and self.device != "cpu":
            raise ValueError(f"run.device must be 'cpu' for numpy, got {self.device!r}")
        if self.checkpoint_every < 1:
            raise ValueError(f"run.checkpoint_every must be at least 1 generation, got {self.checkpoint_every}")


@dataclass(frozen=True)
class TrainingConfig:
    """Everything that defines a training run (module docstring), with the design doc's defaults."""

    name: str = "box-pushing"
    world: WorldParams = field(default_factory=WorldParams)
    sensing: SensingParams = field(default_factory=SensingParams)
    team: TeamConfig = field(default_factory=TeamConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    scenarios: ScenariosConfig = field(default_factory=ScenariosConfig)
    fitness: Weights = field(default_factory=Weights)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def __post_init__(self) -> None:
        if not self.name or any(c in self.name for c in "/\\"):
            raise ValueError(f"name must be a non-empty name without slashes, got {self.name!r}")

    @property
    def evaluations(self) -> int:
        """The evaluation budget: the initial population, then `generations − 1` generations of `offspring` children."""
        return self.strategy.population + (self.budget.generations - 1) * self.strategy.offspring

    @property
    def max_steps(self) -> int:
        return self.world.simulation.max_steps

    def with_generations(self, generations: int) -> "TrainingConfig":
        """The same run with another budget (the one change a resume allows)."""
        return _replace_dataclass(self, budget=BudgetConfig(generations))


def _replace_dataclass(value: Any, **changes: Any) -> Any:
    return type(value)(**{**{f.name: getattr(value, f.name) for f in fields(value) if f.init}, **changes})


def _coerce(hint: Any, value: Any, key: str) -> Any:
    """`value` as the type `hint`, or a `ConfigError` naming `key`. TOML gives ints, floats, strings, booleans and lists."""
    origin, arguments = typing.get_origin(hint), typing.get_args(hint)
    if origin in (types.UnionType, typing.Union):
        if value is None and type(None) in arguments:
            return None
        (inner,) = [a for a in arguments if a is not type(None)]
        return _coerce(inner, value, key)
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{key} must be true or false, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{key} must be an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{key} must be a number, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{key} must be a string, got {value!r}")
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{key} must be a list, got {value!r}")
        items = typing.cast("list[Any] | tuple[Any, ...]", value)
        return tuple(_coerce(arguments[0], item, f"{key}[{i}]") for i, item in enumerate(items))
    raise ConfigError(f"{key} has a type the config cannot read: {hint}")  # pragma: no cover - a programming error


def _build(cls: type[Any], data: object, path: str) -> Any:
    """A dataclass of type `cls` from a TOML table, recursively; defaults for what is missing, errors naming the key."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path or 'the config'} must be a table, got {data!r}")
    table = typing.cast("Mapping[str, Any]", data)
    hints = typing.get_type_hints(cls)
    known = [f.name for f in fields(cls) if f.init]
    for key in table:
        if key not in known:
            where = f"{path}.{key}" if path else key
            raise ConfigError(f"unknown key {where!r}: {path or 'the config'} has {known}")
    values: dict[str, Any] = {}
    for name in known:
        if name not in table:
            continue
        key = f"{path}.{name}" if path else name
        hint: Any = hints[name]
        nested = isinstance(hint, type) and is_dataclass(hint)
        values[name] = _build(typing.cast("type[Any]", hint), table[name], key) if nested else _coerce(hint, table[name], key)
    try:
        return cls(**values)
    except ValueError as error:
        message = str(error)
        raise ConfigError(message if path == "" or message.startswith(f"{path}.") else f"{path}: {message}") from None


LEGACY_GENERATOR = 1
"""What a config file without `scenarios.generator` means: the generator before M2b (see `ScenariosConfig`)."""


def parse_config(text: str, *, name: str | None = None) -> TrainingConfig:
    """A `TrainingConfig` from TOML text; `name` is the default for its `name` (e.g. the file's name)."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"the config is not valid TOML: {error}") from None
    if name is not None and "name" not in data:
        data["name"] = name
    scenarios = data.setdefault("scenarios", {})
    if isinstance(scenarios, dict) and "generator" not in scenarios:
        scenarios["generator"] = LEGACY_GENERATOR  # a config from before M2b: its scenarios are generator 1's
    return typing.cast("TrainingConfig", _build(TrainingConfig, data, ""))


def load_config(path: str | Path) -> TrainingConfig:
    """A `TrainingConfig` from a TOML file; its `name` defaults to the file's name without `.toml`."""
    file = Path(path)
    return parse_config(file.read_text(), name=file.stem)


def _flat(value: Any, prefix: str = "") -> dict[str, Any]:
    """A dataclass as `{dotted key: value}`, leaves only."""
    out: dict[str, Any] = {}
    for f in fields(value):
        if not f.init:
            continue
        item, key = getattr(value, f.name), f"{prefix}{f.name}"
        if is_dataclass(item):
            out.update(_flat(item, f"{key}."))
        else:
            out[key] = item
    return out


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return "inf" if value == math.inf else "-inf" if value == -math.inf else "nan" if math.isnan(value) else repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # a TOML basic string: the same escapes as JSON for what names hold
    if isinstance(value, tuple):
        return "[" + ", ".join(_toml_value(v) for v in typing.cast("tuple[Any, ...]", value)) + "]"
    raise TypeError(f"cannot write {value!r} to TOML")  # pragma: no cover - a programming error


def dump_config(config: TrainingConfig) -> str:
    """The effective config as TOML, every field written (defaults included), so that it defines the run on its own. A field
    that is `None` (`sensing.ray_range`, meaning "the sensing range") is left out, as TOML has no null."""
    lines = [f"name = {_toml_value(config.name)}"]
    tables: dict[str, list[str]] = {}
    for key, value in _flat(config).items():
        if key == "name" or value is None:
            continue
        table, _, leaf = key.rpartition(".")
        tables.setdefault(table, []).append(f"{leaf} = {_toml_value(value)}")
    for table, entries in tables.items():
        lines += ["", f"[{table}]", *entries]
    return "\n".join(lines) + "\n"


def config_differences(recorded: TrainingConfig, given: TrainingConfig, *, ignore: tuple[str, ...] = ("budget.", "name")) -> list[str]:
    """The keys whose values differ between two configs, as `key: recorded …, given …` lines, except those under `ignore`."""
    a, b = _flat(recorded), _flat(given)
    return [
        f"{key}: recorded {a.get(key)!r}, given {b.get(key)!r}"
        for key in sorted(set(a) | set(b))
        if not any(key == i or key.startswith(i) for i in ignore) and a.get(key) != b.get(key)
    ]
