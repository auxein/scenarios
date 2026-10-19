"""The world's parameters: their defaults, the derived quantities computed in one place, and the checks at construction; and the
training configuration: TOML, defaults, every key overriding, removed keys and errors naming the key, the round trip, the
held-out seed, the presets."""

import json
import math
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from auxein_scenarios.box_pushing import scenarios, train
from auxein_scenarios.box_pushing.config import (
    REMOVED,
    STABILITY_LIMIT,
    ConfigError,
    SensingParams,
    TrainingConfig,
    WorldParams,
    dump_config,
    load_config,
    parse_config,
)


def test_defaults_and_derived_quantities() -> None:
    p = WorldParams()
    d = p.derived
    assert (p.arena.size, p.box.width, p.box.height, p.agent.radius) == (20.0, 2.0, 1.2, 0.25)
    assert (p.simulation.dt, p.simulation.substeps, p.simulation.max_steps, p.simulation.target_radius) == (0.05, 2, 900, 1.5)
    assert d.static_force == pytest.approx((4 - 0.5) * 2.0)
    assert d.static_torque == pytest.approx(d.static_force * (2.0 + 1.2) / 6.0)
    assert d.kinetic_force == pytest.approx(0.9 * d.static_force)
    assert d.kinetic_torque == pytest.approx(0.9 * d.static_torque)
    assert d.box_inertia == pytest.approx(4.0 * (2.0**2 + 1.2**2) / 12.0)
    assert d.linear_drag == pytest.approx(2.0 / 1.5)
    assert 0 < d.stability <= STABILITY_LIMIT


def test_k_minus_one_agents_cannot_move_the_box_and_k_can() -> None:
    for k in (2, 3, 4):
        p = WorldParams().replace(friction={"k": k})
        thrust = p.agent.max_thrust
        assert (k - 1) * thrust < p.derived.static_force < k * thrust
        assert (k - 1) * thrust < p.derived.kinetic_force  # k - 1 cannot keep it moving either


def test_unstable_contacts_are_refused() -> None:
    with pytest.raises(ValueError, match="unstable.*substep of at most"):
        WorldParams().replace(simulation={"substeps": 1, "dt": 0.1})
    with pytest.raises(ValueError, match="box–wall or obstacle"):
        WorldParams().replace(contact={"box_stiffness": 1e5})


def test_kinetic_friction_that_lets_k_minus_one_keep_the_box_moving_is_refused() -> None:
    with pytest.raises(ValueError, match="lets 3 agents keep a moving box moving"):
        WorldParams().replace(friction={"kinetic_ratio": 0.8})
    with pytest.raises(ValueError, match="lets 5 agents"):
        WorldParams().replace(friction={"k": 6})  # 0.9 is not enough for k = 6: the message says what is


@pytest.mark.parametrize(
    "change",
    [{"box": {"mass": 0.0}}, {"friction": {"k": 0}}, {"friction": {"margin": 1.0}}, {"agent": {"reverse_factor": 2.0}}],
)
def test_invalid_values_are_refused(change: dict[str, dict[str, float]]) -> None:
    with pytest.raises(ValueError):
        WorldParams().replace(**change)


def test_json_round_trip_with_unlimited_speed() -> None:
    p = WorldParams().replace(agent={"max_speed": math.inf, "lateral_damping": 0.0}, friction={"k": 3})
    data = json.loads(json.dumps(p.to_dict()))
    assert data["agent"]["max_speed"] == "inf"
    assert WorldParams.from_dict(data) == p
    assert WorldParams.from_dict(data).derived.linear_drag == 0.0
    assert WorldParams.from_dict({}) == WorldParams()


def test_unknown_sections_and_fields_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown sections"):
        WorldParams().replace(gravity={"g": 9.8})
    with pytest.raises(TypeError):
        WorldParams().replace(box={"colour": 1.0})


# --- the training configuration ---

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_training_defaults() -> None:
    c = TrainingConfig()
    assert (c.team.agents, c.brain.hidden, c.sensing) == (40, (32,), SensingParams(range=5.0))
    assert (c.sensing.rays, c.sensing.sectors, c.sensing.reach) == (12, 8, 5.0)
    assert (c.scenarios.easy, c.scenarios.full, c.scenarios.held_out, c.scenarios.seed, c.scenarios.held_out_seed) == (4, 28, 64, 1, 2)
    assert (c.fitness.progress, c.fitness.delivery, c.fitness.time, c.fitness.approach) == (1.0, 2.0, 1.0, 0.1)
    assert c.fitness.weights.progress_floor == -1.0
    assert (c.evolution.population, c.evolution.generations) == (100, 400) and c.evaluations == 100 * 400
    assert (c.run.seed, c.run.backend, c.run.device, c.run.precision) == (1, "torch", "mps", "float32")
    assert c.max_steps == 900 and c.world == WorldParams()


def _leaves(table: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in table.items():
        if isinstance(value, dict):
            out.update(_leaves(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


KEYS = _leaves(tomllib.loads(dump_config(TrainingConfig())))
SPECIAL: dict[str, tuple[str, Any]] = {
    "name": ('name = "other"', "other"),
    "run.backend": ('[run]\nbackend = "numpy"\ndevice = "cpu"', "numpy"),
    "run.device": ('[run]\ndevice = "cpu"', "cpu"),
    "run.precision": ('[run]\nprecision = "float64"', "float64"),
    "brain.hidden": ("[brain]\nhidden = [16, 8]", [16, 8]),
    "world.friction.k": ("[world.friction]\nk = 3", 3),
    "world.friction.margin": ("[world.friction]\nmargin = 0.4", 0.4),
    "world.friction.kinetic_ratio": ("[world.friction]\nkinetic_ratio = 0.95", 0.95),
    "world.agent.reverse_factor": ("[world.agent]\nreverse_factor = 0.4", 0.4),
    "world.contact.stiffness": ("[world.contact]\nstiffness = 150.0", 150.0),
    "world.contact.box_stiffness": ("[world.contact]\nbox_stiffness = 1500.0", 1500.0),
    "world.contact.damping_ratio": ("[world.contact]\ndamping_ratio = 0.2", 0.2),
    "world.simulation.dt": ("[world.simulation]\ndt = 0.04", 0.04),
}


def _override(key: str) -> tuple[str, Any]:
    if key in SPECIAL:
        return SPECIAL[key]
    value = KEYS[key]
    new = value + 1 if isinstance(value, int) else value * 1.1
    table, _, leaf = key.rpartition(".")
    return f"[{table}]\n{leaf} = {new!r}", new


def test_every_key_is_listed() -> None:
    """The keys a config can set: all of the world's physics, and these, no more."""
    assert {k for k in KEYS if not k.startswith("world.")} == {
        "name", "sensing.range", "team.agents", "brain.hidden", "scenarios.easy", "scenarios.full", "scenarios.held_out",
        "scenarios.seed", "fitness.progress", "fitness.delivery", "fitness.time", "fitness.approach", "evolution.population",
        "evolution.generations", "run.seed", "run.backend", "run.device", "run.precision",
    }  # fmt: skip
    assert {k.split(".")[1] for k in KEYS if k.startswith("world.")} == {"arena", "box", "friction", "agent", "contact", "simulation"}


@pytest.mark.parametrize("key", sorted(KEYS))
def test_every_key_overrides(key: str) -> None:
    text, expected = _override(key)
    changed = _leaves(tomllib.loads(dump_config(parse_config(text))))
    assert changed[key] == pytest.approx(expected) if isinstance(expected, float) else changed[key] == expected
    assert {k for k in KEYS if changed[k] != KEYS[k]} <= {key, "run.device"}, "only the key changes (numpy's device is cpu)"


@pytest.mark.parametrize(
    "text",
    [
        "[scenarios]\ngenerator = 2",
        "[scenarios]\nheld_out_seed = 2",
        "[strategy]\npopulation = 10",
        "[budget]\ngenerations = 3",
        "[sensing]\nrays = 12",
        "[sensing]\nsectors = 8",
        "[sensing]\nray_range = 8.0",
        "[brain]\nscale = 1.0",
        "[run]\ncheckpoint_every = 1",
        "[fitness]\nprogress_floor = -1.0",
    ],
)
def test_removed_keys_are_errors_naming_the_key(text: str) -> None:
    table = tomllib.loads(text)
    (section,) = table
    (leaf,) = table[section]
    key = section if section in ("strategy", "budget") else f"{section}.{leaf}"
    with pytest.raises(ConfigError, match=re.escape(f"{key!r} is not a config key: {REMOVED[key]}")):
        parse_config(text)


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("[evolution]\npopulaton = 10", "evolution.populaton"),
        ("[world.frction]\nk = 3", "world.frction"),
        ("colour = 1", "colour"),
        ("[team]\nagents = 'forty'", "team.agents"),
        ("[team]\nagents = 2.5", "team.agents"),
        ("[brain]\nhidden = 32", "brain.hidden"),
        ("[brain]\nhidden = [32, 'x']", "brain.hidden[1]"),
        ("[brain]\nhidden = []", "brain.hidden"),
        ("[run]\nbackend = 'jax'", "run.backend"),
        ("[run]\nbackend = 'numpy'\ndevice = 'mps'", "run.device"),
        ("[fitness]\napproach = -1.0", "fitness.approach"),
        ("[evolution]\ngenerations = 0", "evolution.generations"),
        ("[evolution]\npopulation = 1", "evolution.population"),
        ("[sensing]\nrange = 0.0", "sensing.range"),
        ("[world.friction]\nk = 0", "friction.k"),
        ("[world.simulation]\ndt = 1.0", "contact is unstable"),
        ("[team]\nagents = true", "team.agents"),
        ("team = 3", "team"),
    ],
)
def test_bad_configs_are_errors_naming_the_key(text: str, key: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(key)):
        parse_config(text)


def test_not_toml_is_an_error() -> None:
    with pytest.raises(ConfigError, match="not valid TOML"):
        parse_config("[team\nagents = 3")


def test_the_effective_config_round_trips_through_toml() -> None:
    changed = parse_config("[world.agent]\nmax_speed = inf\n[brain]\nhidden = [16, 8]\n[fitness]\napproach = 0.05\n", name="x")
    for config in (TrainingConfig(), changed):
        assert parse_config(dump_config(config)) == config
    assert math.isinf(parse_config(dump_config(changed)).world.agent.max_speed)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_the_held_out_seed_is_derived_and_its_set_disjoint(seed: int) -> None:
    """The held-out set comes from `seed + 1` on a stream of its own: never a training layout."""
    config = parse_config(f"[team]\nagents = 6\n[scenarios]\neasy = 2\nfull = 6\nheld_out = 8\nseed = {seed}")
    assert config.scenarios.held_out_seed == seed + 1
    training, held = train.training_set(config), train.held_out_set(config)
    scenarios.check_disjoint(training, held)
    direct = scenarios.held_out_set(6, config.world, seed=seed + 1, count=8)
    assert held.fingerprint == direct.fingerprint and all(s.params["difficulty"] == "full" for s in held)


@pytest.mark.parametrize("preset", ["smoke", "pilot", "full"])
def test_the_presets_load(preset: str) -> None:
    config = load_config(CONFIGS / f"{preset}.toml")
    assert config.name == preset
    if preset == "smoke":
        assert (config.run.backend, config.team.agents, config.evolution.population, config.evolution.generations) == ("numpy", 10, 10, 3)
    else:
        assert (config.team.agents, config.evolution.population, config.scenarios.easy, config.scenarios.full) == (40, 100, 4, 28)
        assert (config.run.backend, config.run.device, config.max_steps, config.scenarios.held_out) == ("torch", "mps", 900, 64)
        assert config.evolution.generations == {"pilot": 25, "full": 400}[preset]
