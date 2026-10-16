"""The world's parameters: the design's defaults, the derived quantities computed in one place, and the checks at construction;
and the training configuration (M2a): TOML, defaults, errors naming the key, the round trip, the presets."""

import json
import math
import re
from pathlib import Path

import pytest

from auxein_scenarios.box_pushing.config import (
    STABILITY_LIMIT,
    ConfigError,
    SensingParams,
    TrainingConfig,
    WorldParams,
    config_differences,
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


# --- the training configuration (M2a) ---

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_training_defaults_are_the_design_docs() -> None:
    c = TrainingConfig()
    assert (c.team.agents, c.strategy.population, c.strategy.offspring) == (40, 100, 100)
    assert (c.scenarios.easy, c.scenarios.full, c.scenarios.held_out) == (4, 12, 64)
    assert c.max_steps == 900 and c.world == WorldParams() and c.sensing == SensingParams()
    assert (c.fitness.progress, c.fitness.delivery, c.fitness.time, c.fitness.approach, c.fitness.progress_floor) == (
        1.0,
        2.0,
        1.0,
        0.1,
        -1.0,
    )
    assert (c.brain.hidden, c.brain.scale) == ((32,), 1.0)
    assert (c.run.backend, c.run.device, c.run.precision, c.run.checkpoint_every) == ("torch", "mps", "float32", 10)
    assert c.budget.generations == 400 and c.evaluations == 100 + 399 * 100


def test_a_partial_config_overrides_only_what_it_lists() -> None:
    c = parse_config(
        """
        name = "try"
        [world.friction]
        k = 3
        [world.simulation]
        max_steps = 700
        [strategy]
        population = 20
        [sensing]
        ray_range = 8
        """
    )
    default = TrainingConfig()
    assert c.name == "try" and c.world.friction.k == 3 and c.max_steps == 700 and c.strategy.population == 20
    assert c.sensing.reach == 8.0 and c.sensing.rays == default.sensing.rays
    assert c.world.agent == default.world.agent and c.strategy.offspring == 100 and c.team == default.team
    assert config_differences(default, c) == [
        "sensing.ray_range: recorded None, given 8.0",
        "strategy.population: recorded 100, given 20",
        "world.friction.k: recorded 4, given 3",
        "world.simulation.max_steps: recorded 900, given 700",
    ]


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("[strategy]\npopulaton = 10", "strategy.populaton"),
        ("[world.frction]\nk = 3", "world.frction"),
        ("colour = 1", "colour"),
        ("[team]\nagents = 'forty'", "team.agents"),
        ("[team]\nagents = 2.5", "team.agents"),
        ("[brain]\nhidden = 32", "brain.hidden"),
        ("[brain]\nhidden = [32, 'x']", "brain.hidden[1]"),
        ("[run]\nbackend = 'jax'", "run.backend"),
        ("[run]\nbackend = 'numpy'\ndevice = 'mps'", "run.device"),
        ("[fitness]\napproach = -1.0", "fitness.approach"),
        ("[budget]\ngenerations = 0", "budget.generations"),
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
        text = dump_config(config)
        assert parse_config(text) == config
        assert "ray_range" not in text  # None is left out: TOML has no null
    assert math.isinf(parse_config(dump_config(changed)).world.agent.max_speed)


def test_config_differences_ignore_the_budget() -> None:
    a = TrainingConfig()
    assert config_differences(a, a.with_generations(900)) == []
    b = parse_config("[fitness]\ntime = 2.0\n[team]\nagents = 20")
    assert config_differences(a, b) == ["fitness.time: recorded 1.0, given 2.0", "team.agents: recorded 40, given 20"]


@pytest.mark.parametrize("preset", ["smoke", "pilot", "full"])
def test_the_presets_load(preset: str) -> None:
    config = load_config(CONFIGS / f"{preset}.toml")
    assert config.name == preset
    if preset == "smoke":
        assert (config.run.backend, config.team.agents, config.strategy.population, config.budget.generations) == ("numpy", 10, 10, 3)
    else:
        assert (config.team.agents, config.strategy.population, config.scenarios.easy + config.scenarios.full) == (40, 100, 16)
        assert (config.run.backend, config.run.device, config.max_steps) == ("torch", "mps", 900)
    assert config.budget.generations == {"smoke": 3, "pilot": 25, "full": 400}[preset]
