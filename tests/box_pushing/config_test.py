"""The world's parameters: the design's defaults, the derived quantities computed in one place, and the checks at construction."""

import json
import math

import pytest

from auxein_scenarios.box_pushing.config import STABILITY_LIMIT, WorldParams


def test_defaults_and_derived_quantities() -> None:
    p = WorldParams()
    d = p.derived
    assert (p.arena.size, p.box.width, p.box.height, p.agent.radius) == (20.0, 2.0, 1.2, 0.25)
    assert (p.simulation.dt, p.simulation.substeps, p.simulation.max_steps, p.simulation.target_radius) == (0.05, 2, 600, 1.5)
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
