"""Fitness: the per-episode score against hand computations, gathering against pushing, and the aggregator."""

import math

import numpy as np
import pytest
from auxein.backend import Backend
from auxein.environments import ScenarioSet

from auxein_scenarios.box_pushing import scripted
from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.environment import BoxPushingWorld
from auxein_scenarios.box_pushing.fitness import DESCRIPTORS, OBJECTIVE, EpisodeScore, Weights, aggregator
from auxein_scenarios.box_pushing.scenarios import layout_params
from tests.box_pushing.support import FACING_X, StatePolicy, host

PARAMS = WorldParams()


def measurements(backend: Backend, **columns: list[float]) -> dict[str, object]:
    """`(1, s)` measurement arrays, one scenario per value."""
    return {name: backend.asarray(np.asarray([values], dtype=np.float64)) for name, values in columns.items()}


# four episodes: delivered early, delivered late, not delivered, pushed away
EPISODES = {
    "delivered": [1.0, 1.0, 0.0, 0.0],
    "steps": [300.0, 880.0, 900.0, 900.0],
    "initial_distance": [10.0, 8.0, 10.0, 4.0],
    "final_distance": [1.0, 1.2, 7.0, 12.0],
    "approach": [0.9, 0.95, 0.5, 0.2],
    "max_pushers": [5.0, 4.0, 2.0, 1.0],
}
EXPECTED = [
    0.9 + 2.0 + 1.0 * (1.0 - 300.0 / 900.0) + 0.1 * 0.9,  # progress 0.9, delivered at a third of the limit
    0.85 + 2.0 + 1.0 * (1.0 - 880.0 / 900.0) + 0.1 * 0.95,  # progress 6.8 / 8, delivered at the last moment
    0.3 + 0.1 * 0.5,  # progress 0.3, not delivered
    -1.0 + 0.1 * 0.2,  # pushed from 4 to 12: progress −2, clipped at −1
]


def test_the_episode_score_against_hand_computations(backend: Backend) -> None:
    score = EpisodeScore(Weights(), 900)(measurements(backend, **EPISODES))
    tolerance = 1e-12 if backend.precision == "float64" else 1e-6
    np.testing.assert_allclose(host(backend, score)[0], EXPECTED, atol=tolerance)


def test_the_weights_and_the_floor_apply(backend: Backend) -> None:
    weights = Weights(progress=2.0, delivery=0.0, time=0.0, approach=0.0, progress_floor=-0.5)
    score = host(backend, EpisodeScore(weights, 900)(measurements(backend, **EPISODES)))[0]
    np.testing.assert_allclose(score, [1.8, 1.7, 0.6, -1.0], atol=1e-6)


def test_the_aggregator(backend: Backend) -> None:
    assert OBJECTIVE.name == "score" and OBJECTIVE.direction == "maximise"
    # two candidates with the same four episodes, the second in reverse order: the same means
    two = {name: backend.asarray(np.asarray([values, values[::-1]], dtype=np.float64)) for name, values in EPISODES.items()}
    aggregated = aggregator(Weights(), 900).aggregate(two, backend)
    assert set(aggregated.objectives) == {"score"} and set(aggregated.descriptors) == set(DESCRIPTORS)
    np.testing.assert_allclose(aggregated.objectives["score"], [np.mean(EXPECTED)] * 2, atol=1e-6)
    np.testing.assert_allclose(aggregated.descriptors["success_rate"], [0.5, 0.5])
    np.testing.assert_allclose(aggregated.descriptors["progress"], [np.mean([0.9, 0.85, 0.3, -1.0])] * 2, atol=1e-6)
    np.testing.assert_allclose(aggregated.descriptors["approach"], [np.mean(EPISODES["approach"])] * 2, atol=1e-6)
    np.testing.assert_allclose(aggregated.descriptors["max_pushers"], [3.0, 3.0])


def test_the_label_shows_the_weights() -> None:
    text = aggregator(Weights(), 900).describe()
    assert "episode_score(progress=1.0, delivery=2.0, time=1.0, approach=0.1, progress_floor=-1.0, max_steps=900)" in text
    assert aggregator(Weights(approach=0.2), 900).describe() != text
    assert aggregator(Weights(), 600).describe() != text


def test_weights_are_validated() -> None:
    with pytest.raises(ValueError, match="fitness.approach"):
        Weights(approach=-0.1)
    with pytest.raises(ValueError, match="fitness.progress_floor"):
        Weights(progress_floor=0.5)
    with pytest.raises(ValueError, match="fitness.time"):
        Weights(time=math.nan)


def test_gathering_scores_less_than_moving_the_box_a_little(corner_backend: Backend) -> None:
    """The approach term must not dominate, or a team could score by gathering without pushing. Twelve agents that only gather
    around the box (all touching it, approach 1, the most the term gives) against the push team in the same scenario for 50
    steps (2.5 s), which moves the box about a third of a unit, 4 % of the way. A push wins once its progress exceeds
    0.1 × (the gatherers' approach − its own): after 40 steps (1.4 % of the way, the back of the column still far from the
    box) it does not yet."""
    r, half_long, half_short = PARAMS.agent.radius, PARAMS.box.width / 2.0, PARAMS.box.height / 2.0
    # the box faces x: its long faces at x = ±0.6, its ends at y = ±1
    ring = [[-(half_short + r), y, 0.0] for y in (-0.7, -0.2, 0.3, 0.8)] + [[half_short + r, y, math.pi] for y in (-0.7, -0.2, 0.3, 0.8)]
    ring += [[x, s * (half_long + r), -s * math.pi / 2] for x in (-0.3, 0.3) for s in (1.0, -1.0)]
    start = [[-(half_short + r) - 0.6 * i, 0.5 * ((i % 4) - 1.5), 0.0] for i in range(12)]
    gathering = layout_params([0.0, 0.0, FACING_X], [8.0, 0.0], ring)
    pushing = layout_params([0.0, 0.0, FACING_X], [8.0, 0.0], start)
    world = BoxPushingWorld(PARAMS, steps=50)
    scores = []
    for layout, controller in ((gathering, scripted.idle), (pushing, scripted.push())):
        arrays = world.layout(tuple(ScenarioSet.from_params([layout], ids=["s"])), corner_backend)
        policy = StatePolicy(controller, arrays, PARAMS)
        rollout = world.run(policy.act_batch, arrays, 1, hook=policy.hook)
        scores.append(float(host(corner_backend, EpisodeScore(Weights(), 900)(rollout.measurements))[0, 0]))
        if controller is scripted.idle:
            assert float(host(corner_backend, rollout.measurements["approach"])[0, 0]) == pytest.approx(1.0)
        else:
            moved = 8.0 - float(host(corner_backend, rollout.measurements["final_distance"])[0, 0])
            assert 0.1 < moved < 0.5, f"the push should move the box a little, it moved {moved:.2f}"
    gathered, pushed = scores
    assert gathered == pytest.approx(0.1, abs=1e-6)
    assert pushed > gathered, f"gathering scored {gathered:.3f}, a short push {pushed:.3f}"
