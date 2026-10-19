"""The world through Auxein: `EpisodeEvaluator` on both paths, a recorded two-generation run, and determinism end to end.

The aggregator here is a **placeholder** (the mean final distance, minimised), so that these tests depend on the world and Auxein
only, not on the fitness."""

import warnings
from pathlib import Path
from typing import Any

import auxein
import numpy as np
import pytest
from auxein.aggregators import Aggregator, mean
from auxein.backend import Backend
from auxein.core import Objective, Status
from auxein.environments import ScenarioSet
from auxein.random import RandomStream
from auxein.recording.reader import open_run

from auxein_scenarios.box_pushing import scenarios
from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.environment import MEASUREMENTS, BoxPushingWorld, brain
from tests.box_pushing.support import host

PARAMS = WorldParams()
AGENTS = 10
SET = scenarios.training_set(AGENTS, PARAMS, seed=1, mix=((scenarios.EASY, 2), (scenarios.FULL, 1)))
POLICY = brain()


def placeholder() -> Aggregator:
    """The tests' stand-in for the fitness's aggregator: the mean final distance of the box to the target, over the scenarios."""
    return Aggregator(objectives={"final_distance": mean("final_distance")}, descriptors={"delivered": mean("delivered")})


class PerEpisode:
    """`MLPPolicy` without `decode_batch`, so that `EpisodeEvaluator` takes the per-episode path."""

    def decode(self, genome: Any) -> Any:
        return POLICY.decode(genome)

    def __repr__(self) -> str:
        return "PerEpisode(brain)"


def run(backend: Backend, run_dir: Path | None, *, batched: bool = True, generations: int = 2) -> auxein.RunResult[Any]:
    world = BoxPushingWorld(PARAMS, steps=40, backend=None if batched else backend)
    evaluator = auxein.EpisodeEvaluator(POLICY if batched else PerEpisode(), world, SET, placeholder())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", auxein.RecordingDisabledWarning)
        return auxein.run(
            strategy=auxein.RandomSearch(),
            evaluator=evaluator,
            space=POLICY.space(),
            objectives=[Objective("final_distance", "minimise")],
            descriptors=["delivered"],
            budget=auxein.Budget(evaluations=4 * generations),
            batch_size=4,
            seed=3,
            backend=backend,
            run_dir=run_dir,
        )


def test_a_recorded_two_generation_run(corner_backend: Backend, tmp_path: Path) -> None:
    result = run(corner_backend, tmp_path / "run")
    assert result.best is not None and result.best.status is Status.OK
    reader = open_run(tmp_path / "run")
    try:
        assert len(reader.generation_history()) == 2
        episodes = reader.episodes(result.best.candidate.id)
        assert [e.scenario_id for e in episodes] == list(SET.ids)
        for episode in episodes:
            assert episode.status is Status.OK and set(episode.measurements) == set(MEASUREMENTS)
            assert episode.measurements["steps"] == 40.0
    finally:
        reader.close()


def test_the_evaluators_two_paths_agree() -> None:
    """`EpisodeEvaluator` on the batched path (`decode_batch`, `run_batch`) and the per-episode one (`decode`, `run_episode`)
    find the same best candidate with the same objective, along the same trace (numpy float64: the brain's products are a
    few ulps apart, grouped differently)."""
    backend = Backend("numpy", "cpu", "float64")
    batched, per_episode = run(backend, None), run(backend, None, batched=False)
    assert batched.best is not None and per_episode.best is not None
    assert batched.best.candidate.id == per_episode.best.candidate.id
    assert batched.best.objectives["final_distance"] == pytest.approx(per_episode.best.objectives["final_distance"], rel=1e-7)
    assert [count for count, _ in batched.trace] == [count for count, _ in per_episode.trace]
    np.testing.assert_allclose([v for _, v in batched.trace], [v for _, v in per_episode.trace], rtol=1e-7)


# --- determinism end to end ---


def measurements(backend: Backend, world: BoxPushingWorld, genomes: Any, scenario_set: Any) -> dict[str, np.ndarray]:
    stream = RandomStream(np.random.SeedSequence(0), backend)
    result = world.run_batch(POLICY.decode_batch(genomes), tuple(scenario_set), stream)
    return {name: host(backend, value) for name, value in result.measurements.items()}


def tolerance(backend: Backend) -> float:
    """Bit for bit on numpy. On torch float32 a candidate's numbers may depend on the batch's shape (a CPU or GPU kernel can
    block the brain's matrix products differently), and the contacts carry that on: within 1e-3, as the physics' own tests
    allow (on a CPU they are usually identical too)."""
    return 0.0 if backend.name == "numpy" else 1e-3


@pytest.fixture
def genomes(corner_backend: Backend) -> Any:
    return POLICY.space().sample_genomes(4, RandomStream(np.random.SeedSequence(11), corner_backend), corner_backend)


def test_the_same_genomes_give_the_same_measurements_twice(corner_backend: Backend, genomes: Any) -> None:
    first = measurements(corner_backend, BoxPushingWorld(steps=80), genomes, SET)
    second = measurements(corner_backend, BoxPushingWorld(steps=80), genomes, SET)
    for name in MEASUREMENTS:
        assert np.array_equal(first[name], second[name]), name


def test_a_candidate_alone_or_in_a_batch(corner_backend: Backend, genomes: Any) -> None:
    world = BoxPushingWorld(steps=80)
    together = measurements(corner_backend, world, genomes, SET)
    for row in (0, 3):
        alone = measurements(corner_backend, world, genomes[row : row + 1], SET)
        for name in MEASUREMENTS:
            np.testing.assert_allclose(
                alone[name][0], together[name][row], rtol=tolerance(corner_backend), atol=tolerance(corner_backend), err_msg=name
            )


def test_scenario_order_does_not_change_a_scenarios_result(corner_backend: Backend, genomes: Any) -> None:
    """Up to the last bit on numpy: reordering the scenarios moves each one's agents in memory, and on x86 numpy's vectorised
    mean over the agents can round a different last bit depending on where a row starts (1e-16 in `approach`)."""
    world = BoxPushingWorld(steps=80)
    forward = measurements(corner_backend, world, genomes, SET)
    reversed_set = ScenarioSet.from_params([s.params for s in reversed(SET)], seed=1, ids=list(reversed(SET.ids)))
    backward = measurements(corner_backend, world, genomes, reversed_set)
    for name in MEASUREMENTS:
        rtol = tolerance(corner_backend) or 1e-12
        np.testing.assert_allclose(backward[name][:, ::-1], forward[name], rtol=rtol, atol=tolerance(corner_backend), err_msg=name)
