"""Judging brains on scenarios: the measurements of each episode, their summaries, and the two baselines.

`train` uses it to judge its snapshot brains on the run's held-out set (`held_out.csv`, for the generalisation chart), and
`deploy` to judge any brains on any scenarios. Everything runs through the world's batched step loop (`run_batch`), the path
training scores with, so a brain is judged exactly as it was trained:

- **An episode** (`Episode`) is one subject (a brain or a baseline) on one scenario: whether the box was delivered, at which
  step, how far it moved toward the target (`progress`, clipped at the fitness's floor, as the score is), the most agents
  touching the box at once (`max_pushers`), and the scenario's facts (the box's starting distance to the target, the obstacles).
- **A summary** (`summarise`) is over the episodes that ran: success, mean progress, mean `max_pushers`, the median delivery
  step of the delivered ones; `bands` splits success by the box's starting distance (6–8, 8–10, 10–12 units for full
  scenarios), because distance is what most decides whether a push finishes in time.
- **Two baselines** put a brain's numbers in context: **random brains** (`BASELINE_BRAINS` networks sampled from the brain's
  genome space with a fixed seed, as a run's first generation is), which show what an untrained team does, and the
  **senses-only controller**, a hand-written memoryless policy on the same observation, which shows what these senses allow.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from auxein.backend import Backend
from auxein.environments import MLPPolicy, Scenario
from auxein.random import RandomStream

from auxein_scenarios.box_pushing import fitness, scripted
from auxein_scenarios.box_pushing.config import FitnessConfig, WorldParams
from auxein_scenarios.box_pushing.environment import BoxPushingWorld

__all__ = [
    "BANDS",
    "BASELINE_BRAINS",
    "BASELINE_SEED",
    "Episode",
    "Results",
    "Scorer",
    "bands",
    "baselines",
    "episodes",
    "measure",
    "scorer",
    "summarise",
]

BASELINE_BRAINS = 20
"""How many random brains make the random baseline: enough for a stable mean, a fifth of a generation's cost."""
BASELINE_SEED = 0
"""The random brains' seed, fixed so that the baseline is the same every time."""

BANDS: tuple[tuple[str, float, float], ...] = (("6-8", 6.0, 8.0), ("8-10", 8.0, 10.0), ("10-12", 10.0, math.inf))
"""Success by the box's starting distance to the target: thirds of the full difficulty's range (the last takes anything from
10 up); a distance below 6 (an easy scenario) falls in no band."""

Measurements = Mapping[str, float]
Scorer = Callable[[Measurements], float]


@dataclass(frozen=True)
class Episode:
    """One subject on one scenario: the scenario's facts and the measurements read from the episode."""

    scenario_id: str
    difficulty: str
    distance: float
    """The box's starting distance to the target."""
    obstacles: int
    ok: bool
    """False for a failed episode (a non-finite state): its measurements are meaningless and left out of the summaries."""
    delivered: bool
    steps: int
    progress: float
    """`(d₀ − d) / d₀`, clipped at the fitness's floor, as the score and the `progress` descriptor are."""
    max_pushers: int
    score: float


@dataclass(frozen=True)
class Results:
    """A summary over a set of episodes. Means over the episodes that ran (`ok`); `failed` counts the others."""

    episodes: int
    failed: int
    delivered: int
    success: float
    progress: float
    max_pushers: float
    delivery_steps: float
    """The median delivery step of the delivered episodes, NaN when none was delivered."""
    score: float


def summarise(chosen: Sequence[Episode]) -> Results:
    ok = [e for e in chosen if e.ok]
    delivered = [e for e in ok if e.delivered]

    def mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else math.nan

    return Results(
        episodes=len(chosen),
        failed=len(chosen) - len(ok),
        delivered=len(delivered),
        success=mean([1.0 if e.delivered else 0.0 for e in ok]),
        progress=mean([e.progress for e in ok]),
        max_pushers=mean([float(e.max_pushers) for e in ok]),
        delivery_steps=float(np.median([e.steps for e in delivered])) if delivered else math.nan,
        score=mean([e.score for e in ok]),
    )


def bands(chosen: Sequence[Episode]) -> list[tuple[str, int, int]]:
    """`(band, delivered, episodes)` for each of `BANDS`, over the episodes that ran."""
    out: list[tuple[str, int, int]] = []
    for name, lo, hi in BANDS:
        inside = [e for e in chosen if e.ok and lo <= e.distance < hi]
        out.append((name, sum(e.delivered for e in inside), len(inside)))
    return out


def scorer(weights: FitnessConfig, max_steps: int) -> Scorer:
    """The score of one episode's measurements with these weights (`fitness.EpisodeScore`, on the host)."""
    episode_score = fitness.EpisodeScore(weights.weights, max_steps)

    def score(m: Measurements) -> float:
        arrays = {name: np.asarray([[value]], dtype=np.float64) for name, value in m.items()}
        return float(np.asarray(episode_score(cast("Any", arrays)))[0, 0])

    return score


def _facts(scenario: Scenario) -> tuple[str, str, float, int]:
    p = scenario.params
    box, target = cast("list[float]", p["box"]), cast("list[float]", p["target"])
    obstacles = len(cast("list[object]", p["circles"])) + len(cast("list[object]", p["rects"]))
    return scenario.id, str(p["difficulty"]), math.dist(box[:2], target), obstacles


def episodes(scenarios: Sequence[Scenario], measurements: Sequence[Measurements | None], score: Scorer) -> tuple[Episode, ...]:
    """One subject's episodes on `scenarios`, from its measurements per scenario (None for a failed episode)."""
    floor = fitness.Weights().progress_floor
    out: list[Episode] = []
    for scenario, m in zip(scenarios, measurements, strict=True):
        scenario_id, difficulty, distance, obstacles = _facts(scenario)
        if m is None:
            out.append(Episode(scenario_id, difficulty, distance, obstacles, False, False, 0, math.nan, 0, math.nan))
            continue
        d0, d = m["initial_distance"], m["final_distance"]
        progress = max(floor, (d0 - d) / d0) if d0 > 0 else 0.0
        out.append(
            Episode(
                scenario_id, difficulty, distance, obstacles, True, m["delivered"] > 0.5, int(m["steps"]), progress,
                int(m["max_pushers"]), score(m),
            )
        )  # fmt: skip
    return tuple(out)


def measure(world: BoxPushingWorld, agents: Any, scenarios: Sequence[Scenario], backend: Backend) -> list[list[dict[str, float] | None]]:
    """Every candidate of `agents` (a batch policy) on every scenario, through `run_batch`: per candidate, per scenario, the
    measurements on the host (None for a failed episode)."""
    result = world.run_batch(agents, tuple(scenarios), RandomStream(np.random.SeedSequence(0), backend))
    host = {name: np.asarray(backend.to_numpy(v), dtype=np.float64) for name, v in result.measurements.items()}
    n, s = next(iter(host.values())).shape
    failed = set(result.failures)
    return [
        [None if (i, j) in failed else {name: float(values[i, j]) for name, values in host.items()} for j in range(s)] for i in range(n)
    ]


def baselines(
    world: BoxPushingWorld, policy: MLPPolicy, scenarios: Sequence[Scenario], backend: Backend, score: Scorer
) -> dict[str, list[tuple[Episode, ...]]]:
    """The two baselines (module docstring) on `scenarios` in `world`: per baseline, the episodes of each of its members (the
    random brains, then the one controller). `policy` gives the random brains' genome space."""
    genomes = policy.space().sample_genomes(BASELINE_BRAINS, RandomStream(np.random.SeedSequence(BASELINE_SEED), backend), backend)
    params: WorldParams = world.params
    senses = scripted.SensesController(1, backend, world.sensing, params)
    return {
        "random brains": [episodes(scenarios, m, score) for m in measure(world, policy.decode_batch(genomes), scenarios, backend)],
        "senses-only controller": [episodes(scenarios, m, score) for m in measure(world, senses, scenarios, backend)],
    }
