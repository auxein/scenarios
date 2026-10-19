"""Fitness: the per-episode score from the six raw measurements, and the Auxein `Aggregator` around it.

The world returns raw measurements (`environment.MEASUREMENTS`), never a score; **the score lives here, in the aggregator**, so
that a recorded run can be re-judged with other weights (`RunReader.reaggregate`) without re-simulating. Per episode, with `d₀`
and `d` the box's initial and final distance to the target, `t` the delivery step and `T` the step limit:

    score = w_p · max(−1, (d₀ − d) / d₀)                      progress, clipped at −1
          + delivered · (w_d + w_t · (1 − t / T))             delivery and time bonuses
          + w_a · approach                                    approach, in [0, 1]

with the default weights `w_p = 1`, `w_d = 2`, `w_t = 1`, `w_a = 0.1`. The run's objective is the **mean score over the
scenarios, maximised**; the descriptors are the success rate (mean of `delivered`), the mean progress, the mean approach and the
mean `max_pushers`.

**The approach term cannot dominate.** It is at most `w_a = 0.1`, so a team that only gathers around the box scores at most 0.1,
and any team that moves the box 10 % of the way to the target beats it whatever its approach. In practice the agents moving the
box are at the box too (their approach is near 1), so even a small push beats gathering (a test checks it with scripted teams).

**The score's label shows its weights** (`EpisodeScore.describe`), so the run's description, which Auxein compares on resume,
says which weights scored it, and a changed weight is refused on resume.
"""

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from auxein.aggregators import Aggregator, mean
from auxein.backend import Array, backend_of
from auxein.core import Objective

__all__ = ["DESCRIPTORS", "OBJECTIVE", "EpisodeProgress", "EpisodeScore", "Weights", "aggregator"]

OBJECTIVE = Objective("score", "maximise")
"""The run's one objective."""

DESCRIPTORS = ("success_rate", "progress", "approach", "max_pushers")
"""The run's descriptors, each a mean over the scenarios."""


@dataclass(frozen=True)
class Weights:
    """The weights of the per-episode score, and the floor of the progress term."""

    progress: float = 1.0
    delivery: float = 2.0
    time: float = 1.0
    approach: float = 0.1
    progress_floor: float = -1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"fitness.{name} must be finite, got {value}")
        for name in ("progress", "delivery", "time", "approach"):
            if getattr(self, name) < 0:
                raise ValueError(f"fitness.{name} must not be negative, got {getattr(self, name)}")
        if self.progress_floor > 0:
            raise ValueError(f"fitness.progress_floor must be at most 0, got {self.progress_floor}")


def _progress(measurements: Mapping[str, Array], floor: float) -> Array:
    initial, final = measurements["initial_distance"], measurements["final_distance"]
    xp = backend_of(initial).xp
    there = initial > 0.0  # the generator never starts the box at the target; a hand-built layout might
    progress = xp.where(there, (initial - final) / xp.where(there, initial, 1.0), 0.0)
    return xp.clip(progress, floor, None)


class EpisodeProgress:
    """The progress of the box toward the target, `(d₀ − d) / d₀` clipped at `floor`: an aggregator source, `(n, s)` in and out."""

    def __init__(self, floor: float = -1.0) -> None:
        self.floor = floor

    def __call__(self, measurements: Mapping[str, Array]) -> Array:
        return _progress(measurements, self.floor)

    def describe(self) -> str:
        return f"episode_progress(floor={self.floor!r})"

    def __repr__(self) -> str:
        return self.describe()


class EpisodeScore:
    """The per-episode score (module docstring), as an aggregator source: the dict of `(n, s)` measurement arrays to the
    `(n, s)` array of scores, in their namespace and on their device. `max_steps` is the episode's step limit `T`, which the
    time bonus divides by. Its description lists the weights and `T`."""

    def __init__(self, weights: Weights, max_steps: int) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        self.weights, self.max_steps = weights, max_steps

    def __call__(self, measurements: Mapping[str, Array]) -> Array:
        w = self.weights
        xp = backend_of(measurements["delivered"]).xp
        delivered = measurements["delivered"]
        early = 1.0 - xp.clip(measurements["steps"] / self.max_steps, 0.0, 1.0)
        bonus = delivered * (w.delivery + w.time * early)
        return w.progress * _progress(measurements, w.progress_floor) + bonus + w.approach * measurements["approach"]

    def describe(self) -> str:
        w = self.weights
        return (
            f"episode_score(progress={w.progress!r}, delivery={w.delivery!r}, time={w.time!r}, approach={w.approach!r}, "
            f"progress_floor={w.progress_floor!r}, max_steps={self.max_steps})"
        )

    def __repr__(self) -> str:
        return self.describe()


def aggregator(weights: Weights | None = None, max_steps: int = 900) -> Aggregator:
    """The run's aggregator: the objective `score` (the mean episode score, maximised: `OBJECTIVE`) and the `DESCRIPTORS`."""
    w = weights or Weights()
    return Aggregator(
        objectives={OBJECTIVE.name: mean(EpisodeScore(w, max_steps))},
        descriptors={
            "success_rate": mean("delivered"),
            "progress": mean(EpisodeProgress(w.progress_floor)),
            "approach": mean("approach"),
            "max_pushers": mean("max_pushers"),
        },
    )
