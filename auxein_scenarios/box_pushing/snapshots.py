"""Snapshots: the generations of a run worth keeping a brain of, and their champions.

**Snapshots** are log-spaced: 1, 2, 5, 10, 20, 50, 100, 200, 500, … (1, 2 and 5 times the powers of ten), **plus the last
generation**: early generations change fast, so they get more of them. A snapshot whose champion is **the same candidate** as
the previous snapshot's is skipped: it would be the same brain twice.

The champion of snapshot `k` is `RunReader.best_up_to(k)`: the run's best so far at generation `k`. A run saves the brain of
each snapshot as it reaches it (`brains/gen-<k>.safetensors`) and judges them all on its held-out set at the end
(`held_out.csv`), so the files and the chart talk about the same generations. Auxein's `champions(every=n)` is arithmetic
(every `n`th generation), so the log spacing is computed here.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from auxein.recording.reader import RecordedEvaluation, open_run

__all__ = ["Snapshot", "SnapshotError", "default_generations", "find_snapshots", "is_snapshot", "select"]


class SnapshotError(ValueError):
    """A list of generations that the run cannot answer: empty, out of range, or a run with no champion yet."""


@dataclass(frozen=True)
class Snapshot:
    """A generation to show and its champion (the run's best up to it)."""

    generation: int
    champion: RecordedEvaluation

    @property
    def candidate_id(self) -> int:
        return int(self.champion.candidate_id)

    @property
    def score(self) -> float:
        return float(self.champion.objectives["score"])


def default_generations(last: int) -> list[int]:
    """1, 2, 5, 10, 20, 50, … up to `last`, and `last` itself: log-spaced, so the fast early changes get more snapshots."""
    if last < 1:
        raise SnapshotError(f"the run has no generation yet (last = {last})")
    out: list[int] = []
    decade = 1
    while decade <= last:
        out += [m * decade for m in (1, 2, 5) if m * decade <= last]
        decade *= 10
    return sorted({*out, last})


def is_snapshot(generation: int) -> bool:
    """Whether `generation` is on the log-spaced schedule (1, 2 or 5 times a power of ten), before the skipping of unchanged
    champions; the last generation is a snapshot too, which only the end of a run knows."""
    decade = 1
    while decade <= generation:
        if generation in (decade, 2 * decade, 5 * decade):
            return True
        decade *= 10
    return False


def select(generations: Sequence[int], champions: Sequence[RecordedEvaluation | None]) -> list[Snapshot]:
    """The snapshots from generations (sorted, distinct) and their champions: a generation with no champion yet (nothing
    succeeded) is left out, and one whose champion is the previous snapshot's is skipped."""
    out: list[Snapshot] = []
    for generation, champion in zip(generations, champions, strict=True):
        if champion is None:
            continue
        if out and int(out[-1].champion.candidate_id) == int(champion.candidate_id):
            continue
        out.append(Snapshot(generation, champion))
    return out


def find_snapshots(recording: str | Path, generations: Sequence[int] | None = None) -> list[Snapshot]:
    """The snapshots of the recorded run in `recording` (Auxein's run directory): `generations` (default: the log-spaced
    list and the last), sorted, each with `best_up_to(k)`, unchanged champions skipped."""
    with open_run(recording) as reader:
        last = reader.generations()
        chosen = default_generations(last) if generations is None else sorted(set(generations))
        out_of_range = [k for k in chosen if not 1 <= k <= last]
        if out_of_range:
            raise SnapshotError(f"the run has generations 1 to {last}, not {out_of_range}")
        snapshots = select(chosen, [reader.best_up_to(k) for k in chosen])
    if not snapshots:
        raise SnapshotError("no candidate of the run has succeeded yet, so there is no champion to show")
    return snapshots
