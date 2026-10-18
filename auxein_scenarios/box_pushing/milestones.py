"""Milestones (design doc §7): the generations of a run worth showing, and their champions.

**Default milestones** are log-spaced: 1, 2, 5, 10, 20, 50, 100, 200, 500, … (1, 2 and 5 times the powers of ten) up to the
run's last generation, **plus the last generation**. A milestone whose champion is **the same candidate** as the previous
milestone's is skipped: it would show the same brain twice. Any list of generations can replace the default
(`--generations 1,50,400`); it is sorted, and the same skipping applies.

The champion of milestone `k` is `RunReader.best_up_to(k)`, one query per milestone: the run's best so far at generation `k`,
which is what `Progress.best` said then. Auxein's `champions(every=n)` is arithmetic (every `n`th generation), so the log
spacing is computed here.

**Every M3 command uses this module**: the held-out report evaluates these champions, the milestone clips and the montage show
them, and the generalisation chart plots them, so all of them talk about the same generations.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from auxein.recording.reader import RecordedEvaluation, open_run

__all__ = ["Milestone", "MilestoneError", "default_generations", "find_milestones", "parse_generations", "select"]


class MilestoneError(ValueError):
    """A list of generations that the run cannot answer: empty, out of range, or a run with no champion yet."""


@dataclass(frozen=True)
class Milestone:
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
    """1, 2, 5, 10, 20, 50, … up to `last`, and `last` itself: log-spaced, so the fast early changes get more milestones."""
    if last < 1:
        raise MilestoneError(f"the run has no generation yet (last = {last})")
    out: list[int] = []
    decade = 1
    while decade <= last:
        out += [m * decade for m in (1, 2, 5) if m * decade <= last]
        decade *= 10
    return sorted({*out, last})


def parse_generations(text: str) -> list[int]:
    """`"1,50,400"` as `[1, 50, 400]` (the `--generations` option)."""
    try:
        values = [int(part) for part in text.replace(" ", "").split(",") if part]
    except ValueError as error:
        raise MilestoneError(f"--generations takes comma-separated generation numbers, e.g. 1,50,400; got {text!r}") from error
    if not values:
        raise MilestoneError("--generations needs at least one generation")
    return values


def select(generations: Sequence[int], champions: Sequence[RecordedEvaluation | None]) -> list[Milestone]:
    """The milestones from generations (sorted, distinct) and their champions: a generation with no champion yet (nothing
    succeeded) is left out, and one whose champion is the previous milestone's is skipped."""
    out: list[Milestone] = []
    for generation, champion in zip(generations, champions, strict=True):
        if champion is None:
            continue
        if out and int(out[-1].champion.candidate_id) == int(champion.candidate_id):
            continue
        out.append(Milestone(generation, champion))
    return out


def find_milestones(recording: str | Path, generations: Sequence[int] | None = None) -> list[Milestone]:
    """The milestones of the recorded run in `recording` (Auxein's run directory): `generations` (default: the log-spaced
    list and the last), sorted, each with `best_up_to(k)`, unchanged champions skipped."""
    with open_run(recording) as reader:
        last = reader.generations()
        chosen = default_generations(last) if generations is None else sorted(set(generations))
        out_of_range = [k for k in chosen if not 1 <= k <= last]
        if out_of_range:
            raise MilestoneError(f"the run has generations 1 to {last}, not {out_of_range}")
        milestones = select(chosen, [reader.best_up_to(k) for k in chosen])
    if not milestones:
        raise MilestoneError("no candidate of the run has succeeded yet, so there is no champion to show")
    return milestones
