"""Milestones (design doc §7): the log spacing, the last generation, unchanged champions skipped, and the override."""

from pathlib import Path

import pytest
from auxein.core import CandidateId, Status
from auxein.recording.reader import RecordedEvaluation, open_run

from auxein_scenarios.box_pushing import train
from auxein_scenarios.box_pushing.config import load_config
from auxein_scenarios.box_pushing.milestones import (
    MilestoneError,
    default_generations,
    find_milestones,
    parse_generations,
    select,
)

ROOT = Path(__file__).resolve().parents[2]


def champion(candidate_id: int) -> RecordedEvaluation:
    """A hand-built recorded evaluation: only its id matters to the skipping."""
    return RecordedEvaluation(CandidateId(candidate_id), 0, "child", (), None, Status.OK, {"score": 0.0}, {}, {}, {}, 0.0, None)


def test_log_spacing_with_the_last_generation() -> None:
    assert default_generations(1) == [1]
    assert default_generations(3) == [1, 2, 3]
    assert default_generations(400) == [1, 2, 5, 10, 20, 50, 100, 200, 400]
    assert default_generations(500) == [1, 2, 5, 10, 20, 50, 100, 200, 500], "the last is not repeated when it is on the grid"
    assert default_generations(1234) == [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1234]
    with pytest.raises(MilestoneError):
        default_generations(0)


def test_unchanged_champions_are_skipped() -> None:
    champions = [champion(3), champion(3), champion(7), None, champion(7), champion(9)]
    chosen = select([1, 2, 5, 10, 20, 50], champions)
    assert [(m.generation, m.candidate_id) for m in chosen] == [(1, 3), (5, 7), (50, 9)]
    # a generation with no champion yet (nothing succeeded) is left out, not an error
    assert [m.generation for m in select([1, 2], [None, champion(4)])] == [2]


def test_the_override() -> None:
    assert parse_generations("1,50,400") == [1, 50, 400]
    assert parse_generations(" 3, 1 ") == [3, 1]
    for bad in ("", "1,x", "one"):
        with pytest.raises(MilestoneError):
            parse_generations(bad)


def test_on_a_recorded_run(tmp_path: Path) -> None:
    run = tmp_path / "smoke"
    train.train(load_config(ROOT / "configs" / "smoke.toml"), run, out=lambda _: None, progress=False, estimate=False)
    recording = run / train.RECORDING
    milestones = find_milestones(recording)
    with open_run(recording) as reader:
        assert milestones[-1].generation == reader.generations() == 3
        for m in milestones:
            best = reader.best_up_to(m.generation)
            assert best is not None and int(best.candidate_id) == m.candidate_id
    ids = [m.candidate_id for m in milestones]
    assert len(ids) == len(set(ids)), "no champion twice"

    def listed(generations: list[int]) -> list[tuple[int, int]]:
        return [(m.generation, m.candidate_id) for m in find_milestones(recording, generations)]

    assert listed([3, 1, 3]) == listed([1, 3]), "the override is sorted and each generation counted once"
    with pytest.raises(MilestoneError, match="generations 1 to 3"):
        find_milestones(recording, [1, 4])
