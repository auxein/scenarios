"""The held-out report (design doc §7.2) on a smoke run: numpy, CI-sized, with the smoke preset's 4 held-out scenarios."""

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest
from auxein.recording.reader import open_run

from auxein_scenarios.box_pushing import report, train
from auxein_scenarios.box_pushing.config import load_config
from auxein_scenarios.box_pushing.milestones import find_milestones

ROOT = Path(__file__).resolve().parents[2]
SMOKE = load_config(ROOT / "configs" / "smoke.toml")


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("report") / "smoke"
    train.train(SMOKE, path, out=lambda _: None, progress=False, estimate=False)
    return path


@pytest.fixture(scope="module")
def built(run: Path) -> report.Report:
    return report.build_report(run, out=lambda _: None)


def fingerprint(directory: Path) -> dict[str, str]:
    """Every file of a directory and its hash."""
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.rglob("*")) if p.is_file()}


def test_every_champion_and_both_baselines_are_evaluated(run: Path, built: report.Report) -> None:
    milestones = find_milestones(run / train.RECORDING)
    assert [c.candidate_id for c in built.champions] == [m.candidate_id for m in milestones]
    assert [c.candidate_id for c in built.auxein.candidates] == [m.candidate_id for m in milestones], "through evaluate_held_out"
    held, training = SMOKE.scenarios.held_out, SMOKE.scenarios.easy + SMOKE.scenarios.full
    for champion in built.champions:
        assert len(champion.held_out) == held and len(champion.training) == training
        assert all(e.scenario_id.startswith("held-") for e in champion.held_out)
    kinds = {s.kind: s for s in built.subjects if s.kind != "champion"}
    assert set(kinds) == {"random brains", "senses only"}
    assert len(kinds["random brains"].held_out) == report.BASELINE_BRAINS * held
    assert len(kinds["senses only"].held_out) == held and len(kinds["senses only"].training) == training


def test_training_numbers_are_the_recordings(run: Path, built: report.Report) -> None:
    with open_run(run / train.RECORDING) as reader:
        for champion in built.champions:
            assert champion.candidate_id is not None
            recorded = reader.best_up_to(champion.generation)
            assert recorded is not None
            t = champion.train()
            assert t.score == pytest.approx(recorded.objectives["score"], abs=1e-12)
            assert t.success == pytest.approx(recorded.descriptors["success_rate"], abs=1e-12)
            assert t.progress == pytest.approx(recorded.descriptors["progress"], abs=1e-12)
            assert t.max_pushers == pytest.approx(recorded.descriptors["max_pushers"], abs=1e-12)
            easy, full = champion.train("easy"), champion.train("full")
            assert (easy.episodes, full.episodes) == (SMOKE.scenarios.easy, SMOKE.scenarios.full)


def test_held_out_numbers_are_auxeins_and_the_gap_is_as_defined(built: report.Report) -> None:
    for champion, judged in zip(built.champions, built.auxein.candidates, strict=True):
        h = champion.held()
        assert h.score == pytest.approx(judged.objectives["score"], abs=1e-12)
        assert h.success == pytest.approx(judged.descriptors["success_rate"], abs=1e-12)
        assert h.progress == pytest.approx(judged.descriptors["progress"], abs=1e-12)
        assert champion.gap == champion.train("full").success - h.success


def test_the_recording_is_unchanged_and_the_report_is_deterministic(run: Path, built: report.Report) -> None:
    before = fingerprint(run / train.RECORDING)
    first = report.write_report(report.build_report(run, out=lambda _: None))
    summary, episodes = (first / report.SUMMARY_FILE).read_text(), (first / report.EPISODES_FILE).read_text()
    second = report.write_report(report.build_report(run, out=lambda _: None))
    assert (second / report.SUMMARY_FILE).read_text() == summary
    assert (second / report.EPISODES_FILE).read_text() == episodes
    assert fingerprint(run / train.RECORDING) == before, "nothing is written into the recording"
    assert not (run / train.RECORDING / "held_out.json").exists()


def test_the_files_exist_and_parse(run: Path, built: report.Report) -> None:
    target = report.write_report(built)
    with open(target / report.SUMMARY_FILE) as file:
        rows = list(csv.DictReader(file))
    assert [r["kind"] for r in rows] == ["champion"] * len(built.champions) + ["random brains", "senses only"]
    final = rows[len(built.champions) - 1]
    assert int(final["candidate_id"]) == built.final.candidate_id
    assert float(final["gap"]) == pytest.approx(built.final.gap, abs=1e-6)
    with open(target / report.EPISODES_FILE) as file:
        assert len(list(csv.DictReader(file))) == sum(len(s.held_out) for s in built.subjects)
    auxein_json = json.loads((target / report.JSON_FILE).read_text())
    assert [c["candidate_id"] for c in auxein_json["candidates"]] == [c.candidate_id for c in built.champions]
    text = (target / report.MARKDOWN_FILE).read_text()
    assert text.startswith("# Held-out report: smoke")
    assert "| gap |" in text and "**Recommendation:" in text and "senses-only controller" in text
    assert report.read_episodes(run) is not None


@pytest.mark.parametrize(
    ("held", "train_full", "exploit", "case"),
    [
        (0.55, 0.9, None, "generalises"),  # at least half: generalises, even with a large gap
        (0.5, 0.5, None, "generalises"),
        (0.2, 0.6, None, "learned the maps"),  # a gap of 0.4
        (0.42, 0.67, None, "limited"),  # a gap of 0.25: not more than it
        (0.1, 0.2, None, "limited"),
        (0.8, 0.8, "ramming", "exploit"),  # an exploit comes first, whatever the numbers
    ],
)
def test_the_recommendation(held: float, train_full: float, exploit: str | None, case: str) -> None:
    r = report.recommend(held, train_full, exploit)
    assert r.case == case
    assert f"{held:.0%}" in r.reason and f"{train_full:.0%}" in r.reason, "the numbers behind it are stated"
    expected = {"generalises": "M4", "learned the maps": "more full training scenarios", "limited": "M2b", "exploit": "M2b"}
    assert expected[case] in r.next_milestone


def test_summaries() -> None:
    def episode(delivered: bool, steps: int, progress: float, ok: bool = True) -> report.Episode:
        return report.Episode("held-0000", "full", 8.0, 3, ok, delivered, steps, progress, 4, 0.0)

    r = report.summarise(
        [episode(True, 300, 0.8), episode(True, 500, 0.9), episode(False, 900, 0.2), episode(False, 0, math.nan, ok=False)]
    )
    assert (r.episodes, r.failed, r.delivered) == (4, 1, 2)
    assert r.success == pytest.approx(2 / 3) and r.progress == pytest.approx(1.9 / 3) and r.delivery_steps == 400
    assert math.isnan(report.summarise([episode(False, 900, 0.1)]).delivery_steps)
