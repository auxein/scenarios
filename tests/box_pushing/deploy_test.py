"""Deploying brains (numpy, the smoke run's brains): one brain and several, fresh scenarios by default, another team size, the
baselines, the files, the video's panels and captions, and the same episodes as Auxein's held-out evaluation."""

import csv
import tomllib
from pathlib import Path
from typing import Any

import auxein
import numpy as np
import pytest
from auxein.backend import Backend
from auxein.driver.held_out import evaluate_held_out

from auxein_scenarios.box_pushing import deploy, evaluation, fitness, render, scenarios, train
from auxein_scenarios.box_pushing.brain import load_brain
from auxein_scenarios.box_pushing.config import load_config
from auxein_scenarios.common import theme
from auxein_scenarios.common.rendering import drawn_text

ROOT = Path(__file__).resolve().parents[2]
SMOKE = load_config(ROOT / "configs" / "smoke.toml")
NUMPY = Backend("numpy", "cpu", "float64")


def quiet(_: str) -> None:
    pass


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("deploy") / "smoke"
    train.train(SMOKE, path, out=quiet, progress=False, estimate=False)
    return path


def rows(path: Path) -> list[dict[str, str]]:
    with open(path) as file:
        return list(csv.DictReader(file))


def test_one_brain_with_its_files_and_video(run_dir: Path, tmp_path: Path) -> None:
    final = run_dir / train.BRAINS / train.FINAL
    done = deploy.deploy([final], count=6, backend=NUMPY, video_scenarios=2, out_dir=tmp_path / "one", out=quiet)
    files = sorted(p.name for p in done.directory.iterdir())
    assert files == sorted([deploy.OPTIONS_FILE, deploy.RESULTS_FILE, deploy.MARKDOWN_FILE, deploy.VIDEO_FILE])
    assert (done.directory / deploy.VIDEO_FILE).read_bytes()[4:8] == b"ftyp"
    # deploy.toml: the effective options and the brain's provenance
    options = tomllib.loads((done.directory / deploy.OPTIONS_FILE).read_text())
    assert options["options"]["scenarios"] == 6 and options["options"]["difficulty"] == "full"
    assert (options["options"]["backend"], options["options"]["precision"]) == ("numpy", "float64")
    (entry,) = options["brains"]
    assert entry["path"] == str(final) and entry["final"] is True and entry["agents"] == SMOKE.team.agents
    assert entry["candidate_id"] == load_brain(final).provenance["candidate_id"] and entry["scenarios"]["held_out_seed"] == 2
    # results.csv: the brain, 20 random brains and the controller, on each scenario
    results = rows(done.directory / deploy.RESULTS_FILE)
    assert len(results) == (1 + evaluation.BASELINE_BRAINS + 1) * 6
    assert {r["kind"] for r in results} == {"brain", "baseline"}
    assert {r["subject"] for r in results if r["kind"] == "baseline"} >= {"random brains 1", "random brains 20", "senses-only controller"}
    for r in results:
        assert r["delivered"] in ("0", "1") and float(r["distance"]) >= 6.0 and int(r["max_pushers"]) >= 0
    # summary.md: a row per brain and baseline
    summary = (done.directory / deploy.MARKDOWN_FILE).read_text()
    assert "smoke · final" in summary and "*random brains*" in summary and "*senses-only controller*" in summary
    assert "| 6–8 units | 8–10 | 10–12 |" in summary


def test_the_default_seed_is_fresh(run_dir: Path, tmp_path: Path) -> None:
    final = run_dir / train.BRAINS / train.FINAL
    done = deploy.deploy([final], count=4, backend=NUMPY, video=False, out_dir=tmp_path / "fresh", out=quiet)
    seed = int(done.options["seed"])  # type: ignore[arg-type]
    assert seed not in (SMOKE.scenarios.seed, SMOKE.scenarios.held_out_seed)
    scenarios.check_disjoint(train.training_set(SMOKE), train.held_out_set(SMOKE), done.brains[0].scenarios)
    assert not (done.directory / deploy.VIDEO_FILE).exists(), "--no-video"
    assert seed != deploy.deploy([final], count=4, backend=NUMPY, video=False, out_dir=tmp_path / "again", out=quiet).options["seed"]
    # an explicit seed is kept, and gives the same scenarios
    a = deploy.deploy([final], count=4, seed=99, backend=NUMPY, video=False, out_dir=tmp_path / "a", out=quiet)
    b = deploy.deploy([final], count=4, seed=99, backend=NUMPY, video=False, out_dir=tmp_path / "b", out=quiet)
    assert a.brains[0].scenarios.fingerprint == b.brains[0].scenarios.fingerprint and a.brains[0].episodes == b.brains[0].episodes


def test_another_team_size(run_dir: Path, tmp_path: Path) -> None:
    final = run_dir / train.BRAINS / train.FINAL
    done = deploy.deploy([final], count=3, agents=7, backend=NUMPY, video=False, out_dir=tmp_path / "seven", out=quiet)
    assert {len(s.params["agents"]) for s in done.brains[0].scenarios} == {7}  # type: ignore[arg-type]
    assert tomllib.loads((done.directory / deploy.OPTIONS_FILE).read_text())["options"]["agents"] == 7
    assert {len(s.params["agents"]) for d in done.baselines for s in d.scenarios} == {7}  # type: ignore[arg-type]


def test_three_brains_and_the_videos_captions(run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Three brains on the same scenarios: a panel per brain and scenario, each captioned with the brain, the scenario and its
    outcome (the drawn text of a few frames; the encoding is stubbed, as it is tested above)."""
    seen: list[list[str]] = []

    def capture(fig: Any, update: Any, frames: Any, path: Any, fps: float = 30.0) -> Path:
        for frame in (frames[0], frames[len(frames) // 2], frames[-1]):
            update(frame)
            seen.append(drawn_text(fig))
        Path(path).write_bytes(b"")
        return Path(path)

    monkeypatch.setattr(render, "write_mp4", capture)
    files = [run_dir / train.BRAINS / "gen-0001.safetensors", run_dir / train.BRAINS / train.FINAL, run_dir / train.BRAINS / train.FINAL]
    done = deploy.deploy(files, count=5, backend=NUMPY, video_scenarios=2, out_dir=tmp_path / "three", out=quiet)
    assert [d.name for d in done.brains] == ["smoke · generation 1", "smoke · final", "smoke · final"]
    assert len({d.scenarios.fingerprint for d in done.brains}) == 1, "the same scenarios for every brain"
    assert len(rows(done.directory / deploy.RESULTS_FILE)) == (3 + evaluation.BASELINE_BRAINS + 1) * 5
    first, last = seen[0], seen[-1]
    assert "3 brains on 2 new scenarios" in first
    labels = [t for t in last if t in (theme.tracked("smoke · generation 1"), theme.tracked("smoke · final"))]
    assert labels == [theme.tracked(n) for n in ("smoke · generation 1",) * 2 + ("smoke · final",) * 4], "a row per brain"
    notes = [t for t in last if t in (theme.tracked("scenario-0000 · full"), theme.tracked("scenario-0001 · full"))]
    assert notes == [theme.tracked(f"scenario-000{j} · full") for _ in range(3) for j in range(2)], "the same two scenarios"
    outcomes = [t for t in last if t.startswith(("delivered at step", "not delivered"))]
    assert len(outcomes) == 3 * 2, "every panel says how its episode ended"


def test_deploying_on_the_held_out_seed_is_the_held_out_evaluation(run_dir: Path, tmp_path: Path) -> None:
    """On numpy float64, a brain deployed on its run's held-out seed has, scenario by scenario, the measurements Auxein's
    `evaluate_held_out` gives the same candidate."""
    final = load_brain(run_dir / train.BRAINS / train.FINAL)
    count = SMOKE.scenarios.held_out
    done = deploy.deploy([final.path], count=count, seed=SMOKE.scenarios.held_out_seed, backend=NUMPY, video=False,
                         out_dir=tmp_path / "held", out=quiet)  # fmt: skip
    held = train.held_out_set(SMOKE)
    training = train.build(SMOKE, NUMPY)
    evaluator = auxein.EpisodeEvaluator(training.policy, training.world, held, fitness.aggregator(SMOKE.fitness.weights, SMOKE.max_steps))
    judged = evaluate_held_out(run_dir / train.RECORDING, evaluator, held, candidates=[final.provenance["candidate_id"]], write=False)
    measured = [dict(s.measurements) for s in judged.candidates[0].scenarios]
    expected = evaluation.episodes(list(held), measured, evaluation.scorer(SMOKE.fitness, SMOKE.max_steps))
    got = done.brains[0].episodes
    assert [(e.delivered, e.steps, e.progress, e.max_pushers, e.score, e.distance) for e in got] == [
        (e.delivered, e.steps, e.progress, e.max_pushers, e.score, e.distance) for e in expected
    ]
    assert np.isfinite([e.progress for e in got]).all()


def test_bad_options_are_refused(run_dir: Path, tmp_path: Path) -> None:
    final = run_dir / train.BRAINS / train.FINAL
    with pytest.raises(deploy.DeployError, match="at least one brain"):
        deploy.deploy([], backend=NUMPY, out=quiet)
    with pytest.raises(deploy.DeployError, match="at least 1"):
        deploy.deploy([final], count=0, backend=NUMPY, out=quiet)
    with pytest.raises(deploy.DeployError, match="difficulty"):
        deploy.deploy([final], difficulty="hard", backend=NUMPY, out=quiet)
    (tmp_path / "junk.safetensors").write_bytes(b"junk")
    with pytest.raises(deploy.DeployError, match="not a brain file"):
        deploy.deploy([tmp_path / "junk.safetensors"], backend=NUMPY, out=quiet)
