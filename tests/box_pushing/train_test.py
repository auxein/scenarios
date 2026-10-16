"""Training end to end on numpy (CI-sized, the `smoke` preset): train, interrupt and resume, extend, refuse a changed config,
and watch a run's best, also while another process is training it."""

import csv
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import auxein
import pytest
from auxein.driver.errors import ConfigurationMismatchError
from auxein.driver.progress import Progress
from auxein.recording.reader import open_run

from auxein_scenarios.box_pushing import train
from auxein_scenarios.box_pushing.config import ConfigError, TrainingConfig, dump_config, load_config, parse_config
from auxein_scenarios.box_pushing.train import CONFIG_FILE, PROGRESS_FILE, RECORDING, Showcase

ROOT = Path(__file__).resolve().parents[2]
SMOKE = load_config(ROOT / "configs" / "smoke.toml")

pytestmark = pytest.mark.filterwarnings("ignore::auxein.driver.errors.ResumeWarning")


def quiet(_: str) -> None:
    pass


def run(config: TrainingConfig, run_dir: Path, **kwargs: Any) -> auxein.RunResult[Any]:
    result, _ = train.train(config, run_dir, out=quiet, progress=False, estimate=False, **kwargs)
    return result


def event_log(run_dir: Path) -> dict[str, list[tuple[object, ...]]]:
    """The deterministic part of a recording: events (but the `resume` ones), candidates and genomes, lineage, evaluations and
    episodes, without clock times."""
    db = sqlite3.connect(run_dir / RECORDING / "events.sqlite")
    try:
        return {
            "events": [r for r in db.execute("SELECT kind, step, payload FROM events ORDER BY seq") if r[0] != "resume"],
            "candidates": list(db.execute("SELECT id, step, origin, genome, genome_hash FROM candidates ORDER BY id")),
            "lineage": list(db.execute("SELECT child_id, parent_id FROM lineage ORDER BY rowid")),
            "evaluations": list(
                db.execute("SELECT candidate_id, status, objectives, descriptors, error FROM evaluations ORDER BY candidate_id")
            ),
            "episodes": list(
                db.execute("SELECT candidate_id, scenario_index, status, measurements FROM episodes ORDER BY candidate_id, scenario_index")
            ),
        }
    finally:
        db.close()


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, auxein.RunResult[Any]]:
    """An uninterrupted smoke run (3 generations)."""
    run_dir = tmp_path_factory.mktemp("reference") / "smoke"
    return run_dir, run(SMOKE, run_dir)


def test_smoke_trains_records_and_writes_its_files(reference: tuple[Path, auxein.RunResult[Any]]) -> None:
    run_dir, result = reference
    assert result.evaluations_used == SMOKE.evaluations == 30 and result.best is not None
    assert load_config(run_dir / CONFIG_FILE) == SMOKE
    with open(run_dir / PROGRESS_FILE) as file:
        rows = list(csv.DictReader(file))
    assert [int(r["generation"]) for r in rows] == [1, 2, 3]
    for column in (
        "score_best",
        "score_mean",
        "success_rate_mean",
        "progress_mean",
        "approach_mean",
        "max_pushers_mean",
        "champion_success_rate",
    ):
        assert all(r[column] != "" for r in rows), column
    with open_run(run_dir / RECORDING) as reader:
        assert reader.generations() == 3
        best = reader.best_up_to()
        assert best is not None and set(best.descriptors) == {"success_rate", "progress", "approach", "max_pushers"}
        assert len(reader.episodes(int(best.candidate_id))) == 4
        assert "episode_score(progress=1.0" in str(reader.metadata["evaluator"])


class InterruptAt:
    """A callback that interrupts the run, as Ctrl-C would, after generation `generation` of a live (not replaying) session."""

    def __init__(self, generation: int) -> None:
        self.generation = generation

    def __call__(self, progress: Progress[Any]) -> None:
        if not progress.replaying and not progress.final and progress.generation == self.generation:
            raise KeyboardInterrupt


def test_interrupt_and_resume_gives_the_uninterrupted_run(reference: tuple[Path, auxein.RunResult[Any]], tmp_path: Path) -> None:
    ref_dir, ref_result = reference
    run_dir = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run(SMOKE, run_dir, callbacks=[InterruptAt(2)])
    with open_run(run_dir / RECORDING) as reader:
        assert reader.generations() == 2
    resumed, _ = train.resume(run_dir, out=quiet, progress=False)
    assert event_log(run_dir) == event_log(ref_dir)
    assert resumed.best is not None and ref_result.best is not None
    assert resumed.best.candidate.id == ref_result.best.candidate.id and resumed.best.objectives == ref_result.best.objectives
    assert (run_dir / PROGRESS_FILE).read_text().count("\n") == 4, "the progress file covers the whole run after a resume"


def test_extending_the_budget_on_resume(reference: tuple[Path, auxein.RunResult[Any]], tmp_path: Path) -> None:
    ref_dir, ref_result = reference
    run_dir = tmp_path / "short"
    run(SMOKE.with_generations(2), run_dir)
    extended, _ = train.resume(run_dir, generations=3, out=quiet, progress=False)
    assert event_log(run_dir) == event_log(ref_dir)
    assert extended.evaluations_used == ref_result.evaluations_used
    assert load_config(run_dir / CONFIG_FILE).budget.generations == 3, "the run's config holds its new budget"
    with pytest.raises(ConfigError, match="already has 3"):
        train.resume(run_dir, generations=2, out=quiet, progress=False)


def test_any_other_change_is_refused_on_resume(reference: tuple[Path, auxein.RunResult[Any]], tmp_path: Path) -> None:
    ref_dir, _ = reference
    changed_weight = parse_config(dump_config(SMOKE).replace("approach = 0.1", "approach = 0.2"))
    with pytest.raises(ConfigError, match=r"fitness.approach: recorded 0.1, given 0.2"):
        train.resume(ref_dir, config=changed_weight, out=quiet, progress=False)
    changed_world = parse_config(dump_config(SMOKE).replace("agents = 10", "agents = 12"))
    with pytest.raises(ConfigError, match="team.agents"):
        train.resume(ref_dir, config=changed_world, out=quiet, progress=False)
    # the same config with a larger budget is a plain extension
    assert train.resume(ref_dir, config=SMOKE, out=quiet, progress=False)[0].evaluations_used == 30


def test_an_edited_weight_in_the_runs_config_is_refused_by_auxein(tmp_path: Path) -> None:
    """Even when the run's own config.toml is edited, Auxein's comparison of the evaluator refuses the resume, because the
    score's label shows its weights (and the world's description its parameters)."""
    run_dir = tmp_path / "edited"
    run(SMOKE.with_generations(1), run_dir)
    config = run_dir / CONFIG_FILE
    config.write_text(config.read_text().replace("approach = 0.1", "approach = 0.3"))
    with pytest.raises(ConfigurationMismatchError, match=r"approach=0.3"):
        train.resume(run_dir, generations=2, out=quiet, progress=False)


def test_a_new_run_does_not_overwrite_one(reference: tuple[Path, auxein.RunResult[Any]]) -> None:
    ref_dir, _ = reference
    with pytest.raises(train.TrainingError, match="already holds a run"):
        run(SMOKE, ref_dir)


def test_watch_renders_what_was_scored(reference: tuple[Path, auxein.RunResult[Any]], tmp_path: Path) -> None:
    """The clip's episodes are the recorded ones: the best brain on two training scenarios gives the measurements the run
    recorded for it (numpy float64: the same numbers), and on a held-out one it runs too."""
    ref_dir, _ = reference
    lines: list[str] = []
    clip = train.watch(
        ref_dir, showcase=[Showcase("train", 0), Showcase("train", 3), Showcase("held", 1)], overlay=True, path=tmp_path / "w.mp4",
        fps=4, speedup=20, out=lines.append,
    )  # fmt: skip
    assert clip.read_bytes()[4:8] == b"ftyp"
    with open_run(ref_dir / RECORDING) as reader:
        best = reader.best_up_to(reader.generations())
        assert best is not None
        recorded = reader.episodes(int(best.candidate_id))
    for line, episode in zip(lines[1:3], (recorded[0], recorded[3]), strict=True):
        delivered = episode.measurements["delivered"] > 0.5
        assert episode.scenario_id in line and (("not delivered" not in line) == delivered)
        assert f"most touching the box {int(episode.measurements['max_pushers'])}" in line


def test_watch_while_another_process_trains(tmp_path: Path) -> None:
    """A training run in another process (the CLI), watched while it goes on: Auxein's reader opens the recording read-only."""
    config = tmp_path / "long.toml"
    config.write_text((ROOT / "configs" / "smoke.toml").read_text().replace("generations = 3", "generations = 200"))
    run_dir = tmp_path / "live"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    # Ctrl-C's default action, in case this process was started with SIGINT ignored (a background job), which a child inherits
    process = subprocess.Popen(
        [sys.executable, "-m", "auxein_scenarios.box_pushing.cli", "train", str(config), "--run-dir", str(run_dir)],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 120
        while not (run_dir / PROGRESS_FILE).exists():
            assert process.poll() is None, process.stderr.read().decode() if process.stderr else "the training process ended"
            assert time.monotonic() < deadline, "no generation within two minutes"
            time.sleep(0.2)
        clip = train.watch(run_dir, showcase=[Showcase("train", 0)], path=tmp_path / "live.mp4", fps=4, speedup=40, out=quiet)
        assert clip.exists()
        assert process.poll() is None, "the run should still be training while it is watched"
    finally:
        process.send_signal(signal.SIGINT)  # once
        code = process.wait(timeout=30)
    assert code == 130, "one Ctrl-C ends the command, after the generation in progress, with the interrupt's exit code"
    with open_run(run_dir / RECORDING) as reader:
        told = reader.generations()
        assert told >= 1 and reader.sessions[-1].get("status") == "interrupted"
    # and it resumes, here shortened: fewer than its 200 generations, one more than it has
    resumed, _ = train.resume(run_dir, generations=told + 1, out=quiet, progress=False)
    assert resumed.evaluations_used == 10 * (told + 1)
