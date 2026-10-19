"""Training end to end on numpy (CI-sized, the `smoke` preset): what a run writes, interrupting and resuming, extending,
refusing a changed run, Ctrl-C through the command line, and the same numbers as the code before brains and deploy."""

import csv
import hashlib
import json
import os
import platform
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import auxein
import numpy as np
import pytest
from auxein.driver.errors import ConfigurationMismatchError
from auxein.driver.progress import Progress
from auxein.recording.reader import open_run

from auxein_scenarios.box_pushing import fitness, train
from auxein_scenarios.box_pushing.brain import load_brain
from auxein_scenarios.box_pushing.charts import NAMES
from auxein_scenarios.box_pushing.config import ConfigError, TrainingConfig, load_config, parse_config
from auxein_scenarios.box_pushing.snapshots import find_snapshots, is_snapshot
from auxein_scenarios.box_pushing.train import BRAINS, CONFIG_FILE, FINAL, HELD_OUT_FILE, PROGRESS_FILE, RECORDING

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent / "data"
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


def brains(run_dir: Path) -> list[str]:
    return sorted(p.name for p in (run_dir / BRAINS).glob("*.safetensors"))


def expected_brains(run_dir: Path) -> list[str]:
    """The snapshot brains a run must have: the log-spaced generations whose champion changed, and the final one."""
    with open_run(run_dir / RECORDING) as reader:
        told = reader.generations()
    on_schedule = [g for g in range(1, told + 1) if is_snapshot(g)]
    return sorted([f"gen-{m.generation:04d}.safetensors" for m in find_snapshots(run_dir / RECORDING, on_schedule)] + [FINAL])


def rows(path: Path) -> list[dict[str, str]]:
    with open(path) as file:
        return list(csv.DictReader(file))


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, auxein.RunResult[Any]]:
    """An uninterrupted smoke run (3 generations)."""
    run_dir = tmp_path_factory.mktemp("reference") / "smoke"
    return run_dir, run(SMOKE, run_dir)


def test_a_run_writes_its_files(reference: tuple[Path, auxein.RunResult[Any]]) -> None:
    run_dir, result = reference
    assert result.evaluations_used == SMOKE.evaluations == 30 and result.best is not None
    assert load_config(run_dir / CONFIG_FILE) == SMOKE
    assert [int(r["generation"]) for r in rows(run_dir / PROGRESS_FILE)] == [1, 2, 3]
    for column in ("score_best", "score_mean", "success_rate_mean", "max_pushers_mean", "champion_success_rate"):
        assert all(r[column] != "" for r in rows(run_dir / PROGRESS_FILE)), column
    # the brains: the snapshots' and the final one, which is the run's best
    assert brains(run_dir) == expected_brains(run_dir) and "gen-0001.safetensors" in brains(run_dir)
    final = load_brain(run_dir / BRAINS / FINAL)
    assert final.provenance["candidate_id"] == int(result.best.candidate.id) and final.provenance["generation"] == 3
    np.testing.assert_array_equal(final.genome, np.asarray(result.best.candidate.genome, dtype=np.float64))
    # held_out.csv: one row per snapshot, the last generation's included
    held = rows(run_dir / HELD_OUT_FILE)
    assert [int(r["generation"]) for r in held] == [m.generation for m in find_snapshots(run_dir / RECORDING)]
    assert all(int(r["held_episodes"]) == SMOKE.scenarios.held_out for r in held)
    assert set(held[0]) >= {"train_full_success", "held_success", "held_delivered", "gap"}
    # the four charts, PNG and SVG
    assert sorted(p.name for p in (run_dir / "charts").iterdir()) == sorted(f"{n}.{x}" for n in NAMES for x in ("png", "svg"))


def test_held_out_csv_is_auxeins_evaluation(reference: tuple[Path, auxein.RunResult[Any]]) -> None:
    """The file's held-out numbers are Auxein's `evaluate_held_out` of the snapshot champions on the run's held-out set."""
    from auxein.driver.held_out import evaluate_held_out

    run_dir, _ = reference
    training = train.build(SMOKE, train.resolve_backend(SMOKE)[0])
    held = train.held_out_set(SMOKE)
    evaluator = auxein.EpisodeEvaluator(training.policy, training.world, held, fitness.aggregator(SMOKE.fitness.weights, SMOKE.max_steps))
    last = find_snapshots(run_dir / RECORDING)[-1]
    judged = evaluate_held_out(run_dir / RECORDING, evaluator, held, candidates=[last.candidate_id], write=False)
    delivered = sum(s.measurements["delivered"] > 0.5 for s in judged.candidates[0].scenarios)
    assert int(rows(run_dir / HELD_OUT_FILE)[-1]["held_delivered"]) == delivered


class InterruptAt:
    """A callback that interrupts the run, as Ctrl-C would, after generation `generation` of a live (not replaying) session."""

    def __init__(self, generation: int) -> None:
        self.generation = generation

    def __call__(self, progress: Progress[Any]) -> None:
        if not progress.replaying and not progress.final and progress.generation == self.generation:
            raise KeyboardInterrupt


def test_an_interrupted_run_has_its_final_brain_and_resumes(reference: tuple[Path, auxein.RunResult[Any]], tmp_path: Path) -> None:
    ref_dir, ref_result = reference
    run_dir = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run(SMOKE, run_dir, callbacks=[InterruptAt(2)])
    with open_run(run_dir / RECORDING) as reader:
        assert reader.generations() == 2
        best = reader.best_up_to(2)
    assert best is not None and FINAL in brains(run_dir), "an interrupted run has a usable brain"
    assert load_brain(run_dir / BRAINS / FINAL).provenance["candidate_id"] == int(best.candidate_id)
    assert not (run_dir / HELD_OUT_FILE).exists(), "the held-out judging is for the end of a run"
    resumed, _ = train.resume(run_dir, out=quiet, progress=False)
    assert event_log(run_dir) == event_log(ref_dir)
    assert resumed.best is not None and ref_result.best is not None
    assert resumed.best.candidate.id == ref_result.best.candidate.id and resumed.best.objectives == ref_result.best.objectives
    assert brains(run_dir) == brains(ref_dir)
    assert (run_dir / HELD_OUT_FILE).read_text() == (ref_dir / HELD_OUT_FILE).read_text()
    assert (run_dir / PROGRESS_FILE).read_text().count("\n") == 4, "the progress file covers the whole run after a resume"


def test_extending_a_run_refreshes_its_brains_charts_and_held_out_results(tmp_path: Path) -> None:
    run_dir = tmp_path / "short"
    run(SMOKE.with_generations(2), run_dir)
    before = {"held": (run_dir / HELD_OUT_FILE).read_text(), "final": load_brain(run_dir / BRAINS / FINAL).provenance}
    chart = (run_dir / "charts" / "fitness.png").stat().st_mtime_ns
    time.sleep(0.01)
    extended, _ = train.resume(run_dir, generations=6, out=quiet, progress=False)
    assert extended.evaluations_used == 60
    assert load_config(run_dir / CONFIG_FILE).evolution.generations == 6, "the run's config holds its new number of generations"
    assert brains(run_dir) == expected_brains(run_dir)
    final = load_brain(run_dir / BRAINS / FINAL).provenance
    assert final["generation"] == 6 and before["final"]["generation"] == 2
    assert [int(r["generation"]) for r in rows(run_dir / HELD_OUT_FILE)][-1] == 6 != int(before["held"].splitlines()[-1].split(",")[0])
    assert (run_dir / "charts" / "fitness.png").stat().st_mtime_ns > chart
    with pytest.raises(ConfigError, match="already has 6"):
        train.resume(run_dir, generations=5, out=quiet, progress=False)


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


def test_ctrl_c_through_the_command_line(tmp_path: Path) -> None:
    """A long run started with the `scenarios` command in another process, stopped by one Ctrl-C after its first generation:
    the command ends with the interrupt's exit code, the run has its final brain, and it resumes."""
    config = tmp_path / "long.toml"
    config.write_text((ROOT / "configs" / "smoke.toml").read_text().replace("generations = 3", "generations = 200"))
    run_dir = tmp_path / "live"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    # Ctrl-C's default action, in case this process was started with SIGINT ignored (a background job), which a child inherits
    process = subprocess.Popen(
        [sys.executable, "-m", "auxein_scenarios.common.cli", "box-pushing", "train", str(config), "--run-dir", str(run_dir)],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 120
        while not (run_dir / PROGRESS_FILE).exists():
            assert process.poll() is None, process.stderr.read().decode() if process.stderr else "the training process ended"
            assert time.monotonic() < deadline, "no generation within two minutes"
            time.sleep(0.2)
        assert process.poll() is None, "the run should still be training"
    finally:
        process.send_signal(signal.SIGINT)  # once
        code = process.wait(timeout=60)
    assert code == 130, "one Ctrl-C ends the command, after the generation in progress, with the interrupt's exit code"
    with open_run(run_dir / RECORDING) as reader:
        told = reader.generations()
        assert told >= 1 and reader.sessions[-1].get("status") == "interrupted"
    assert FINAL in brains(run_dir)
    resumed, _ = train.resume(run_dir, generations=told + 1, out=quiet, progress=False)
    assert resumed.evaluations_used == 10 * (told + 1)


# --- the same numbers as before brains and deploy ---


def _digest(recording: Path) -> str:
    """Every evaluation of the run (genome, objectives, descriptors, each episode's measurements), numbers to 9 significant
    digits: what the platforms agree on (their libm differs in the last bit of a float)."""

    def num(value: object) -> str:
        return f"{float(value):.9g}"  # type: ignore[arg-type]

    h = hashlib.sha256()
    with open_run(recording) as reader:
        for generation in range(1, reader.generations() + 1):
            for e in reader.evaluations_in(generation):
                genome = np.asarray(e.genome, dtype=np.float64).reshape(-1)
                row: list[object] = [
                    generation, int(e.candidate_id), e.status.value, {k: num(v) for k, v in sorted(e.objectives.items())},
                    {k: num(v) for k, v in sorted(e.descriptors.items())}, [num(x) for x in genome],
                ]  # fmt: skip
                for episode in reader.episodes(int(e.candidate_id)):
                    row.append({k: num(v) for k, v in sorted(dict(episode.measurements).items())})
                h.update(json.dumps(row).encode())
    return h.hexdigest()


def test_the_same_numbers_as_before(tmp_path: Path) -> None:
    """A smoke run pinned from the code before brains and deploy (`data/pre_r1_smoke.json`), trained again: the same
    evaluations and final best on every platform, and the same event log bit for bit where it was pinned."""
    pin = json.loads((DATA / "pre_r1_smoke.json").read_text())
    config = parse_config(pin["config"])
    run_dir = tmp_path / "pin"
    result = run(config, run_dir)
    with open_run(run_dir / RECORDING) as reader:
        assert reader.generations() == pin["generations"]
    assert result.best is not None and int(result.best.candidate.id) == pin["final_best_candidate"]
    assert result.best.objectives["score"] == pytest.approx(pin["final_best_score"], rel=1e-12)
    assert _digest(run_dir / RECORDING) == pin["evaluations_sha256_9_digits"]
    machine = f"{sys.platform}-{platform.machine().lower().replace('amd64', 'x86_64').replace('aarch64', 'arm64')}"
    pinned = pin["event_log_sha256"].get(machine)
    if pinned is not None:
        assert hashlib.sha256(repr(event_log(run_dir)).encode()).hexdigest() == pinned
