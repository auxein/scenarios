"""Training: a run built from its `TrainingConfig`, run and resumed through Auxein, ending with its brains, its held-out results
and its charts. No video: videos come from `deploy`.

**`train(config)`** builds everything from the config (the scenario sets, the world, the brain, the aggregator, the
`EpisodeEvaluator` and the `GeneticAlgorithm`) and runs Auxein with recording on, in deterministic mode, with generation
delivery and a checkpoint every `CHECKPOINT_EVERY` generations. The run directory (`runs/<name>-<timestamp>` by default) holds:

- `config.toml`: the effective config, every default filled in, so it defines the run on its own;
- `recording/`: **Auxein's run directory** (its event log, metadata and checkpoints; open it with `auxein.open_run`), a
  directory of its own because Auxein starts a run only in an empty or new directory;
- `progress.csv`: per generation, the scores and descriptors, rewritten from the recording after every generation;
- `brains/`: `gen-<k>.safetensors` for every snapshot (`snapshots.py`), written when the generation is reached, and
  `final.safetensors`, the run's best, written at the end, and also on interruption so that a stopped run has a usable brain;
- `held_out.csv`: the snapshot brains judged on the run's held-out set (Auxein's `evaluate_held_out`, which writes nothing
  into the recording), with their training results and the generalisation gap: what the generalisation chart plots;
- `charts/`: fitness, delivery, cooperation and generalisation (`charts.py`).

**`resume(run_dir)`** continues a run from its recording and its `config.toml`: after an interruption (Ctrl-C stops a run
between generations, resumably), or with another number of generations (`generations`, which must not be fewer than it already
has). Nothing else may change; Auxein itself refuses a resume whose evaluator differs, so even an edited `config.toml` cannot
silently change a run. A resumed run refreshes its brains, `held_out.csv` and charts at the end.

The run stops on its number of generations only: brains are judged on held-out scenarios after training, never during it, so
nothing outside the training set influences evolution.
"""

import csv
import importlib
import importlib.util
import io
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import auxein
import numpy as np
from auxein.backend import Backend
from auxein.callbacks import print_progress
from auxein.core import Status
from auxein.driver.held_out import evaluate_held_out
from auxein.driver.progress import Callback, Progress
from auxein.environments import MLPPolicy, ScenarioSet
from auxein.random import RandomStream
from auxein.recording.reader import RecordedEvaluation, RunReader, open_run
from auxein.strategies.ga import GeneticAlgorithm, SelfAdaptiveMutation, TournamentSelection

from auxein_scenarios.box_pushing import evaluation, fitness, scenarios
from auxein_scenarios.box_pushing.brain import save_brain
from auxein_scenarios.box_pushing.charts import draw_charts
from auxein_scenarios.box_pushing.config import ConfigError, TrainingConfig, dump_config, load_config
from auxein_scenarios.box_pushing.environment import BoxPushingWorld, brain
from auxein_scenarios.box_pushing.snapshots import default_generations, find_snapshots, is_snapshot, select
from auxein_scenarios.common.runs import timestamped

__all__ = [
    "BRAINS",
    "CHECKPOINT_EVERY",
    "CONFIG_FILE",
    "FINAL",
    "HELD_OUT_FILE",
    "PROGRESS_FILE",
    "RECORDING",
    "RUNS",
    "ProgressCsv",
    "Training",
    "TrainingError",
    "build",
    "estimate_generation_seconds",
    "held_out_set",
    "resolve_backend",
    "resume",
    "train",
    "training_set",
]

RUNS = Path("runs")
"""Where run directories go by default (ignored by git)."""

CONFIG_FILE = "config.toml"
PROGRESS_FILE = "progress.csv"
HELD_OUT_FILE = "held_out.csv"
RECORDING = "recording"
"""Auxein's run directory, inside the run's."""
BRAINS = "brains"
FINAL = "final.safetensors"

CHECKPOINT_EVERY = 10
"""Generations between checkpoints: an interrupted run loses at most this many generations' work when it resumes (Auxein
replays from the last checkpoint), and a checkpoint costs little next to a generation."""
TOURNAMENT_SIZE = 2
"""The GA's tournament size: the mildest selection pressure, which keeps the population diverse."""
CROSSOVER_PROBABILITY = 1.0
"""Every child is an intermediate recombination of two parents, then mutated."""
INITIAL_STEP = 0.1
"""The self-adaptive mutation's starting step, relative to the genome space's width."""
GENERATION_OVERHEAD = 1.4
"""A generation's wall time over its simulation's: recording, aggregation and breeding, measured on full-size runs."""

Print = Callable[[str], None]


class TrainingError(RuntimeError):
    """A run that cannot start or continue: a missing backend, a run directory in the way, a refused resume."""


# --- building a run ---


def _torch() -> Any:
    return importlib.import_module("torch")  # optional: imported by name, so the numpy-only install type-checks


def resolve_backend(config: TrainingConfig) -> tuple[Backend, str | None]:
    """The backend the config asks for, or torch on the CPU when its GPU is not available here (with a note saying so)."""
    run = config.run
    precision = cast("Any", run.precision)  # validated by RunConfig: float32 or float64
    if run.backend == "numpy":
        return Backend("numpy", "cpu", precision), None
    if importlib.util.find_spec("torch") is None:
        raise TrainingError("the config asks for torch, which is not installed: `uv sync --extra torch`, or set run.backend = 'numpy'")
    torch = _torch()
    available = {"mps": bool(torch.backends.mps.is_available()), "cuda": bool(torch.cuda.is_available()), "cpu": True}
    kind = run.device.split(":")[0]
    if not available.get(kind, False):
        note = (
            f"{run.device} is not available here: training with torch on the CPU instead, several times slower; the run records "
            "the backend it used, and resuming it uses the same"
        )
        return Backend("torch", "cpu", precision), note
    return Backend("torch", run.device, precision), None


def training_set(config: TrainingConfig) -> ScenarioSet:
    s = config.scenarios
    mix = tuple((d, count) for d, count in ((scenarios.EASY, s.easy), (scenarios.FULL, s.full)) if count > 0)
    return scenarios.training_set(config.team.agents, config.world, seed=s.seed, mix=mix)


def held_out_set(config: TrainingConfig) -> ScenarioSet:
    """The run's held-out set: `scenarios.held_out` full scenarios from `scenarios.held_out_seed`."""
    s = config.scenarios
    return scenarios.held_out_set(config.team.agents, config.world, seed=s.held_out_seed, count=s.held_out)


@dataclass(frozen=True)
class Training:
    """Everything a run is made of, built from its config (`build`)."""

    config: TrainingConfig
    backend: Backend
    world: BoxPushingWorld
    policy: MLPPolicy
    training_set: ScenarioSet
    evaluator: auxein.EpisodeEvaluator[Any]

    def strategy(self) -> GeneticAlgorithm:
        population = self.config.evolution.population
        return GeneticAlgorithm(
            population_size=population,
            offspring_size=population,
            selection=TournamentSelection(TOURNAMENT_SIZE),
            mutation=SelfAdaptiveMutation(initial_step=INITIAL_STEP),
            crossover_probability=CROSSOVER_PROBABILITY,
        )

    def arguments(self, run_dir: Path, callbacks: Sequence[Callback]) -> dict[str, Any]:
        """The arguments of `auxein.run` and `auxein.resume`, which must be the same for a resume (but the budget)."""
        c = self.config
        return {
            "strategy": self.strategy(),
            "evaluator": self.evaluator,
            "space": self.policy.space(),
            "objectives": [fitness.OBJECTIVE],
            "descriptors": list(fitness.DESCRIPTORS),
            "budget": auxein.Budget(evaluations=c.evaluations),
            "seed": c.run.seed,
            "backend": self.backend,
            "batch_size": c.evolution.population,
            "delivery": "generation",
            "deterministic": True,
            "checkpoint_every_evaluations": CHECKPOINT_EVERY * c.evolution.population,
            "run_dir": run_dir,
            "name": c.name,
            "callbacks": list(callbacks),
        }


def build(config: TrainingConfig, backend: Backend) -> Training:
    """The run's components: the training set, the world, the brain, the aggregator and the evaluator."""
    world = BoxPushingWorld(config.world, config.sensing)
    policy = brain(config.sensing, config.brain.hidden)
    train_set = training_set(config)
    evaluator = auxein.EpisodeEvaluator(policy, world, train_set, fitness.aggregator(config.fitness.weights, config.max_steps))
    return Training(config, backend, world, policy, train_set, evaluator)


def estimate_generation_seconds(training: Training, steps: int = 2) -> float:
    """Seconds of simulation per generation, measured: random brains of the run's population size on the training set for
    `steps` steps (after one to warm up), times the step limit. A generation also records, aggregates and breeds
    (`GENERATION_OVERHEAD`), and costs less once teams deliver early."""
    c, backend = training.config, training.backend
    n = c.evolution.population
    world = BoxPushingWorld(c.world, c.sensing, steps=1)
    layout = world.layout(tuple(training.training_set), backend)
    genomes = training.policy.space().sample_genomes(n, RandomStream(np.random.SeedSequence(0), backend), backend)
    agents = training.policy.decode_batch(genomes)
    world.run(agents.act_batch, layout, n)
    timed = BoxPushingWorld(c.world, c.sensing, steps=steps)
    start = time.perf_counter()
    timed.run(agents.act_batch, layout, n, stop_when_done=False)
    return (time.perf_counter() - start) / steps * c.max_steps


# --- progress.csv ---


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _csv(rows: Sequence[dict[str, object]]) -> str:
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return buffer.getvalue()


def _write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)  # atomic: a reader never sees half a file


class ProgressCsv:
    """A progress callback that rewrites `progress.csv` from the recording after every generation: Auxein's
    `generation_history()` (generation, evaluations, the best score so far, and the generation's score best, mean, median and
    worst), plus the means of the descriptors over the generation's successful candidates (success rate, progress, approach,
    `max_pushers`) and the best-so-far's descriptors, and the active seconds where this session saw the generation.

    Rebuilt from the recording rather than from the callbacks alone, so that a resumed run's file covers the whole run (a
    resume replays only from its last checkpoint) and holds the descriptors, which `Progress` does not carry."""

    def __init__(self, run_dir: Path) -> None:
        self.path = Path(run_dir) / PROGRESS_FILE
        self.recording = Path(run_dir) / RECORDING
        self._extra: dict[int, dict[str, object]] = {}
        self._elapsed: dict[int, float] = {}

    def __call__(self, progress: Progress[Any]) -> None:
        if progress.replaying:
            return
        if progress.generation and not progress.final:
            self._elapsed[progress.generation] = progress.elapsed
        self.write()

    def write(self) -> None:
        _write(self.path, _csv(self.rows()))

    def rows(self) -> list[dict[str, object]]:
        """The file's rows, from the recording (`elapsed` only for the generations this session saw)."""
        with open_run(self.recording) as reader:
            rows = reader.generation_history()
            for row in rows:
                generation = int(cast("int", row["generation"]))
                if generation not in self._extra:
                    self._extra[generation] = _descriptor_columns(reader.evaluations_in(generation), reader.best_up_to(generation))
        return [
            {
                **row,
                **self._extra[int(cast("int", row["generation"]))],
                "elapsed": self._elapsed.get(int(cast("int", row["generation"])), math.nan),
            }
            for row in rows
        ]


def _descriptor_columns(evaluations: Sequence[RecordedEvaluation], champion: RecordedEvaluation | None) -> dict[str, object]:
    ok = [e for e in evaluations if e.status.value == "ok"]
    columns: dict[str, object] = {}
    for name in fitness.DESCRIPTORS:
        columns[f"{name}_mean"] = _mean([float(e.descriptors[name]) for e in ok])
    for name in fitness.DESCRIPTORS:
        columns[f"champion_{name}"] = math.nan if champion is None else float(champion.descriptors[name])
    return columns


# --- brains/ ---


def _save(run_dir: Path, config: TrainingConfig, champion: RecordedEvaluation, generation: int, name: str, final: bool = False) -> Path:
    return save_brain(
        run_dir / BRAINS / name, champion.genome, config, run_dir=run_dir, generation=generation,
        candidate_id=int(champion.candidate_id), score=float(champion.objectives["score"]),
        success=float(champion.descriptors["success_rate"]), final=final,
    )  # fmt: skip


class BrainWriter:
    """A progress callback that saves the brain of each snapshot when its generation is reached: `brains/gen-<k>.safetensors`
    for `k` on the log-spaced schedule, unless its champion is the previous snapshot's (`snapshots.select`). The decision
    reads only the recording, so a resumed run makes the same files as an uninterrupted one."""

    def __init__(self, run_dir: Path, config: TrainingConfig) -> None:
        self.run_dir, self.config = run_dir, config

    def __call__(self, progress: Progress[Any]) -> None:
        k = progress.generation
        if progress.replaying or progress.final or not k or not is_snapshot(k):
            return
        with open_run(self.run_dir / RECORDING) as reader:
            chosen = [g for g in default_generations(k) if is_snapshot(g)]
            snapshots = select(chosen, [reader.best_up_to(g) for g in chosen])
        if snapshots and snapshots[-1].generation == k:
            _save(self.run_dir, self.config, snapshots[-1].champion, k, f"gen-{k:04d}.safetensors")


def _save_final(run_dir: Path, config: TrainingConfig) -> Path | None:
    """`brains/final.safetensors`: the run's best so far, or nothing if no generation has been told yet."""
    with open_run(run_dir / RECORDING) as reader:
        told = reader.generations()
        best = reader.best_up_to(told) if told else None
    if best is None:
        return None
    return _save(run_dir, config, best, told, FINAL, final=True)


# --- held_out.csv ---


def _number(value: float, digits: int = 6) -> str:
    return "" if math.isnan(value) else f"{value:.{digits}f}"


def held_out_rows(run_dir: Path, config: TrainingConfig, backend: Backend) -> list[dict[str, object]]:
    """Every snapshot brain (the log-spaced ones and the last generation's) judged on the run's held-out set, through
    Auxein's `evaluate_held_out` with an evaluator rebuilt from the config (the same world, brain and aggregator, another
    scenario set), with its training results from the recording: one row per snapshot."""
    recording = run_dir / RECORDING
    training, held = training_set(config), held_out_set(config)
    scenarios.check_disjoint(training, held)
    snapshots = find_snapshots(recording)
    world = BoxPushingWorld(config.world, config.sensing)
    evaluator = auxein.EpisodeEvaluator(
        brain(config.sensing, config.brain.hidden), world, held, fitness.aggregator(config.fitness.weights, config.max_steps)
    )
    judged = evaluate_held_out(recording, evaluator, held, candidates=[m.candidate_id for m in snapshots], backend=backend, write=False)
    by_id = {c.candidate_id: c for c in judged.candidates}
    score = evaluation.scorer(config.fitness, config.max_steps)
    rows: list[dict[str, object]] = []
    with open_run(recording) as reader:
        for m in snapshots:
            recorded = [dict(e.measurements) if e.status is Status.OK else None for e in reader.episodes(m.candidate_id)]
            on_held = [dict(s.measurements) if s.status is Status.OK else None for s in by_id[m.candidate_id].scenarios]
            train_episodes = evaluation.episodes(list(training), recorded, score)
            t = evaluation.summarise(train_episodes)
            easy = evaluation.summarise([e for e in train_episodes if e.difficulty == "easy"])
            full = evaluation.summarise([e for e in train_episodes if e.difficulty == "full"])
            h = evaluation.summarise(evaluation.episodes(list(held), on_held, score))
            rows.append(
                {
                    "generation": m.generation,
                    "candidate_id": m.candidate_id,
                    "train_score": _number(t.score),
                    "train_success": _number(t.success),
                    "train_easy_success": _number(easy.success),
                    "train_full_success": _number(full.success),
                    "held_episodes": h.episodes - h.failed,
                    "held_delivered": h.delivered,
                    "held_success": _number(h.success),
                    "held_progress": _number(h.progress),
                    "held_max_pushers": _number(h.max_pushers),
                    "held_delivery_steps_median": _number(h.delivery_steps, 1),
                    "gap": _number(full.success - h.success),
                }
            )
    return rows


# --- train and resume ---


def _uv(config: TrainingConfig) -> str:
    return "uv run --extra torch" if config.run.backend == "torch" else "uv run"


def _new_run_dir(config: TrainingConfig, run_dir: str | Path | None) -> Path:
    if run_dir is None:
        return timestamped(RUNS, config.name)
    path = Path(run_dir)
    if (path / RECORDING).exists() or (path / CONFIG_FILE).exists():
        raise TrainingError(f"{path} already holds a run: resume it with `--resume {path}`, or choose another directory")
    return path


def _finish(result: auxein.RunResult[Any], run_dir: Path, config: TrainingConfig, backend: Backend, out: Print) -> None:
    """After the last generation: the final brain, the held-out results, the charts, and lines saying where they are."""
    best = result.best
    if best is None:
        out(f"no candidate succeeded; the run is in {run_dir}")
        return
    d = best.descriptors
    out(
        f"best score {best.objectives['score']:.4f} (candidate {int(best.candidate.id)}): success rate {d['success_rate']:.0%} on "
        f"the training set, mean progress {d['progress']:.2f}, at most {d['max_pushers']:.1f} touching on average; "
        f"{result.evaluations_used:,} evaluations in {result.wall_time / 60:.1f} min ({result.stop_reason})"
    )
    final = _save_final(run_dir, config)
    started = time.perf_counter()
    rows = held_out_rows(run_dir, config, backend)
    _write(run_dir / HELD_OUT_FILE, _csv(rows))
    last = rows[-1]
    out(
        f"held-out: the final brain delivers {last['held_delivered']} of {last['held_episodes']} unseen scenarios "
        f"({float(str(last['held_success'])):.0%}; {time.perf_counter() - started:.0f} s for {len(rows)} snapshot brains)"
    )
    draw_charts(run_dir, config, run_dir / PROGRESS_FILE, run_dir / HELD_OUT_FILE, out=lambda _: None)
    brains = sorted(p.name for p in (run_dir / BRAINS).glob("*.safetensors"))
    out(f"the run is in {run_dir}: brains/ ({', '.join(brains)}), charts/, {HELD_OUT_FILE}, {PROGRESS_FILE}, recording/")
    if final is not None:
        out(f"deploy it: {_uv(config)} scenarios box-pushing deploy {final}")


def _run(training: Training, run_dir: Path, hooks: Sequence[Callback], resuming: bool) -> auxein.RunResult[Any]:
    """`auxein.run` or `auxein.resume`, saving the final brain if the run is interrupted (or fails) after a generation."""
    arguments = training.arguments(run_dir / RECORDING, hooks)
    try:
        return auxein.resume(**arguments) if resuming else auxein.run(**arguments)
    except BaseException:
        if (run_dir / RECORDING / "events.sqlite").exists():
            _save_final(run_dir, training.config)
        raise


def train(
    config: TrainingConfig,
    run_dir: str | Path | None = None,
    *,
    callbacks: Sequence[Callback] = (),
    out: Print = print,
    progress: bool = True,
    estimate: bool = True,
) -> tuple[auxein.RunResult[Any], Path]:
    """Trains a new run (module docstring); returns its result and its directory. `callbacks` are added to the progress line
    (`progress`), `progress.csv` and the brain writer; `estimate` measures the seconds per generation before starting."""
    path = _new_run_dir(config, run_dir)
    backend, note = resolve_backend(config)
    if note:
        out(f"note: {note}")
    training = build(config, backend)
    c = config
    sizes = (
        f"{c.team.agents} agents, population {c.evolution.population}, {len(training.training_set)} training scenarios "
        f"({c.scenarios.easy} easy, {c.scenarios.full} full), {c.max_steps} steps, {c.evolution.generations} generations"
    )
    expected = ""
    if estimate:
        seconds = estimate_generation_seconds(training)
        run_seconds = seconds * GENERATION_OVERHEAD * c.evolution.generations
        duration = f"{run_seconds / 3600.0:.1f} h" if run_seconds >= 3600 else f"{max(run_seconds / 60.0, 0.1):.1f} min"
        # the held-out judging at the end: every snapshot brain on every held-out scenario, in one batch
        judged = len(default_generations(c.evolution.generations)) * c.scenarios.held_out
        minutes = seconds * judged / (c.evolution.population * len(training.training_set)) / 60.0
        expected = (
            f"; about {seconds:.0f} s of simulation per generation, so about {duration} for the run, then about "
            f"{max(minutes, 0.1):.1f} min to judge its snapshot brains on {c.scenarios.held_out} held-out scenarios"
        )
    out(f"training {c.name}: {backend.name} {backend.device} {backend.precision}; {sizes}{expected}")
    path.mkdir(parents=True, exist_ok=True)
    (path / CONFIG_FILE).write_text(dump_config(config))
    out(f"run directory {path} (Ctrl-C stops it resumably: {_uv(config)} scenarios box-pushing train --resume {path})")
    hooks: list[Callback] = [*([print_progress()] if progress else []), ProgressCsv(path), BrainWriter(path, config), *callbacks]
    result = _run(training, path, hooks, resuming=False)
    _finish(result, path, config, backend, out)
    return result, path


def recorded_backend(run_dir: Path) -> Backend:
    """The backend a recorded run used (its metadata), to resume it on the same."""
    with open_run(Path(run_dir) / RECORDING) as reader:
        metadata = reader.metadata
    where = cast("dict[str, str]", metadata["backend"])
    return Backend(cast("Any", where["name"]), where["device"], cast("Any", metadata["precision"]))


def _told(run_dir: Path) -> int:
    reader: RunReader
    with open_run(run_dir / RECORDING) as reader:
        return reader.generations()


def resume(
    run_dir: str | Path,
    *,
    generations: int | None = None,
    callbacks: Sequence[Callback] = (),
    out: Print = print,
    progress: bool = True,
) -> tuple[auxein.RunResult[Any], Path]:
    """Continues a recorded run (module docstring), with its own `config.toml`, or with `generations` generations in all (more
    to extend it, or fewer to end it sooner, but not fewer than it already has). The run's `config.toml` is updated with the new
    number once the run has gone on."""
    path = Path(run_dir)
    if not (path / CONFIG_FILE).exists() or not (path / RECORDING / "events.sqlite").exists():
        raise TrainingError(f"{path} is not a box-pushing run: it needs {CONFIG_FILE} and {RECORDING}/events.sqlite")
    recorded = load_config(path / CONFIG_FILE)
    chosen = recorded if generations is None else recorded.with_generations(generations)
    told = _told(path)
    if chosen.evolution.generations < told:
        raise ConfigError(f"cannot resume {path} with {chosen.evolution.generations} generations: it already has {told}")
    backend = recorded_backend(path)
    training = build(chosen, backend)
    out(f"resuming {path}: {backend.name} {backend.device} {backend.precision}, {chosen.evolution.generations} generations")
    hooks: list[Callback] = [*([print_progress()] if progress else []), ProgressCsv(path), BrainWriter(path, chosen), *callbacks]
    try:
        result = _run(training, path, hooks, resuming=True)
    finally:
        if chosen != recorded and _told(path) > told:
            (path / CONFIG_FILE).write_text(dump_config(chosen))
    _finish(result, path, chosen, backend, out)
    return result, path
