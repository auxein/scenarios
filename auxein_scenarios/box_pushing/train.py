"""Training (design doc §6): a run built from its `TrainingConfig`, run, resumed and watched through Auxein.

- **`train(config)`** builds everything from the config (the scenario sets, the world, the brain, the aggregator, the
  `EpisodeEvaluator` and the `GeneticAlgorithm`) and runs Auxein with recording on, in deterministic mode, generation delivery
  and a checkpoint every `run.checkpoint_every` generations. The run directory (`runs/<name>-<timestamp>` by default) holds
  the effective config (`config.toml`, every default filled in), `progress.csv` (rewritten from the recording after every
  generation, so the run can be charted while it goes on), `clips/` (from `watch`) and **`recording/`, Auxein's run directory**
  (its event log, metadata and checkpoints; open it with `auxein.open_run`). Auxein's directory is a subdirectory of its own
  because Auxein starts a run only in an empty or new directory, and so that the two never mix.
- **`resume(run_dir)`** continues it from its recording and its `config.toml`: after an interruption (Ctrl-C leaves a run
  resumable: Auxein checkpoints between generations), or with a larger budget (`generations`). **Only the budget may change**,
  as Auxein allows; a config given to compare with is refused if anything else differs, fitness weights included.
- **`watch(run_dir)`** renders the best brain so far on a few showcase scenarios, with the run's backend, precision and device
  (design doc §7), through the environment's step loop. It reads the recording read-only, so it can run while training goes on.

The run stops on its budget only: no held-out evaluation or early stop during the run (design doc §6, §10.5).
"""

import csv
import importlib
import importlib.util
import io
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import auxein
import numpy as np
import numpy.typing as npt
from auxein.backend import Backend
from auxein.callbacks import print_progress
from auxein.driver.progress import Callback, Progress
from auxein.environments import MLPPolicy, Scenario, ScenarioSet
from auxein.random import RandomStream
from auxein.recording.reader import RecordedEvaluation, open_run
from auxein.strategies.ga import GeneticAlgorithm, SelfAdaptiveMutation, TournamentSelection

from auxein_scenarios.box_pushing import fitness, scenarios
from auxein_scenarios.box_pushing.config import ConfigError, TrainingConfig, config_differences, dump_config, load_config
from auxein_scenarios.box_pushing.environment import BoxPushingWorld, brain

__all__ = [
    "CONFIG_FILE",
    "PROGRESS_FILE",
    "RECORDING",
    "RUNS",
    "ProgressCsv",
    "Replay",
    "Showcase",
    "Training",
    "TrainingError",
    "build",
    "default_showcase",
    "outcome",
    "estimate_generation_seconds",
    "held_out_set",
    "progress_rows",
    "recorded_backend",
    "replay_backend",
    "replay",
    "resolve_backend",
    "resume",
    "showcase_scenarios",
    "train",
    "watch",
]

RUNS = Path("runs")
"""Where run directories go by default (ignored by git)."""

CONFIG_FILE = "config.toml"
PROGRESS_FILE = "progress.csv"
RECORDING = "recording"
"""Auxein's run directory, inside the run's."""

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
            f"{run.device} is not available here: training with torch on the CPU instead, several times slower "
            "(design doc §10.3); the run records the backend it used, and resuming or watching it uses the same"
        )
        return Backend("torch", "cpu", precision), note
    return Backend("torch", run.device, precision), None


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
        s = self.config.strategy
        return GeneticAlgorithm(
            population_size=s.population,
            offspring_size=s.offspring,
            selection=TournamentSelection(s.tournament_size),
            mutation=SelfAdaptiveMutation(initial_step=s.initial_step),
            crossover_probability=s.crossover_probability,
        )

    def arguments(self, run_dir: Path, callbacks: Sequence[Callback]) -> dict[str, Any]:
        """The arguments of `auxein.run` and `auxein.resume`, which must be the same for a resume (but the budget)."""
        c = self.config
        return {
            "strategy": self.strategy(),
            "evaluator": self.evaluator,
            "space": self.policy.space(c.brain.scale),
            "objectives": [fitness.OBJECTIVE],
            "descriptors": list(fitness.DESCRIPTORS),
            "budget": auxein.Budget(evaluations=c.evaluations),
            "seed": c.run.seed,
            "backend": self.backend,
            "batch_size": c.strategy.offspring,
            "delivery": "generation",
            "deterministic": True,
            "checkpoint_every_evaluations": c.run.checkpoint_every * c.strategy.offspring,
            "run_dir": run_dir,
            "name": c.name,
            "callbacks": list(callbacks),
        }


def training_set(config: TrainingConfig) -> ScenarioSet:
    s = config.scenarios
    easy, full = scenarios.difficulties(s.generator)
    mix = tuple((d, count) for d, count in ((easy, s.easy), (full, s.full)) if count > 0)
    return scenarios.training_set(config.team.agents, config.world, seed=s.seed, mix=mix)


def held_out_set(config: TrainingConfig, count: int | None = None) -> ScenarioSet:
    """The run's held-out set (design doc §4.3): `scenarios.held_out` full scenarios from `held_out_seed`, made by the run's
    generator version. `count` builds only the first ones (each layout has its own stream, so they are the whole set's first)."""
    s = config.scenarios
    _, full = scenarios.difficulties(s.generator)
    return scenarios.held_out_set(
        config.team.agents, config.world, seed=s.held_out_seed, count=s.held_out if count is None else count, difficulty=full
    )


def build(config: TrainingConfig, backend: Backend) -> Training:
    """The run's components (design doc §6): the training set, the world, the brain, the aggregator and the evaluator."""
    world = BoxPushingWorld(config.world, config.sensing)
    policy = brain(config.sensing, config.brain.hidden)
    train_set = training_set(config)
    evaluator = auxein.EpisodeEvaluator(policy, world, train_set, fitness.aggregator(config.fitness, config.max_steps))
    return Training(config, backend, world, policy, train_set, evaluator)


def estimate_generation_seconds(training: Training, steps: int = 2) -> float:
    """Seconds per generation, measured: random brains of the run's population size on the training set for `steps` steps
    (after one to warm up), times the step limit: the simulation alone. A generation also records, aggregates and breeds,
    which the pilot run measured at about a quarter more (23 s estimated, 29 s per generation on Metal), and costs less once
    teams deliver early."""
    c, backend = training.config, training.backend
    world = BoxPushingWorld(c.world, c.sensing, steps=1)
    layout = world.layout(tuple(training.training_set), backend)
    genomes = training.policy.space(c.brain.scale).sample_genomes(
        c.strategy.offspring, RandomStream(np.random.SeedSequence(0), backend), backend
    )
    agents = training.policy.decode_batch(genomes)
    world.run(agents.act_batch, layout, c.strategy.offspring)
    timed = BoxPushingWorld(c.world, c.sensing, steps=steps)
    start = time.perf_counter()
    timed.run(agents.act_batch, layout, c.strategy.offspring, stop_when_done=False)
    return (time.perf_counter() - start) / steps * c.max_steps


# --- progress.csv ---


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else math.nan


class ProgressCsv:
    """A progress callback that rewrites `progress.csv` from the recording after every generation: Auxein's
    `generation_history()` (generation, evaluations, the best score so far, and the generation's score best, mean, median and
    worst), plus the means of the descriptors over the generation's successful candidates (success rate, progress, approach,
    `max_pushers`) and the champion's descriptors, and the active seconds where this session saw the generation.

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

    def write(self) -> str:
        out = self.rows()
        buffer = io.StringIO()
        if out:
            writer = csv.DictWriter(buffer, fieldnames=list(out[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(out)
        text = buffer.getvalue()
        temporary = self.path.with_suffix(".csv.tmp")
        temporary.write_text(text)
        temporary.replace(self.path)  # atomic: a reader never sees half a file
        return text

    def rows(self) -> list[dict[str, object]]:
        """The file's rows, from the recording (`elapsed` only for the generations this session saw)."""
        with open_run(self.recording) as reader:
            rows = reader.generation_history()
            for row in rows:
                generation = int(cast("int", row["generation"]))
                if generation not in self._extra:
                    self._extra[generation] = _descriptor_columns(reader.evaluations_in(generation), reader.best_up_to(generation))
        out: list[dict[str, object]] = []
        for row in rows:
            generation = int(cast("int", row["generation"]))
            out.append({**row, **self._extra[generation], "elapsed": self._elapsed.get(generation, math.nan)})
        return out


def progress_rows(run_dir: str | Path) -> list[dict[str, float]]:
    """The run's per-generation rows as numbers: from its `progress.csv`, or rebuilt from the recording if the file is
    missing (without writing it, so the active times it holds are never lost)."""
    path = Path(run_dir) / PROGRESS_FILE
    if path.exists():
        with open(path) as file:
            raw: list[dict[str, object]] = [dict(row) for row in csv.DictReader(file)]
    else:
        raw = ProgressCsv(Path(run_dir)).rows()

    def number(value: object) -> float:
        try:
            return float(cast("float", value)) if value not in ("", None) else math.nan
        except (TypeError, ValueError):
            return math.nan

    return [{key: number(value) for key, value in row.items()} for row in raw]


def _descriptor_columns(evaluations: Sequence[RecordedEvaluation], champion: RecordedEvaluation | None) -> dict[str, object]:
    ok = [e for e in evaluations if e.status.value == "ok"]
    columns: dict[str, object] = {}
    for name in fitness.DESCRIPTORS:
        columns[f"{name}_mean"] = _mean([float(e.descriptors[name]) for e in ok])
    for name in fitness.DESCRIPTORS:
        columns[f"champion_{name}"] = math.nan if champion is None else float(champion.descriptors[name])
    return columns


# --- train and resume ---


def _new_run_dir(config: TrainingConfig, run_dir: str | Path | None) -> Path:
    if run_dir is None:
        return RUNS / f"{config.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    path = Path(run_dir)
    if (path / RECORDING).exists() or (path / CONFIG_FILE).exists():
        raise TrainingError(f"{path} already holds a run: resume it with `box-pushing train --resume {path}`, or choose another directory")
    return path


def _summary(result: auxein.RunResult[Any], run_dir: Path) -> list[str]:
    best = result.best
    if best is None:
        return [f"no candidate succeeded; the run is in {run_dir}"]
    d = best.descriptors
    return [
        f"best score {best.objectives['score']:.4f} (candidate {int(best.candidate.id)}): success rate {d['success_rate']:.0%} on "
        f"the training set, mean progress {d['progress']:.2f}, mean approach {d['approach']:.2f}, "
        f"at most {d['max_pushers']:.1f} touching on average",
        f"{result.evaluations_used:,} evaluations in {result.wall_time / 60:.1f} min ({result.stop_reason}); the run is in {run_dir}",
        f"watch it: uv run box-pushing watch {run_dir}",
    ]


def _generator_note(config: TrainingConfig) -> str:
    generator = config.scenarios.generator
    if generator == 1:
        return "generator 1 (before M2b: a box may start where agents cannot get behind it)"
    _, full = scenarios.difficulties(generator)
    return f"generator {generator} (the box starts {full.box_clearance:g} unit clear of walls and obstacles)"


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
    (`progress`) and `progress.csv`; `estimate` measures the seconds per generation before starting."""
    path = _new_run_dir(config, run_dir)
    backend, note = resolve_backend(config)
    if note:
        out(f"note: {note}")
    training = build(config, backend)
    c = config
    sizes = (
        f"{c.team.agents} agents, population {c.strategy.population} + {c.strategy.offspring} children, "
        f"{len(training.training_set)} training scenarios ({c.scenarios.easy} easy, {c.scenarios.full} full), {c.max_steps} steps, "
        f"{c.budget.generations} generations"
    )
    expected = ""
    if estimate:
        seconds = estimate_generation_seconds(training)
        # a generation is its simulation plus recording, aggregation and breeding: the pilots measured 1.26 (M2a, 16 scenarios)
        # and 1.4 (M2b, 32) times the simulation (§10.7, §10.9)
        hours = seconds * 1.4 * c.budget.generations / 3600.0
        expected = f", about {seconds:.0f} s of simulation per generation (about {hours:.1f} h for the run)"
    out(f"training {c.name}: {backend.name} {backend.device} {backend.precision}; {sizes}{expected}")
    out(f"scenarios: {_generator_note(c)}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.toml").write_text(dump_config(config))
    out(f"run directory {path} (Ctrl-C stops it resumably: uv run box-pushing train --resume {path})")
    hooks: list[Callback] = [*([print_progress()] if progress else []), ProgressCsv(path), *callbacks]
    result = auxein.run(**training.arguments(path / RECORDING, hooks))
    for line in _summary(result, path):
        out(line)
    return result, path


def recorded_backend(run_dir: Path) -> Backend:
    """The backend a recorded run used (its metadata), to resume or watch it on the same."""
    with open_run(Path(run_dir) / RECORDING) as reader:
        metadata = reader.metadata
    where = cast("dict[str, str]", metadata["backend"])
    return Backend(cast("Any", where["name"]), where["device"], cast("Any", metadata["precision"]))


def replay_backend(run_dir: str | Path) -> tuple[Backend, str | None]:
    """The backend to replay a recorded run's brains on (clips, the held-out report): the run's own backend, precision and
    device (design doc §7), or torch on the CPU, with a note saying so, when the run's GPU is not available here."""
    recorded = recorded_backend(Path(run_dir))
    if recorded.name != "torch" or recorded.device == "cpu":
        return recorded, None
    resolved, note = resolve_backend(load_config(Path(run_dir) / CONFIG_FILE))
    return Backend("torch", resolved.device, recorded.precision), note


def resume(
    run_dir: str | Path,
    *,
    generations: int | None = None,
    config: TrainingConfig | None = None,
    callbacks: Sequence[Callback] = (),
    out: Print = print,
    progress: bool = True,
) -> tuple[auxein.RunResult[Any], Path]:
    """Continues a recorded run: after an interruption, or with another budget (`generations`: more to extend it, or fewer to
    end it sooner, but not fewer than it already has). The run's own `config.toml`
    defines it; a `config` given to compare with is refused if it differs in anything but the budget (whose own budget then
    applies, unless `generations` is given). The run's `config.toml` is updated with the new budget once the run has gone on."""
    path = Path(run_dir)
    if not (path / CONFIG_FILE).exists() or not (path / RECORDING / "events.sqlite").exists():
        raise TrainingError(f"{path} is not a box-pushing run: it needs {CONFIG_FILE} and {RECORDING}/events.sqlite")
    recorded = load_config(path / CONFIG_FILE)
    if config is not None:
        differences = config_differences(recorded, config)
        if differences:
            listing = "\n".join(f"  - {line}" for line in differences)
            raise ConfigError(f"cannot resume {path}: only the budget may change when resuming a run, but the config differs:\n{listing}")
    chosen = config if config is not None else recorded
    if generations is not None:
        chosen = chosen.with_generations(generations)
    with open_run(path / RECORDING) as reader:
        told = reader.generations()
    if chosen.budget.generations < told:
        raise ConfigError(f"cannot resume {path} with {chosen.budget.generations} generations: it already has {told}")
    backend = recorded_backend(path)
    training = build(chosen, backend)
    out(f"resuming {path}: {backend.name} {backend.device} {backend.precision}, budget {chosen.budget.generations} generations")
    hooks: list[Callback] = [*([print_progress()] if progress else []), ProgressCsv(path), *callbacks]
    started = False

    def mark(progress: Progress[Any]) -> None:
        nonlocal started
        started = True

    try:
        result = auxein.resume(**training.arguments(path / RECORDING, [mark, *hooks]))
        started = True
    finally:
        if started and chosen != recorded:
            (path / CONFIG_FILE).write_text(dump_config(chosen))
    for line in _summary(result, path):
        out(line)
    return result, path


# --- watching a run ---


@dataclass(frozen=True)
class Showcase:
    """A scenario to watch a run on: `set` is `"train"` or `"held"`, `index` its place in that set."""

    set: str
    index: int

    @staticmethod
    def parse(text: str) -> "Showcase":
        name, _, index = text.partition(":")
        if name not in ("train", "held") or not index.isdigit():
            raise ValueError(f"a showcase scenario is train:<index> or held:<index>, got {text!r}")
        return Showcase(name, int(index))

    @property
    def note(self) -> str:
        """The panel's honest label (design doc §7): a clip never passes a training scenario off as an unseen one."""
        return "training scenario" if self.set == "train" else "held-out: never seen in training"


def outcome(measurements: dict[str, float]) -> str:
    """How an episode ended, for a panel's footer: delivered at step k, or the progress made."""
    d0, d = measurements["initial_distance"], measurements["final_distance"]
    if measurements["delivered"] > 0.5:
        return f"delivered at step {int(measurements['steps'])}"
    return f"not delivered  ·  progress {(d0 - d) / d0 if d0 > 0 else 0.0:+.2f}"


def default_showcase(config: TrainingConfig) -> list[Showcase]:
    """Two training scenarios (the first easy and the first full, if both are in the set) and the first two held-out ones."""
    s = config.scenarios
    chosen = [Showcase("train", 0)]
    if s.easy and s.full:
        chosen.append(Showcase("train", s.easy))
    return [*chosen, Showcase("held", 0), Showcase("held", min(1, s.held_out - 1))]


def showcase_scenarios(config: TrainingConfig, chosen: Sequence[Showcase]) -> list[Scenario]:
    """The scenarios of `chosen`, from the run's training and held-out sets (generated from the config)."""
    train_set = training_set(config)
    held = [c.index for c in chosen if c.set == "held"]
    held_set = None
    if held:
        if max(held) >= config.scenarios.held_out:
            raise ValueError(f"the held-out set has {config.scenarios.held_out} scenarios, not {max(held) + 1}")
        # each layout comes from its own stream (by index), so the first k of the set are the first k of the whole set
        held_set = held_out_set(config, count=max(held) + 1)
    out: list[Scenario] = []
    for c in chosen:
        source = train_set if c.set == "train" else held_set
        assert source is not None
        if c.index >= len(source):
            raise ValueError(f"the training set has {len(source)} scenarios, not {c.index + 1}")
        out.append(source[c.index])
    return out


@dataclass(frozen=True)
class Replay:
    """Champions of a run replayed on showcase scenarios through the environment's step loop, on the run's backend (design
    doc §7): what `watch`, the milestone clips and the montage draw. Row `i` of the trajectory and of `measurements` (host
    arrays, `(n, s)`) is the `i`th genome replayed; column `j` is `scenarios[j]`, chosen as `showcase[j]`."""

    config: TrainingConfig
    backend: Backend
    showcase: list[Showcase]
    scenarios: list[Scenario]
    trajectory: Any  # world.Trajectory (Any: train.py imports the world lazily, through environment)
    senses: Any  # environment.SensingTrace | None
    measurements: dict[str, Any]

    @property
    def labels(self) -> list[str]:
        return [f"{s.id} · {s.params['difficulty']}" for s in self.scenarios]

    def episode(self, row: int, column: int) -> dict[str, float]:
        return {name: float(values[row, column]) for name, values in self.measurements.items()}

    def footers(self, row: int) -> list[tuple[str, str]]:
        """Each panel's footer for the `row`th genome: its honest label and how its episode ended."""
        return [(c.note, outcome(self.episode(row, j))) for j, c in enumerate(self.showcase)]

    def records(self, row: int) -> int:
        """How many records the `row`th genome's clip needs: up to the step its last showcase episode ended (the batch runs
        on until every genome's episodes have ended, frozen once they have)."""
        done = np.asarray(self.trajectory.done)[:, row, :].all(axis=1)
        ended = np.nonzero(done)[0]
        return int(ended[0]) + 1 if ended.size else int(self.trajectory.records)


def replay(
    run_dir: str | Path, genomes: Sequence[object], showcase: Sequence[Showcase], *, overlay: bool = False, out: Print = print
) -> Replay:
    """Replays `genomes` (recorded genomes of the run) on `showcase`, all in one batch, with the sensing overlay of agent 0
    if `overlay`."""
    run = Path(run_dir)
    config = load_config(run / CONFIG_FILE)
    backend, note = replay_backend(run)
    if note:
        out(f"note: {note}; the clip may differ from what was scored (design doc §7)")
    chosen = list(showcase)
    selected = showcase_scenarios(config, chosen)
    world = BoxPushingWorld(config.world, config.sensing)
    policy = brain(config.sensing, config.brain.hidden)
    rows: list[npt.NDArray[np.float64]] = [np.asarray(g, dtype=np.float64).reshape(-1) for g in genomes]
    stacked = backend.asarray(np.asarray(rows, dtype=np.float64))
    trajectory, senses, rollout = world.watch(policy.decode_batch(stacked), selected, agents=(0,) if overlay else ())
    measured = {name: np.asarray(backend.to_numpy(v), dtype=np.float64) for name, v in rollout.measurements.items()}
    return Replay(config, backend, chosen, selected, trajectory, senses, measured)


def watch(
    run_dir: str | Path,
    *,
    generation: int | None = None,
    showcase: Sequence[Showcase] | None = None,
    overlay: bool = False,
    path: str | Path | None = None,
    fps: float = 30.0,
    speedup: float = 2.0,
    out: Print = print,
) -> Path:
    """Renders the best brain up to `generation` (the last told one by default) on showcase scenarios, with the run's backend,
    precision and device (design doc §7), through the environment's step loop. Returns the clip's path (by default in the run
    directory's `clips/`). Reads the recording read-only: safe while the run is training."""
    from auxein_scenarios.box_pushing.render import render_clip  # matplotlib, only when rendering

    run = Path(run_dir)
    config = load_config(run / CONFIG_FILE)
    with open_run(run / RECORDING) as reader:
        told = reader.generations()
        if told == 0:
            raise TrainingError(f"{run} has no generation yet: wait for the first one")
        k = told if generation is None else generation
        if not 1 <= k <= told:
            raise TrainingError(f"{run} has generations 1 to {told}, not {k}")
        best = reader.best_up_to(k)
    if best is None:
        raise TrainingError(f"no candidate of {run} succeeded by generation {k}")
    played = replay(run, [best.genome], list(showcase) if showcase is not None else default_showcase(config), overlay=overlay, out=out)
    score = best.objectives["score"]
    target = Path(path) if path is not None else run / "clips" / f"best-gen{k:04d}{'-overlay' if overlay else ''}.mp4"
    clip = render_clip(
        played.trajectory,
        [dict(s.params) for s in played.scenarios],
        config.world,
        target,
        cells=[(0, i) for i in range(len(played.scenarios))],
        labels=played.labels,
        senses=played.senses,
        sensing=config.sensing,
        kicker=f"Run · {config.name}",
        title=f"The best brain after generation {k}",
        subtitle=f"Candidate {int(best.candidate_id)}, score {score:.3f} on the training set "
        f"(success rate {best.descriptors['success_rate']:.0%}). Training and held-out scenarios, {config.team.agents} agents.",
        fps=fps,
        speedup=speedup,
        columns=2,
        footers=played.footers(0),
    )
    out(str(clip))
    for i, label in enumerate(played.labels):
        m = played.episode(0, i)
        d0, d = m["initial_distance"], m["final_distance"]
        state = f"delivered at step {int(m['steps'])}" if m["delivered"] > 0.5 else "not delivered"
        out(f"  {label}: {state}, progress {(d0 - d) / d0:+.2f}, most touching the box {int(m['max_pushers'])}")
    return clip
