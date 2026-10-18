"""The held-out report (design doc §7.2): how a run's champions do on scenarios they never saw, against two baselines.

`box-pushing report DIR` answers the question the run cannot answer by itself: **did the team learn to push boxes, or did it
learn these sixteen maps?** Everything is rebuilt from the run directory (its `config.toml` and recording):

- **Who is evaluated.** The champion of every milestone (`milestones.py`), through Auxein's `evaluate_held_out` with their
  candidate ids, on the run's **held-out set** (`scenarios.held_out` full-difficulty layouts from `scenarios.held_out_seed`,
  disjoint from the training set by construction; `check_disjoint` verifies it), with an `EpisodeEvaluator` rebuilt from the
  run: the same world, brain and aggregator, another scenario set. Nothing is written into the recording (`write=False`);
  Auxein's report goes to `DIR/report/held_out.json` instead.
- **Two baselines on the same scenarios**, to put "evolved" in context: **random brains** (`BASELINE_BRAINS` networks sampled
  from `MLPPolicy.space()` with a fixed seed, as the GA's first generation was) and the **senses-only controller** (M1b, a
  hand-written memoryless policy on the same 43 inputs). They run through the same world on both sets.
- **Per milestone:** the training-set results from the recording (all, easy and full separately), the held-out results
  (success rate, mean progress, mean `max_pushers`, the median delivery step of the delivered), and **the generalisation
  gap**, full-training success minus held-out success: both are full difficulty, so the gap is what the training maps added.
- **The recommendation** (`recommend`), from the final champion's held-out success and gap, with the numbers behind it.

Outputs, under `DIR/report/`: `report.md` (the table and a short paragraph per finding), `held_out.csv` (one row per
champion and baseline), `held_out_episodes.csv` (one row per subject and held-out scenario: what the clips' `best-held` and
`worst-held` read) and `held_out.json` (Auxein's `HeldOutReport`). **Deterministic**: the same command twice gives the same
files, on a given backend (float32 on a GPU depends slightly on the batch's shape, so a clip of one champion on four scenarios
can differ a little from the report, which evaluates all champions on all held-out scenarios at once; design doc §10.7).
"""

import csv
import io
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import auxein
import numpy as np
from auxein.backend import Backend
from auxein.core import Status
from auxein.driver.held_out import HeldOutReport, evaluate_held_out
from auxein.environments import ScenarioSet
from auxein.random import RandomStream
from auxein.recording.reader import open_run

from auxein_scenarios.box_pushing import fitness, scenarios, scripted, train
from auxein_scenarios.box_pushing.config import TrainingConfig, load_config
from auxein_scenarios.box_pushing.environment import BoxPushingWorld, brain
from auxein_scenarios.box_pushing.milestones import Milestone, find_milestones

__all__ = [
    "BASELINE_BRAINS",
    "BASELINE_SEED",
    "EPISODES_FILE",
    "GAP_THRESHOLD",
    "JSON_FILE",
    "MARKDOWN_FILE",
    "REPORT",
    "SUCCESS_THRESHOLD",
    "SUMMARY_FILE",
    "Episode",
    "Recommendation",
    "Report",
    "Results",
    "Subject",
    "build_report",
    "held_out_set",
    "read_episodes",
    "recommend",
    "summarise",
    "write_report",
]

REPORT = "report"
"""The report's directory, inside the run's."""
MARKDOWN_FILE = "report.md"
SUMMARY_FILE = "held_out.csv"
EPISODES_FILE = "held_out_episodes.csv"
JSON_FILE = "held_out.json"

BASELINE_BRAINS = 20
"""How many random brains make the random baseline: enough for a stable mean, a fifth of a generation's cost."""
BASELINE_SEED = 0
"""The random brains' seed, fixed so the report is deterministic."""

SUCCESS_THRESHOLD = 0.5
"""Decision 7: a final champion delivering at least half the held-out scenarios generalises."""
GAP_THRESHOLD = 0.25
"""Decision 7: a held-out success more than this below the full-training success means the team learned the maps."""
BORDERLINE = 0.03
"""A gap this close to `GAP_THRESHOLD` is at it ("about 0.25"): the report says the other case is indicated too."""


# --- results ---


@dataclass(frozen=True)
class Episode:
    """One subject (a champion or a baseline) on one scenario: the measurements the report reads."""

    scenario_id: str
    difficulty: str
    distance: float
    """The box's starting distance to the target."""
    obstacles: int
    ok: bool
    delivered: bool
    steps: int
    progress: float
    """Clipped at the fitness's floor, as the score and the `progress` descriptor are."""
    max_pushers: int
    score: float


@dataclass(frozen=True)
class Results:
    """A subject's summary over a set of episodes. Means over the episodes that ran (`ok`); `failed` counts the others."""

    episodes: int
    failed: int
    delivered: int
    success: float
    progress: float
    max_pushers: float
    delivery_steps: float
    """The median delivery step of the delivered episodes, NaN when none was delivered."""
    score: float


def summarise(episodes: Sequence[Episode]) -> Results:
    ok = [e for e in episodes if e.ok]
    delivered = [e for e in ok if e.delivered]

    def mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else math.nan

    return Results(
        episodes=len(episodes),
        failed=len(episodes) - len(ok),
        delivered=len(delivered),
        success=mean([1.0 if e.delivered else 0.0 for e in ok]),
        progress=mean([e.progress for e in ok]),
        max_pushers=mean([float(e.max_pushers) for e in ok]),
        delivery_steps=float(np.median([e.steps for e in delivered])) if delivered else math.nan,
        score=mean([e.score for e in ok]),
    )


@dataclass(frozen=True)
class Subject:
    """A champion (`kind` "champion", with its generation and candidate) or a baseline, on the training and held-out sets."""

    kind: Literal["champion", "random brains", "senses only"]
    name: str
    generation: int | None
    candidate_id: int | None
    training: tuple[Episode, ...]
    held_out: tuple[Episode, ...]

    def train(self, difficulty: str | None = None) -> Results:
        return summarise([e for e in self.training if difficulty is None or e.difficulty == difficulty])

    def held(self) -> Results:
        return summarise(self.held_out)

    @property
    def gap(self) -> float:
        """Full-training success minus held-out success: both full difficulty (NaN without full training scenarios)."""
        return self.train("full").success - self.held().success


@dataclass(frozen=True)
class Recommendation:
    """What the report recommends doing next (decision 7), and the numbers behind it."""

    case: Literal["exploit", "generalises", "learned the maps", "limited"]
    next_milestone: str
    reason: str


def recommend(held_success: float, train_full_success: float, exploit: str | None = None) -> Recommendation:
    """Decision 7, from the final champion's held-out success rate (full difficulty) and its full-training success.

    In this order: an **exploit** seen in the clips is fixed first (M2b, physics or fitness), whatever the numbers; **at least
    half** delivered on held-out means the scenario generalises (M4); a **gap above `GAP_THRESHOLD`** means it learned the maps
    (M2b, more full training scenarios). Otherwise it is below half without a large gap: the limit is the task as set, not
    overfitting, so M2b looks at what stops it before adding scenarios (a case decision 7 did not name)."""
    gap = round(train_full_success - held_success, 9)  # so that a gap of exactly the threshold is not above it by a rounding
    numbers = f"held-out success {held_success:.0%}, full-training success {train_full_success:.0%}, gap {gap:+.2f}"
    if exploit:
        return Recommendation("exploit", "M2b: fix the physics or the fitness first", f"the clips show an exploit ({exploit}); {numbers}")
    if held_success >= SUCCESS_THRESHOLD:
        return Recommendation(
            "generalises",
            "M4: the CMA-ES comparison and polished clips",
            f"{numbers}: at least {SUCCESS_THRESHOLD:.0%} of unseen layouts delivered",
        )
    if gap > GAP_THRESHOLD:
        return Recommendation(
            "learned the maps",
            "M2b: more full training scenarios (e.g. 4 easy + 28 full) and retrain",
            f"{numbers}: held-out success more than {GAP_THRESHOLD:.2f} below training",
        )
    return Recommendation(
        "limited",
        "M2b: what stops it on full scenarios (the step limit, the search) before more scenarios",
        f"{numbers}: below {SUCCESS_THRESHOLD:.0%} on unseen layouts, but within {GAP_THRESHOLD:.2f} of training",
    )


@dataclass(frozen=True)
class Report:
    """A run's held-out report: its champions and baselines, the held-out set, and Auxein's own report."""

    run_dir: Path
    config: TrainingConfig
    backend: Backend
    subjects: tuple[Subject, ...]
    auxein: HeldOutReport
    notes: tuple[str, ...] = field(default=())

    @property
    def champions(self) -> list[Subject]:
        return [s for s in self.subjects if s.kind == "champion"]

    @property
    def final(self) -> Subject:
        return self.champions[-1]

    def recommendation(self, exploit: str | None = None) -> Recommendation:
        return recommend(self.final.held().success, self.final.train("full").success, exploit)


# --- building it ---


def held_out_set(config: TrainingConfig) -> ScenarioSet:
    """The run's held-out set, from its config: `held_out` full scenarios from `held_out_seed`."""
    s = config.scenarios
    return scenarios.held_out_set(config.team.agents, config.world, seed=s.held_out_seed, count=s.held_out)


def _layout_facts(scenario_set: ScenarioSet) -> list[tuple[str, str, float, int]]:
    facts: list[tuple[str, str, float, int]] = []
    for s in scenario_set:
        p = s.params
        box, target = cast("list[float]", p["box"]), cast("list[float]", p["target"])
        obstacles = len(cast("list[object]", p["circles"])) + len(cast("list[object]", p["rects"]))
        facts.append((s.id, str(p["difficulty"]), math.dist(box[:2], target), obstacles))
    return facts


def _episodes(
    facts: Sequence[tuple[str, str, float, int]],
    measurements: Sequence[Mapping[str, float] | None],
    score: Callable[[Mapping[str, float]], float],
    floor: float,
) -> tuple[Episode, ...]:
    out: list[Episode] = []
    for (scenario_id, difficulty, distance, obstacles), m in zip(facts, measurements, strict=True):
        if m is None:  # a failed episode: no measurements
            out.append(Episode(scenario_id, difficulty, distance, obstacles, False, False, 0, math.nan, 0, math.nan))
            continue
        d0, d = m["initial_distance"], m["final_distance"]
        progress = max(floor, (d0 - d) / d0) if d0 > 0 else 0.0
        out.append(
            Episode(
                scenario_id, difficulty, distance, obstacles, True, m["delivered"] > 0.5, int(m["steps"]), progress,
                int(m["max_pushers"]), score(m),
            )
        )  # fmt: skip
    return tuple(out)


def _scorer(config: TrainingConfig) -> Callable[[Mapping[str, float]], float]:
    """The episode score of one episode's measurements, with the run's weights (fitness.EpisodeScore, on the host)."""
    episode_score = fitness.EpisodeScore(config.fitness, config.max_steps)

    def score(m: Mapping[str, float]) -> float:
        arrays = {name: np.asarray([[value]], dtype=np.float64) for name, value in m.items()}
        return float(np.asarray(episode_score(cast("Any", arrays)))[0, 0])

    return score


def _simulated(world: BoxPushingWorld, policy: Any, scenario_set: ScenarioSet, backend: Backend) -> list[list[dict[str, float] | None]]:
    """Per candidate of `policy`, per scenario, its measurements (None for a failed episode), through `run_batch`."""
    result = world.run_batch(policy, tuple(scenario_set), RandomStream(np.random.SeedSequence(0), backend))
    host = {name: np.asarray(backend.to_numpy(v), dtype=np.float64) for name, v in result.measurements.items()}
    n, s = next(iter(host.values())).shape
    failed = set(result.failures)
    return [
        [None if (i, j) in failed else {name: float(values[i, j]) for name, values in host.items()} for j in range(s)] for i in range(n)
    ]


def build_report(
    run_dir: str | Path,
    *,
    generations: Sequence[int] | None = None,
    out: Callable[[str], None] = print,
) -> Report:
    """Evaluates the run's milestone champions and the baselines (module docstring). Writes nothing: see `write_report`."""
    run = Path(run_dir)
    config = load_config(run / train.CONFIG_FILE)
    recording = run / train.RECORDING
    milestones = find_milestones(recording, generations)
    backend, note = train.replay_backend(run)
    notes = [f"{note}; the numbers may differ slightly from the run's device"] if note else []
    for line in notes:
        out(f"note: {line}")

    training = train.training_set(config)
    held = held_out_set(config)
    scenarios.check_disjoint(training, held)
    world = BoxPushingWorld(config.world, config.sensing)
    policy = brain(config.sensing, config.brain.hidden)
    aggregator = fitness.aggregator(config.fitness, config.max_steps)
    evaluator = auxein.EpisodeEvaluator(policy, world, held, aggregator)
    score, floor = _scorer(config), config.fitness.progress_floor
    train_facts, held_facts = _layout_facts(training), _layout_facts(held)

    out(f"held-out: {len(milestones)} champions on {len(held)} unseen scenarios ({backend.name} {backend.device} {backend.precision})")
    report = evaluate_held_out(recording, evaluator, held, candidates=[m.candidate_id for m in milestones], backend=backend, write=False)
    by_id = {c.candidate_id: c for c in report.candidates}
    subjects: list[Subject] = []
    with open_run(recording) as reader:
        for m in milestones:
            recorded = reader.episodes(m.candidate_id)
            train_measurements = [dict(e.measurements) if e.status is Status.OK else None for e in recorded]
            held_measurements = [dict(s.measurements) if s.status is Status.OK else None for s in by_id[m.candidate_id].scenarios]
            subjects.append(_champion(m, train_facts, held_facts, train_measurements, held_measurements, score, floor))

    out(f"baselines: {BASELINE_BRAINS} random brains and the senses-only controller, on both sets")
    genomes = policy.space(config.brain.scale).sample_genomes(
        BASELINE_BRAINS, RandomStream(np.random.SeedSequence(BASELINE_SEED), backend), backend
    )
    random_brains = policy.decode_batch(genomes)
    senses = scripted.SensesController(1, backend, world.sensing, config.world)
    for kind, name, agents in (
        ("random brains", f"{BASELINE_BRAINS} random brains", random_brains),
        ("senses only", "senses-only controller", senses),
    ):
        on_train = [e for per in _simulated(world, agents, training, backend) for e in _episodes(train_facts, per, score, floor)]
        on_held = [e for per in _simulated(world, agents, held, backend) for e in _episodes(held_facts, per, score, floor)]
        subjects.append(Subject(cast("Any", kind), name, None, None, tuple(on_train), tuple(on_held)))
    return Report(run, config, backend, tuple(subjects), report, tuple(notes))


def _champion(
    m: Milestone,
    train_facts: Sequence[tuple[str, str, float, int]],
    held_facts: Sequence[tuple[str, str, float, int]],
    train_measurements: Sequence[Mapping[str, float] | None],
    held_measurements: Sequence[Mapping[str, float] | None],
    score: Callable[[Mapping[str, float]], float],
    floor: float,
) -> Subject:
    return Subject(
        "champion",
        f"generation {m.generation}",
        m.generation,
        m.candidate_id,
        _episodes(train_facts, train_measurements, score, floor),
        _episodes(held_facts, held_measurements, score, floor),
    )


# --- writing it ---


def _number(value: float, digits: int = 3) -> str:
    return "" if math.isnan(value) else f"{value:.{digits}f}"


def _summary_rows(report: Report) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for s in report.subjects:
        t, te, tf, h = s.train(), s.train("easy"), s.train("full"), s.held()
        rows.append(
            {
                "kind": s.kind,
                "name": s.name,
                "generation": "" if s.generation is None else s.generation,
                "candidate_id": "" if s.candidate_id is None else s.candidate_id,
                "train_episodes": t.episodes,
                "train_score": _number(t.score, 6),
                "train_success": _number(t.success, 6),
                "train_easy_success": _number(te.success, 6),
                "train_full_success": _number(tf.success, 6),
                "train_progress": _number(t.progress, 6),
                "train_max_pushers": _number(t.max_pushers, 6),
                "held_episodes": h.episodes,
                "held_failed": h.failed,
                "held_delivered": h.delivered,
                "held_score": _number(h.score, 6),
                "held_success": _number(h.success, 6),
                "held_progress": _number(h.progress, 6),
                "held_max_pushers": _number(h.max_pushers, 6),
                "held_delivery_steps_median": _number(h.delivery_steps, 1),
                "gap": _number(s.gap, 6),
            }
        )
    return rows


def _episode_rows(report: Report) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for s in report.subjects:
        for e in s.held_out:
            rows.append(
                {
                    "kind": s.kind,
                    "name": s.name,
                    "generation": "" if s.generation is None else s.generation,
                    "candidate_id": "" if s.candidate_id is None else s.candidate_id,
                    "scenario_id": e.scenario_id,
                    "distance": _number(e.distance, 3),
                    "obstacles": e.obstacles,
                    "ok": int(e.ok),
                    "delivered": int(e.delivered),
                    "steps": e.steps,
                    "progress": _number(e.progress, 6),
                    "max_pushers": e.max_pushers,
                    "score": _number(e.score, 6),
                }
            )
    return rows


def _csv(rows: Sequence[Mapping[str, object]]) -> str:
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return buffer.getvalue()


def _percent(results: Results) -> str:
    return "" if math.isnan(results.success) else f"{results.success:.0%}"


def _fraction(results: Results) -> str:
    ok = results.episodes - results.failed
    return f"{results.delivered}/{ok}" if ok else ""


def _table(report: Report) -> list[str]:
    lines = [
        "| | training score | easy delivered | full delivered | held-out delivered | held-out progress | held-out `max_pushers` "
        "| held-out delivery step (median) | gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in report.subjects:
        t, te, tf, h = s.train(), s.train("easy"), s.train("full"), s.held()
        label = f"generation {s.generation} (candidate {s.candidate_id})" if s.kind == "champion" else f"*{s.name}*"
        steps = "" if math.isnan(h.delivery_steps) else f"{h.delivery_steps:.0f}"
        gap = "" if math.isnan(s.gap) else f"{s.gap:+.2f}"
        lines.append(
            f"| {label} | {t.score:.3f} | {_fraction(te)} | {_fraction(tf)} ({_percent(tf)}) | {_fraction(h)} ({_percent(h)}) "
            f"| {h.progress:+.2f} | {h.max_pushers:.1f} | {steps} | {gap} |"
        )
    return lines


def _bands(episodes: Sequence[Episode]) -> list[tuple[str, int, int]]:
    """Held-out success by box–target distance, in thirds of the full difficulty's range (6–8, 8–10, 10–12 units)."""
    edges = [6.0, 8.0, 10.0, math.inf]
    out: list[tuple[str, int, int]] = []
    for lo, hi in zip(edges, edges[1:], strict=False):
        chosen = [e for e in episodes if e.ok and lo <= e.distance < hi]
        name = f"{lo:.0f}–{hi:.0f}" if math.isfinite(hi) else f"{lo:.0f}+"
        out.append((name, sum(e.delivered for e in chosen), len(chosen)))
    below = [e for e in episodes if e.ok and e.distance < edges[0]]
    if below:
        out.insert(0, (f"under {edges[0]:.0f}", sum(e.delivered for e in below), len(below)))
    return out


def _gap_sentence(final: Subject) -> str:
    tf, h = final.train("full").success, final.held().success
    if math.isnan(final.gap):
        return ""
    if tf > 0:
        return f"At the end it is {final.gap:+.2f}: on unseen layouts the champion keeps {h / tf:.0%} of its full-training success."
    return f"At the end it is {final.gap:+.2f}: the champion delivers no full training scenario, so there is nothing to carry over yet."


def _baseline_sentence(final: Subject, baselines: Sequence[Subject]) -> str:
    """How the final champion compares with the senses-only controller, the stronger baseline: a hand-written policy on the
    same senses, so matching it is a result in itself, and beating it is evolution finding more than its author did."""
    senses = next((b for b in baselines if b.kind == "senses only"), None)
    if senses is None:
        return ""
    fh, sh = final.held(), senses.held()
    if fh.delivered > sh.delivered:
        verdict = f"beats the hand-written controller ({fh.delivered} against {sh.delivered})"
    elif fh.delivered == sh.delivered:
        verdict = f"matches the hand-written controller ({fh.delivered} each)"
    else:
        verdict = f"comes close to the hand-written controller ({fh.delivered} against {sh.delivered})"
    pace = ""
    if not (math.isnan(fh.delivery_steps) or math.isnan(sh.delivery_steps)):
        pace = (
            f", delivering at a median step of {fh.delivery_steps:.0f} against the controller's {sh.delivery_steps:.0f}"
            if fh.delivery_steps > sh.delivery_steps
            else f", and delivers sooner (a median step of {fh.delivery_steps:.0f} against {sh.delivery_steps:.0f})"
        )
    return f"From random brains that deliver nothing, evolution reached a team that {verdict}{pace}."


def _findings(report: Report, exploit: str | None) -> list[str]:
    final, first = report.final, report.champions[0]
    fh, ft = final.held(), final.train("full")
    baselines = [s for s in report.subjects if s.kind != "champion"]
    paragraphs: list[str] = []
    paragraphs.append(
        f"**Unseen layouts.** The final champion (generation {final.generation}, candidate {final.candidate_id}) delivers "
        f"**{fh.delivered} of {fh.episodes - fh.failed} held-out scenarios ({_percent(fh)})**, moving the box {fh.progress:.0%} "
        f"of the way on average, against {ft.delivered} of {ft.episodes - ft.failed} full training scenarios ({_percent(ft)}). "
        f"The first generation's champion delivers {first.held().delivered}."
    )
    gaps = ", ".join(f"{s.generation}: {s.gap:+.2f}" for s in report.champions if not math.isnan(s.gap))
    paragraphs.append(
        f"**The generalisation gap** (full-training success minus held-out success, both full difficulty), by generation: {gaps}. "
        + _gap_sentence(final)
    )
    paragraphs.append(
        "**Baselines** on the same held-out scenarios: "
        + "; ".join(
            f"the {b.name} {'deliver' if b.kind == 'random brains' else 'delivers'} {b.held().delivered} of {b.held().episodes} "
            f"({_percent(b.held())}), progress {b.held().progress:+.2f}"
            for b in baselines
        )
        + ". "
        + _baseline_sentence(final, baselines)
    )
    bands = _bands(final.held_out)
    short = [e.progress for e in final.held_out if e.ok and not e.delivered]
    if bands:
        partway = f" The undelivered ones end {float(np.mean(short)):.0%} of the way on average." if short else ""
        paragraphs.append(
            "**Distance.** The final champion's held-out deliveries by the box's starting distance to the target: "
            + ", ".join(f"{name} units {d}/{n}" + (f" ({d / n:.0%})" if n else "") for name, d, n in bands)
            + f".{partway} Success that falls with distance, with the rest partway, points at the {report.config.max_steps}-step "
            "limit as well as the skill."
        )
    stuck = [e.scenario_id for e in final.held_out if e.ok and e.progress < 0.05]
    if stuck:
        paragraphs.append(f"**Never moved** (progress under 5 %): {', '.join(stuck)}. Watch them to see why (`--showcase worst-held`).")
    failed = sum(s.held().failed for s in report.subjects)
    if failed:
        paragraphs.append(f"**Failed episodes:** {failed} held-out episodes failed (a non-finite state) and are left out of the means.")
    r = report.recommendation(exploit)
    borderline = ""
    if r.case in ("limited", "learned the maps") and abs(final.gap - GAP_THRESHOLD) <= BORDERLINE:
        other = "more full training scenarios" if r.case == "limited" else "what stops it besides the maps"
        borderline = f" The gap is at the threshold (about {GAP_THRESHOLD:.2f}), so {other} is indicated too."
    paragraphs.append(f"**Recommendation: {r.next_milestone}.** {r.reason[0].upper()}{r.reason[1:]} (case: {r.case}).{borderline}")
    return [p for p in paragraphs if p]


def markdown(report: Report, exploit: str | None = None) -> str:
    c = report.config
    held = report.final.held()
    lines = [
        f"# Held-out report: {c.name}",
        "",
        f"Run `{report.run_dir}`. {c.team.agents} agents, {c.scenarios.easy} easy + {c.scenarios.full} full training "
        f"scenarios, {c.max_steps} steps, {c.budget.generations} generations. Held-out set: {held.episodes} full-difficulty "
        f"scenarios from seed {c.scenarios.held_out_seed}, never seen in training. Evaluated with "
        f"{report.backend.name} {report.backend.device} {report.backend.precision}.",
        "",
        "Each champion is the run's best up to its generation (log-spaced milestones, unchanged champions skipped). "
        "Training numbers are the recording's; held-out ones are Auxein's `evaluate_held_out`. The gap is full-training "
        "success minus held-out success.",
        "",
        *_table(report),
        "",
        *[line for p in _findings(report, exploit) for line in (p, "")],
    ]
    for note in report.notes:
        lines += [f"Note: {note}.", ""]
    return "\n".join(lines)


def write_report(report: Report, exploit: str | None = None) -> Path:
    """Writes `report.md`, `held_out.csv`, `held_out_episodes.csv` and `held_out.json` under `DIR/report/`; returns it."""
    target = report.run_dir / REPORT
    target.mkdir(parents=True, exist_ok=True)
    (target / SUMMARY_FILE).write_text(_csv(_summary_rows(report)))
    (target / EPISODES_FILE).write_text(_csv(_episode_rows(report)))
    (target / JSON_FILE).write_text(json.dumps(report.auxein.to_json(), indent=2) + "\n")
    (target / MARKDOWN_FILE).write_text(markdown(report, exploit))
    return target


def read_episodes(run_dir: str | Path) -> list[dict[str, str]] | None:
    """The rows of a written report's `held_out_episodes.csv`, or None if the report has not been run."""
    path = Path(run_dir) / REPORT / EPISODES_FILE
    if not path.exists():
        return None
    with open(path) as file:
        return list(csv.DictReader(file))
