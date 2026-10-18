"""The demo's material from a run (design doc §7): milestone clips, the montage, the charts, and `showcase`, which makes all
of them after the held-out report.

Everything is rebuilt from the run directory (its `config.toml` and recording) and written under it, never into the
recording: `DIR/clips/milestones/gen-<k>.mp4`, `DIR/clips/montage-<layout>.mp4`, `DIR/charts/*.png|svg` and `DIR/report/`.
Every command uses the same milestones (`milestones.py`), and clips replay the champions **with the run's backend, precision
and device** through the environment's step loop (`train.replay`, the path of `watch`).

- **Milestone clips:** one clip per milestone, its champion on the same showcase scenarios at every milestone, so they
  compare; each panel's footer says plainly whether it is a training scenario or a held-out one never seen in training, and
  how its episode ended.
- **The montage:** `sequence` (the milestones one after another, each opened by a title card, with a fitness curve whose
  marker moves) or `grid` (three or four milestones side by side on one scenario, simultaneously).
- **Charts:** fitness, delivery and cooperation over generations (from `progress.csv`, itself rebuilt from the recording), and
  held-out against training success per milestone (from the report; without one, the chart says so).
- **Showcase scenarios are not cherry-picked by default** (the first easy and full training scenarios and the first two
  held-out ones). `best-held` and `worst-held` pick the final champion's best and worst held-out scenarios (by progress, a
  delivery first), so a website can show both honestly; the report always gives the full held-out numbers.
"""

import csv
import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from auxein.random import RandomStream
from auxein.recording.reader import open_run

from auxein_scenarios.box_pushing import report as held_out_report
from auxein_scenarios.box_pushing import train
from auxein_scenarios.box_pushing.config import TrainingConfig, load_config
from auxein_scenarios.box_pushing.environment import BoxPushingWorld, brain
from auxein_scenarios.box_pushing.milestones import Milestone, find_milestones
from auxein_scenarios.box_pushing.render import Curve, Segment, render_clip, render_grid_montage, render_sequence_montage
from auxein_scenarios.box_pushing.train import Showcase
from auxein_scenarios.common import theme
from auxein_scenarios.common.charts import Series, line_chart

__all__ = [
    "CHARTS",
    "CLIPS",
    "MILESTONE_CLIPS",
    "Layout",
    "charts",
    "grid_milestones",
    "held_extremes",
    "milestone_clips",
    "montage",
    "resolve_showcase",
    "showcase",
]

CLIPS = "clips"
MILESTONE_CLIPS = "milestones"
CHARTS = "charts"

Layout = Literal["sequence", "grid"]
Print = Callable[[str], None]


# --- showcase scenarios ---


def held_extremes(run_dir: str | Path, out: Print = print) -> tuple[Showcase, Showcase]:
    """The final champion's best and worst held-out scenarios: by progress, a delivered episode before any that is not (a
    delivery stops at the target zone's edge, so its progress is below 1 and could tie with a near miss); ties go to the
    lower index. From the report's `held_out_episodes.csv` when it has the final champion, else evaluated here."""
    run = Path(run_dir)
    config = load_config(run / train.CONFIG_FILE)
    final = find_milestones(run / train.RECORDING)[-1]
    rows = held_out_report.read_episodes(run)
    mine = [r for r in rows or [] if r["kind"] == "champion" and r["candidate_id"] == str(final.candidate_id)]
    if mine:
        ranked = [(int(r["delivered"]), float(r["progress"] or "nan"), int(r["scenario_id"].split("-")[-1])) for r in mine]
    else:
        out("best-held / worst-held: no report yet, so the final champion is evaluated on the held-out set here")
        ranked = _held_progress(run, config, final)
    ranked = [(d, p if math.isfinite(p) else -math.inf, i) for d, p, i in ranked]
    best = max(ranked, key=lambda r: (r[0], r[1], -r[2]))
    worst = min(ranked, key=lambda r: (r[0], r[1], r[2]))
    return Showcase("held", best[2]), Showcase("held", worst[2])


def _held_progress(run: Path, config: TrainingConfig, final: Milestone) -> list[tuple[int, float, int]]:
    backend, _ = train.replay_backend(run)
    world = BoxPushingWorld(config.world, config.sensing)
    policy = brain(config.sensing, config.brain.hidden)
    genome = backend.asarray(np.asarray(final.champion.genome, dtype=np.float64).reshape(1, -1))
    held = held_out_report.held_out_set(config)
    result = world.run_batch(policy.decode_batch(genome), tuple(held), RandomStream(np.random.SeedSequence(0), backend))
    m = {name: np.asarray(backend.to_numpy(v), dtype=np.float64)[0] for name, v in result.measurements.items()}
    failed = {j for _, j in result.failures}
    out: list[tuple[int, float, int]] = []
    for j in range(len(held)):
        d0, d = float(m["initial_distance"][j]), float(m["final_distance"][j])
        progress = math.nan if j in failed else ((d0 - d) / d0 if d0 > 0 else 0.0)
        out.append((int(m["delivered"][j] > 0.5), progress, j))
    return out


def resolve_showcase(run_dir: str | Path, tokens: Sequence[str] | None, out: Print = print) -> list[Showcase]:
    """`--showcase` tokens (`train:<i>`, `held:<i>`, `best-held`, `worst-held`) as showcase scenarios; the default without."""
    run = Path(run_dir)
    if not tokens:
        return train.default_showcase(load_config(run / train.CONFIG_FILE))
    extremes: tuple[Showcase, Showcase] | None = None
    chosen: list[Showcase] = []
    for token in tokens:
        if token in ("best-held", "worst-held"):
            extremes = extremes or held_extremes(run, out)
            chosen.append(extremes[0] if token == "best-held" else extremes[1])
        else:
            chosen.append(Showcase.parse(token))
    return chosen


# --- milestone clips ---


def _delivered_text(m: Milestone, config: TrainingConfig) -> str:
    n = config.scenarios.easy + config.scenarios.full
    delivered = round(float(m.champion.descriptors["success_rate"]) * n)
    return f"delivers {delivered} of {n} training scenarios"


def milestone_clips(
    run_dir: str | Path,
    *,
    generations: Sequence[int] | None = None,
    showcase: Sequence[str] | None = None,
    overlay: bool = False,
    fps: float = 30.0,
    speedup: float = 2.0,
    out: Print = print,
) -> list[Path]:
    """One clip per milestone, its champion on the same showcase scenarios (module docstring), into
    `DIR/clips/milestones/gen-<k>.mp4` (k zero-padded to four digits, so the files sort by generation)."""
    run = Path(run_dir)
    milestones = find_milestones(run / train.RECORDING, generations)
    chosen = resolve_showcase(run, showcase, out)
    played = train.replay(run, [m.champion.genome for m in milestones], chosen, overlay=overlay, out=out)
    config = played.config
    paths: list[Path] = []
    for row, m in enumerate(milestones):
        path = run / CLIPS / MILESTONE_CLIPS / f"gen-{m.generation:04d}{'-overlay' if overlay else ''}.mp4"
        render_clip(
            played.trajectory,
            [dict(s.params) for s in played.scenarios],
            config.world,
            path,
            cells=[(row, j) for j in range(len(played.scenarios))],
            labels=played.labels,
            senses=played.senses,
            sensing=config.sensing,
            kicker=f"Run · {config.name} · milestone {row + 1} of {len(milestones)}",
            title=f"Generation {m.generation}",
            subtitle=f"The best brain so far, candidate {m.candidate_id}: score {m.score:.3f} on the training set, it "
            f"{_delivered_text(m, config)}. The same scenarios at every milestone; {config.team.agents} agents.",
            fps=fps,
            speedup=speedup,
            columns=2,
            footers=played.footers(row),
            records=played.records(row),
        )
        paths.append(path)
        outcomes = "; ".join(f"{s.id}: {o}" for s, (_, o) in zip(played.scenarios, played.footers(row), strict=True))
        out(f"{path}  (generation {m.generation}, score {m.score:.3f}: {outcomes})")
    return paths


# --- the montage ---


def _curve(run: Path) -> tuple[list[float], list[float], list[float]]:
    """Generations, the best score so far and the generation's mean, for the montage's inset."""
    rows = train.progress_rows(run)
    return [r["generation"] for r in rows], [r["best"] for r in rows], [r["score_mean"] for r in rows]


def grid_milestones(milestones: Sequence[Milestone]) -> list[Milestone]:
    """Three or four milestones for the grid: the first, one or two in between, the last (all of them if there are fewer)."""
    if len(milestones) <= 4:
        return list(milestones)
    last = len(milestones) - 1
    return [milestones[i] for i in sorted({0, round(last / 3), round(2 * last / 3), last})]


def montage(
    run_dir: str | Path,
    *,
    layout: Layout = "sequence",
    size: tuple[int, int] = (1920, 1080),
    generations: Sequence[int] | None = None,
    showcase: Sequence[str] | None = None,
    fps: float = 30.0,
    speedup: float | None = None,
    out: Print = print,
) -> Path:
    """The montage (module docstring) into `DIR/clips/montage-<layout>.mp4`. The grid shows one scenario: the first of
    `showcase`, by default the first held-out one (`held:0`, not chosen for how it went)."""
    run = Path(run_dir)
    milestones = find_milestones(run / train.RECORDING, generations)
    x, best, mean = _curve(run)
    path = run / CLIPS / f"montage-{layout}.mp4"
    config = load_config(run / train.CONFIG_FILE)
    if layout == "sequence":
        chosen = resolve_showcase(run, showcase, out)
        played = train.replay(run, [m.champion.genome for m in milestones], chosen, out=out)
        curve = Curve(x, best, faint=mean, marks=[float(m.generation) for m in milestones], label="best score so far  ·  generation mean")
        segments = [
            Segment(
                candidate=row,
                records=played.records(row),
                label=f"Milestone {row + 1} of {len(milestones)}",
                headline=f"Generation {m.generation}",
                lead=f"Score {m.score:.3f}: it {_delivered_text(m, config)}",
                at=float(m.generation),
                footers=played.footers(row),
            )
            for row, m in enumerate(milestones)
        ]
        render_sequence_montage(
            played.trajectory, [dict(s.params) for s in played.scenarios], config.world, path,
            scenarios=list(range(len(played.scenarios))), labels=played.labels, segments=segments, curve=curve,
            kicker=f"Run · {config.name} · {config.team.agents} agents", size=size, fps=fps, speedup=speedup or 4.0,
        )  # fmt: skip
    else:
        picked = grid_milestones(milestones)
        scenario = resolve_showcase(run, showcase, out)[0] if showcase else Showcase("held", 0)
        played = train.replay(run, [m.champion.genome for m in picked], [scenario], out=out)
        curve = Curve(x, best, faint=mean, marks=[float(m.generation) for m in picked], label="best score so far  ·  generation mean")
        span = f"generation {picked[0].generation} to {picked[-1].generation}"
        render_grid_montage(
            played.trajectory, [dict(played.scenarios[0].params)], config.world, path,
            scenario=0, candidates=list(range(len(picked))),
            labels=[f"Generation {m.generation}" for m in picked],
            footers=[(f"{played.scenarios[0].id} · {scenario.note}", train.outcome(played.episode(row, 0))) for row in range(len(picked))],
            curve=curve, kicker=f"Run · {config.name} · before and after",
            title=f"One scenario, {span}",
            lead=f"The best brain at {len(picked)} generations, side by side on {played.scenarios[0].id} "
            f"({scenario.note}), {config.team.agents} agents.",
            records=max(played.records(row) for row in range(len(picked))), size=size, fps=fps, speedup=speedup or 3.0,
        )  # fmt: skip
    out(str(path))
    return path


# --- charts ---


def charts(run_dir: str | Path, out: Print = print) -> list[Path]:
    """The four charts (module docstring), PNG and SVG, into `DIR/charts/`."""
    run = Path(run_dir)
    config = load_config(run / train.CONFIG_FILE)
    rows = train.progress_rows(run)
    g = [r["generation"] for r in rows]
    target = run / CHARTS
    kicker = f"Run · {config.name}"
    written: list[Path] = []

    def column(name: str) -> list[float]:
        return [r.get(name, math.nan) for r in rows]

    written += line_chart(
        target / "fitness",
        [
            Series("best so far", g, column("best"), theme.TEXT),
            Series("generation mean", g, column("score_mean"), theme.SILVER),
            Series("generation median", g, column("score_median"), theme.STEEL, "dashed"),
        ],
        kicker=kicker, title="Fitness over generations",
        lead="The score on the training set: the best brain so far, and the mean and median of each generation's children.",
        xlabel="generation", ylabel="score",
    )  # fmt: skip
    written += line_chart(
        target / "delivery",
        [
            Series("champion", g, column("champion_success_rate"), theme.TEXT),
            Series("generation mean", g, column("success_rate_mean"), theme.SILVER, "dashed"),
        ],
        kicker=kicker, title="Delivery over generations",
        lead=f"The share of the {config.scenarios.easy + config.scenarios.full} training scenarios delivered: by the best brain "
        "so far, and on average by each generation's children.",
        xlabel="generation", ylabel="training scenarios delivered", ylim=(0.0, 1.0), percent=True,
    )  # fmt: skip
    written += line_chart(
        target / "cooperation",
        [
            Series("champion", g, column("champion_max_pushers"), theme.TEXT),
            Series("generation mean", g, column("max_pushers_mean"), theme.SILVER, "dashed"),
        ],
        kicker=kicker, title="Cooperation over generations",
        lead="The most agents touching the box at once, averaged over the training scenarios: by the best brain so far, and on "
        f"average by each generation's children. One agent cannot move the box; {config.world.friction.k} can.",
        xlabel="generation", ylabel="agents at the box (max, mean over scenarios)",
    )  # fmt: skip
    written += _generalisation_chart(run, target, kicker)
    for path in written:
        if path.suffix == ".png":
            out(str(path))
    return written


def _generalisation_chart(run: Path, target: Path, kicker: str) -> tuple[Path, Path]:
    summary = run / held_out_report.REPORT / held_out_report.SUMMARY_FILE
    title, lead = (
        "Held-out against training",
        "Full-difficulty scenarios delivered by each milestone's champion: the training ones, and unseen ones.",
    )
    if not summary.exists():
        message = "No held-out report yet: run  box-pushing report DIR  first, then this chart again."
        return line_chart(target / "generalisation", [], kicker=kicker, title=title, lead=lead, message=message)
    with open(summary) as file:
        champions = [r for r in csv.DictReader(file) if r["kind"] == "champion"]
    g = [float(r["generation"]) for r in champions]
    full = [float(r["train_full_success"] or "nan") for r in champions]
    held = [float(r["held_success"] or "nan") for r in champions]
    return line_chart(
        target / "generalisation",
        [Series("training (full)", g, full, theme.SILVER, "dashed", True), Series("held-out (unseen)", g, held, theme.TEXT, "solid", True)],
        kicker=kicker, title=title, lead=lead, xlabel="generation (milestones)", ylabel="full scenarios delivered",
        ylim=(0.0, 1.0), percent=True, log_x=True,
    )  # fmt: skip


# --- everything ---


def showcase(
    run_dir: str | Path,
    *,
    generations: Sequence[int] | None = None,
    showcase_tokens: Sequence[str] | None = None,
    exploit: str | None = None,
    size: tuple[int, int] = (1920, 1080),
    out: Print = print,
) -> dict[str, float]:
    """The report, the milestone clips, both montages and the charts, in that order (the generalisation chart needs the
    report); returns each step's seconds."""
    run = Path(run_dir)
    with open_run(run / train.RECORDING) as reader:
        if reader.generations() == 0:
            raise train.TrainingError(f"{run} has no generation yet")
    seconds: dict[str, float] = {}

    def timed(name: str, step: Callable[[], object]) -> None:
        start = time.perf_counter()
        out(f"--- {name}")
        step()
        seconds[name] = time.perf_counter() - start
        out(f"    {seconds[name]:.0f} s")

    def report_step() -> None:
        built = held_out_report.build_report(run, generations=generations, out=out)
        out(str(held_out_report.write_report(built, exploit) / held_out_report.MARKDOWN_FILE))

    timed("report", report_step)
    timed("milestone clips", lambda: milestone_clips(run, generations=generations, showcase=showcase_tokens, out=out))
    timed(
        "montage (sequence)", lambda: montage(run, layout="sequence", size=size, generations=generations, showcase=showcase_tokens, out=out)
    )
    timed("montage (grid)", lambda: montage(run, layout="grid", size=size, generations=generations, out=out))
    timed("charts", lambda: charts(run, out=out))
    out(f"done: the report in {run / held_out_report.REPORT}, clips in {run / CLIPS}, charts in {run / CHARTS}")
    return seconds
