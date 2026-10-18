"""Comparing runs on one held-out set (design doc §7.1, M2b): `box-pushing compare DIR... [--held-out DIR_OR_CONFIG]`.

A run's own report judges it on its own held-out set; two runs with different scenario generators, or different strategies
(M4), have different held-out sets, so their reports do not compare. `compare` puts them on **one common held-out set**:

- **The common set** is the held-out set of a reference: a run directory or a config file (`--held-out`), by default the newest
  of the runs (by when it started). Its config fixes the generator version, the seed, the count and the team size.
- **Each run's champion keeps its own world, sensing and brain** (and aggregator), rebuilt from its own config, through the
  report's `champions_on` (Auxein's `evaluate_held_out`); only the scenarios are the common ones. A run whose layouts cannot
  be the reference's (another team size, arena, box or agent size) is refused with a message saying why, and so is one whose
  training set shares a layout with the common set.
- **The final champion** of each run, or every milestone's with `milestones=True`; the two baselines of the report, in the
  reference's world, on the common set and the reference's training set.
- **Output**, under the reference run's `report/` (the newest run's when the reference is a config file) or `out`:
  `compare.md` and `compare.csv`, per champion: success on the common set, mean progress, mean `max_pushers`, the median
  delivery step, success by the box's starting distance (6–8, 8–10, 10–12 units), and its own training numbers. Nothing is
  written into any recording, and the same command gives the same files.
"""

import csv
import io
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from auxein.environments import ScenarioSet
from auxein.recording.reader import open_run

from auxein_scenarios.box_pushing import report, scenarios, train
from auxein_scenarios.box_pushing.config import TrainingConfig, load_config
from auxein_scenarios.box_pushing.milestones import find_milestones
from auxein_scenarios.box_pushing.report import Subject

__all__ = ["CSV_FILE", "MARKDOWN_FILE", "Compared", "CompareError", "Comparison", "Reference", "build_comparison", "write_comparison"]

MARKDOWN_FILE = "compare.md"
CSV_FILE = "compare.csv"
BANDS = (("6-8", 6.0, 8.0), ("8-10", 8.0, 10.0), ("10-12", 10.0, math.inf))
"""The distance bands of the CSV (units from the box's start to the target); the last takes anything from 10 up."""

Print = Callable[[str], None]


class CompareError(ValueError):
    """Runs that cannot be compared on the chosen held-out set."""


@dataclass(frozen=True)
class Reference:
    """Where the common held-out set comes from: a run directory or a config file, and its config."""

    source: Path
    config: TrainingConfig
    run_dir: Path | None


@dataclass(frozen=True)
class Compared:
    """One run's champions on the common set."""

    run_dir: Path
    config: TrainingConfig
    champions: tuple[Subject, ...]


@dataclass(frozen=True)
class Comparison:
    reference: Reference
    held: ScenarioSet
    runs: tuple[Compared, ...]
    baselines: tuple[Subject, ...]


def _started(run: Path) -> datetime:
    with open_run(run / train.RECORDING) as reader:
        return datetime.fromisoformat(str(reader.metadata["started_at"]))


def _reference(runs: Sequence[Path], held_out: str | Path | None) -> Reference:
    if held_out is None:
        newest = max(runs, key=_started)
        return Reference(newest, load_config(newest / train.CONFIG_FILE), newest)
    source = Path(held_out)
    if source.is_dir():
        if not (source / train.CONFIG_FILE).exists():
            raise CompareError(f"{source} is not a run directory (no {train.CONFIG_FILE}); give a run or a config file")
        return Reference(source, load_config(source / train.CONFIG_FILE), source)
    if not source.exists():
        raise CompareError(f"--held-out {source}: no such run directory or config file")
    return Reference(source, load_config(source), None)


def _check(run: Path, config: TrainingConfig, reference: TrainingConfig, held: ScenarioSet) -> None:
    """Refuses a run whose brains cannot be judged fairly on the reference's layouts (module docstring)."""
    reasons: list[str] = []
    if config.team.agents != reference.team.agents:
        reasons.append(f"its team has {config.team.agents} agents, the held-out layouts {reference.team.agents}")
    for part in ("arena", "box", "agent"):
        if getattr(config.world, part) != getattr(reference.world, part):
            reasons.append(f"its world.{part} differs from the reference's, which the layouts were generated for")
    if reasons:
        raise CompareError(f"cannot compare {run} on this held-out set: " + "; ".join(reasons))
    try:
        scenarios.check_disjoint(train.training_set(config), held)
    except ValueError as error:
        raise CompareError(f"cannot compare {run}: its training set shares a layout with the held-out set ({error})") from None


def build_comparison(
    run_dirs: Sequence[str | Path], *, held_out: str | Path | None = None, milestones: bool = False, out: Print = print
) -> Comparison:
    """Evaluates the runs' champions on the common held-out set, and the baselines (module docstring). Writes nothing."""
    runs = [Path(r) for r in run_dirs]
    if not runs:
        raise CompareError("give at least one run directory")
    for run in runs:
        if not (run / train.CONFIG_FILE).exists() or not (run / train.RECORDING).exists():
            raise CompareError(f"{run} is not a box-pushing run (it needs {train.CONFIG_FILE} and {train.RECORDING}/)")
    reference = _reference(runs, held_out)
    held = train.held_out_set(reference.config)
    configs = [load_config(run / train.CONFIG_FILE) for run in runs]
    for run, config in zip(runs, configs, strict=True):
        _check(run, config, reference.config, held)
    out(
        f"common held-out set: {len(held)} scenarios of generator {reference.config.scenarios.generator}, seed "
        f"{reference.config.scenarios.held_out_seed} (from {reference.source})"
    )
    compared: list[Compared] = []
    for run, config in zip(runs, configs, strict=True):
        chosen = find_milestones(run / train.RECORDING)
        chosen = chosen if milestones else chosen[-1:]
        champions, _ = report.champions_on(run, chosen, held, out=out)
        compared.append(Compared(run, config, tuple(champions)))
    backend, _ = train.resolve_backend(reference.config)
    baselines = report.baselines_on(reference.config, train.training_set(reference.config), held, backend, out)
    return Comparison(reference, held, tuple(compared), tuple(baselines))


# --- writing it ---


def _band(subject: Subject, lo: float, hi: float) -> tuple[int, int]:
    chosen = [e for e in subject.held_out if e.ok and lo <= e.distance < hi]
    return sum(e.delivered for e in chosen), len(chosen)


def _rows(comparison: Comparison) -> list[dict[str, object]]:
    def row(run: str, generator: object, subject: Subject) -> dict[str, object]:
        h, t = subject.held(), subject.train()
        values: dict[str, object] = {
            "run": run,
            "generator": generator,
            "kind": subject.kind,
            "generation": "" if subject.generation is None else subject.generation,
            "candidate_id": "" if subject.candidate_id is None else subject.candidate_id,
            "held_episodes": h.episodes - h.failed,
            "held_delivered": h.delivered,
            "held_success": f"{h.success:.6f}",
            "held_progress": f"{h.progress:.6f}",
            "held_max_pushers": f"{h.max_pushers:.6f}",
            "held_delivery_steps_median": "" if math.isnan(h.delivery_steps) else f"{h.delivery_steps:.1f}",
        }
        for name, lo, hi in BANDS:
            delivered, total = _band(subject, lo, hi)
            values[f"held_{name}_delivered"], values[f"held_{name}_episodes"] = delivered, total
        values |= {
            "train_score": f"{t.score:.6f}",
            "train_easy_success": "" if math.isnan(subject.train("easy").success) else f"{subject.train('easy').success:.6f}",
            "train_full_success": "" if math.isnan(subject.train("full").success) else f"{subject.train('full').success:.6f}",
        }
        return values

    out = [row(c.run_dir.name, c.config.scenarios.generator, s) for c in comparison.runs for s in c.champions]
    out += [row("baseline", comparison.reference.config.scenarios.generator, b) for b in comparison.baselines]
    return out


def _markdown(comparison: Comparison) -> str:
    ref = comparison.reference.config.scenarios
    lines = [
        "# Comparison on a common held-out set",
        "",
        f"{len(comparison.held)} full-difficulty scenarios of generator {ref.generator} from seed {ref.held_out_seed} (the held-out "
        f"set of `{comparison.reference.source}`), never seen in training by any run. Each run's champion is judged with its own "
        "world, sensing and brain; only the scenarios are shared.",
        "",
        "| run | champion | delivered | progress | `max_pushers` | delivery step (median) | 6–8 units | 8–10 | 10–12 | "
        "its own training: full delivered |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in _rows(comparison):
        who = f"generation {r['generation']} (candidate {r['candidate_id']})" if r["kind"] == "champion" else f"*{r['kind']}*"
        bands = [f"{r[f'held_{n}_delivered']}/{r[f'held_{n}_episodes']}" for n, _, _ in BANDS]
        success = float(str(r["held_success"]))
        full = str(r["train_full_success"])
        lines.append(
            f"| {r['run']} (generator {r['generator']}) | {who} | {r['held_delivered']}/{r['held_episodes']} ({success:.0%}) "
            f"| {float(str(r['held_progress'])):+.2f} | {float(str(r['held_max_pushers'])):.1f} | {r['held_delivery_steps_median']} "
            f"| {' | '.join(bands)} | {f'{float(full):.0%}' if full else ''} |"
        )
    return "\n".join(lines) + "\n"


def write_comparison(comparison: Comparison, out_dir: str | Path | None = None) -> Path:
    """Writes `compare.md` and `compare.csv` (module docstring); returns the directory."""
    if out_dir is not None:
        target = Path(out_dir)
    else:
        home = comparison.reference.run_dir or max((c.run_dir for c in comparison.runs), key=_started)
        target = home / report.REPORT
    target.mkdir(parents=True, exist_ok=True)
    rows = _rows(comparison)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (target / CSV_FILE).write_text(buffer.getvalue())
    (target / MARKDOWN_FILE).write_text(_markdown(comparison))
    return target
