"""A run's charts, PNG and SVG in the theme, into `charts/` of the run directory: what `train` draws at the end of a run.

- **fitness**: the score on the training set over generations: the best so far, and each generation's mean and median;
- **delivery**: the share of training scenarios delivered: by the best brain so far, and on average by each generation;
- **cooperation**: the most agents touching the box at once, averaged over the training scenarios, for the best brain so far
  and on average: a team that cannot move the box alone must learn to gather;
- **generalisation**: full-difficulty success per snapshot brain, on the training set and on the held-out set it never saw
  (from `held_out.csv`): the gap between the two lines is what the training maps added.

The first three read `progress.csv`, the last `held_out.csv`; both are rebuilt from the recording, so the charts can be drawn
again at any time.
"""

import csv
import math
from collections.abc import Callable
from pathlib import Path

from auxein_scenarios.box_pushing.config import TrainingConfig
from auxein_scenarios.common import theme
from auxein_scenarios.common.charts import Series, line_chart

__all__ = ["NAMES", "draw_charts"]

NAMES = ("fitness", "delivery", "cooperation", "generalisation")


def _rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with open(path) as file:
        raw = list(csv.DictReader(file))

    def number(value: str | None) -> float:
        try:
            return float(value) if value not in ("", None) else math.nan
        except ValueError:
            return math.nan

    return [{key: number(value) for key, value in row.items()} for row in raw]


def draw_charts(
    run_dir: Path, config: TrainingConfig, progress_csv: Path, held_out_csv: Path, out: Callable[[str], None] = print
) -> list[Path]:
    """The four charts (module docstring), PNG and SVG, into `run_dir/charts/`; returns the files written."""
    rows = _rows(progress_csv)
    g = [r["generation"] for r in rows]
    target = run_dir / "charts"
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
            Series("best so far", g, column("champion_success_rate"), theme.TEXT),
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
            Series("best so far", g, column("champion_max_pushers"), theme.TEXT),
            Series("generation mean", g, column("max_pushers_mean"), theme.SILVER, "dashed"),
        ],
        kicker=kicker, title="Cooperation over generations",
        lead="The most agents touching the box at once, averaged over the training scenarios: by the best brain so far, and on "
        f"average by each generation's children. One agent cannot move the box; {config.world.friction.k} can.",
        xlabel="generation", ylabel="agents at the box (max, mean over scenarios)",
    )  # fmt: skip
    snapshots = _rows(held_out_csv)
    title, lead = (
        "Held-out against training",
        "Full-difficulty scenarios delivered by each snapshot's brain: the training ones, and unseen ones.",
    )
    if snapshots:
        written += line_chart(
            target / "generalisation",
            [
                Series("training (full)", [r["generation"] for r in snapshots], [r["train_full_success"] for r in snapshots],
                       theme.SILVER, "dashed", True),
                Series("held-out (unseen)", [r["generation"] for r in snapshots], [r["held_success"] for r in snapshots],
                       theme.TEXT, "solid", True),
            ],
            kicker=kicker, title=title, lead=lead, xlabel="generation (snapshots)", ylabel="full scenarios delivered",
            ylim=(0.0, 1.0), percent=True, log_x=True,
        )  # fmt: skip
    else:
        message = "No held-out results yet: the run writes them when it ends."
        written += line_chart(target / "generalisation", [], kicker=kicker, title=title, lead=lead, message=message)
    for path in written:
        if path.suffix == ".png":
            out(str(path))
    return written
