"""Charts in the scenarios' theme (design doc §7): line charts as PNG and SVG, for the website and talks.

The theme's rule holds for charts too: **no hue**. Series are told apart by brightness and by line style (solid, dashed,
dotted), and each is labelled where it ends, so a chart reads without a legend box. A chart has the header of the clips (a
tracked label, a light headline, a muted lead) and quiet axes: hairline spines on two sides, small readouts.

`line_chart` draws one; with `message` instead of series it draws the header and that message where the chart would be (for
a chart whose data does not exist yet, so the file always exists and says why it is empty).
"""

# matplotlib's stubs type many methods' **kwargs as unknown: only that check is off here.
# pyright: reportUnknownMemberType=false

import math
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib.ticker import MaxNLocator

from auxein_scenarios.common import theme
from auxein_scenarios.common.rendering import figure

__all__ = ["Series", "line_chart"]

Style = Literal["solid", "dashed", "dotted"]
_DASHES: dict[Style, object] = {"solid": "-", "dashed": (0, (5, 3)), "dotted": (0, (1, 2.5))}


@dataclass(frozen=True)
class Series:
    """One line: its `label` (written at its end), points, brightness (`color`, a theme colour) and `style`; `markers` puts a
    dot on every point (for a few points, such as milestones)."""

    label: str
    x: Sequence[float]
    y: Sequence[float]
    color: str = theme.SILVER
    style: Style = "solid"
    markers: bool = False


def line_chart(
    path: str | Path,
    series: Sequence[Series],
    *,
    kicker: str = "",
    title: str = "",
    lead: str = "",
    xlabel: str = "",
    ylabel: str = "",
    ylim: tuple[float, float] | None = None,
    percent: bool = False,
    log_x: bool = False,
    message: str = "",
    size: tuple[float, float] = (9.6, 5.4),
    dpi: int = 200,
) -> tuple[Path, Path]:
    """Draws the chart and saves it as `path` with the suffixes `.png` and `.svg`; returns both paths. `percent` shows the y
    axis as percentages (values in [0, 1])."""
    width, height = size
    fig = figure(width, height, dpi)
    fig.set_facecolor(theme.BACKGROUND)
    left = 0.5 / width
    y = 1.0 - 0.32 / height
    if kicker:
        fig.text(left, y, theme.tracked(kicker), color=theme.SILVER, fontproperties=theme.font("label", 8), va="top")
        y -= 0.3 / height
    if title:
        fig.text(left, y, title, color=theme.TEXT, fontproperties=theme.font("headline", 20), va="top")
        y -= 0.45 / height
    for line in textwrap.wrap(lead, width=int(width * 13)) if lead else []:
        fig.text(left, y, line, color=theme.MUTED, fontproperties=theme.font("lead", 10), va="top")
        y -= 0.2 / height
    top = y - 0.25 / height
    ax = fig.add_axes((left + 0.35 / width, 0.55 / height, 1.0 - left - 1.9 / width, top - 0.55 / height))
    ax.set_facecolor(theme.BACKGROUND)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.LINE)
    ax.tick_params(colors=theme.SUBTLE, length=3)
    if message:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=theme.MUTED, fontproperties=theme.font("lead", 12)
        )
    else:
        if log_x:
            ax.set_xscale("log")
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for s in series:
            xs, ys = np.asarray(s.x, dtype=np.float64), np.asarray(s.y, dtype=np.float64)
            ax.plot(xs, ys, color=s.color, linestyle=_DASHES[s.style], linewidth=1.4, marker="o" if s.markers else None, markersize=3.5)
            finite = np.nonzero(np.isfinite(ys))[0]
            if finite.size:
                end = int(finite[-1])
                ax.annotate(
                    s.label, (float(xs[end]), float(ys[end])), xytext=(6, 0), textcoords="offset points", va="center",
                    color=s.color, fontproperties=theme.font("readout", 8.5), annotation_clip=False,
                )  # fmt: skip
        if ylim is not None:
            ax.set_ylim(*ylim)
        if percent:
            ticks = ax.get_yticks()
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:.0%}" if math.isfinite(t) else "" for t in ticks])
            if ylim is not None:
                ax.set_ylim(*ylim)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(theme.font("readout", 8))
    if xlabel:
        ax.set_xlabel(xlabel, color=theme.SUBTLE, fontproperties=theme.font("readout", 9))
    if ylabel:
        ax.set_ylabel(ylabel, color=theme.SUBTLE, fontproperties=theme.font("readout", 9))
    stem = Path(path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    png, svg = stem.with_suffix(".png"), stem.with_suffix(".svg")
    fig.savefig(str(png), format="png", facecolor=theme.BACKGROUND)
    fig.savefig(str(svg), format="svg", facecolor=theme.BACKGROUND)
    return png, svg
