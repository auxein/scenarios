"""Offline rendering helpers shared by the scenarios: matplotlib figures to MP4 through ffmpeg, and PNG galleries.

- **ffmpeg without a system install.** The `imageio-ffmpeg` wheel ships a static ffmpeg binary (macOS arm64, Linux x86-64), and
  matplotlib's `FFMpegWriter` is pointed at it (`animation.ffmpeg_path`), so `uv sync` is enough to render on a Mac and in CI.
- **No pyplot.** Figures are built with matplotlib's object API on an Agg canvas: no global figure state, no GUI backend, and
  nothing that depends on the machine's display, so rendering works the same in a terminal, in CI and in a worker.
- **Time.** A video shows simulated time sped up by `speedup` at `fps` frames per second; `frame_records` picks, for each frame,
  the recorded state nearest to that frame's simulated time, so a trajectory can be recorded every step or every k-th.
- **Pieces for montages** (M3): `TitleCard`, a full-figure card with a label, a headline and a lead, shown between clips;
  `CurveInset`, a small curve (fitness over generations) with a marker that moves to the current point; `parse_size` for
  `--size 1920x1080`; and `drawn_text`, the text a figure shows, for tests that check labels without comparing pixels.
"""

# matplotlib's stubs type many methods' **kwargs as unknown (set_xticks, plot, text, savefig, grab_frame...), and imageio-ffmpeg
# ships none: strict mode would flag every call. Only that check is off here; everything else stays strict.
# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false

import bisect
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

import imageio_ffmpeg
import matplotlib
import numpy as np
from auxein.backend import HostArray
from matplotlib.animation import FFMpegWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator

from auxein_scenarios.common import theme

Frame = TypeVar("Frame")
"""What `write_mp4` hands `update` per frame: a record index, or anything a montage needs."""

__all__ = ["CurveInset", "TitleCard", "drawn_text", "figure", "frame_records", "grid_shape", "parse_size", "save_png", "write_mp4"]


def figure(width: float, height: float, dpi: int = 100) -> Figure:
    """A figure of `width × height` inches on its own Agg canvas, outside pyplot."""
    fig = Figure(figsize=(width, height), dpi=dpi)
    FigureCanvasAgg(fig)
    return fig


def grid_shape(count: int, columns: int | None = None) -> tuple[int, int]:
    """Rows and columns for `count` panels: `columns` if given, else as square as possible (wider than tall)."""
    if count < 1:
        raise ValueError(f"need at least one panel, got {count}")
    cols = columns if columns is not None else math.ceil(math.sqrt(count))
    cols = max(1, min(cols, count))
    return math.ceil(count / cols), cols


def frame_records(record_steps: HostArray, dt: float, fps: float, speedup: float, hold: float = 1.0) -> list[int]:
    """For each video frame, the index of the recorded state to show.

    `record_steps` are the episode steps of the records (increasing), `dt` the seconds of simulated time per step. A frame shows
    `speedup / fps` seconds of simulated time after the previous one; the last record is held for `hold` seconds of video."""
    if fps <= 0 or speedup <= 0:
        raise ValueError(f"fps and speedup must be positive, got {fps} and {speedup}")
    steps: list[float] = np.asarray(record_steps, dtype=np.float64).tolist()
    duration = (steps[-1] - steps[0]) * dt
    frames = int(math.floor(duration * fps / speedup)) + 1
    last = len(steps) - 1
    nearest: list[int] = []
    for frame in range(frames):  # a frame at a time: a few hundred, on the host
        want = steps[0] + frame * speedup / fps / dt
        after = min(bisect.bisect_left(steps, want), last)
        before = max(after - 1, 0)
        nearest.append(before if abs(steps[before] - want) <= abs(steps[after] - want) else after)
    return nearest + [last] * int(round(hold * fps))


def write_mp4(fig: Figure, update: Callable[[Frame], None], frames: Sequence[Frame], path: str | Path, fps: float = 30.0) -> Path:
    """Writes an MP4 (H.264) of `fig`: for each value in `frames`, `update(value)` changes the figure, which is grabbed.

    The bitrate is left to ffmpeg's quality-based default; `yuv420p` keeps the file playable everywhere (browsers, QuickTime)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # H.264 with yuv420p needs even pixel dimensions: round the figure down to them
    width, height = (2 * math.floor(inches * fig.dpi / 2) / fig.dpi for inches in fig.get_size_inches())
    fig.set_size_inches(width, height)
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = FFMpegWriter(fps=round(fps), codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    with writer.saving(fig, str(target), dpi=fig.dpi):
        for value in frames:
            update(value)
            writer.grab_frame()
    return target


def save_png(fig: Figure, path: str | Path) -> Path:
    """Saves `fig` as a PNG, creating the directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(target), format="png")
    return target


def parse_size(text: str) -> tuple[int, int]:
    """`"1920x1080"` as `(1920, 1080)` pixels: both even (H.264 with yuv420p needs it) and at least 320 × 180."""
    width, sep, height = text.lower().partition("x")
    if not (sep and width.isdigit() and height.isdigit()):
        raise ValueError(f"a size is WIDTHxHEIGHT in pixels, e.g. 1920x1080; got {text!r}")
    w, h = int(width), int(height)
    if w < 320 or h < 180 or w % 2 or h % 2:
        raise ValueError(f"a size must be even and at least 320x180, got {text!r}")
    return w, h


def drawn_text(fig: Figure) -> list[str]:
    """Every visible, non-empty text of `fig` (its own and its axes'), in drawing order: what a viewer can read in a frame."""
    texts: list[Text] = [*fig.texts]
    for ax in fig.axes:
        if ax.get_visible():
            texts += [*ax.texts, ax.title, ax.xaxis.label, ax.yaxis.label]
            texts += [t for t in ax.get_xticklabels() + ax.get_yticklabels()]
    return [t.get_text() for t in texts if t.get_visible() and t.get_text() and (t.axes is None or t.axes.get_visible())]


class TitleCard:
    """A card over the whole figure, between clips: a tracked label, a large light headline and a muted lead, centred on the
    background. Hidden until `show`; `hide` reveals the figure under it again."""

    def __init__(self, fig: Figure, scale: float = 1.0) -> None:
        self.ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), zorder=50)
        theme.style_axes(self.ax)
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)

        def centred(y: float, color: str, role: theme.Role, size: float) -> Text:
            return self.ax.text(
                0.5, y, "", ha="center", va="center", transform=self.ax.transAxes, color=color, fontproperties=theme.font(role, size)
            )

        self.label = centred(0.62, theme.SILVER, "label", 12 * scale)
        self.headline = centred(0.52, theme.TEXT, "headline", 54 * scale)
        self.lead = centred(0.41, theme.MUTED, "lead", 16 * scale)
        self.ax.set_visible(False)

    def show(self, label: str, headline: str, lead: str = "") -> None:
        self.label.set_text(theme.tracked(label) if label else "")
        self.headline.set_text(headline)
        self.lead.set_text(lead)
        self.ax.set_visible(True)

    def hide(self) -> None:
        self.ax.set_visible(False)


class CurveInset:
    """A small chart in the theme, at `rect` (figure fractions): a bright curve (`y` over `x`), an optional faint one, a few
    quiet marks (e.g. every milestone), and a marker with a vertical rule that `mark` moves to the current point."""

    def __init__(
        self,
        fig: Figure,
        rect: tuple[float, float, float, float],
        x: Sequence[float],
        y: Sequence[float],
        *,
        faint: Sequence[float] | None = None,
        marks: Sequence[float] = (),
        label: str = "",
        xlabel: str = "",
        scale: float = 1.0,
        log_x: bool = False,
    ) -> None:
        self.ax = ax = fig.add_axes(rect)
        self.x, self.y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        ax.set_facecolor(theme.BACKGROUND)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(theme.LINE)
        ax.tick_params(colors=theme.SUBTLE, labelsize=7 * scale, length=2)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(theme.font("readout", 7 * scale))
        if log_x:
            ax.set_xscale("log")
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # generations are whole numbers
        if faint is not None:
            ax.plot(self.x, np.asarray(faint, dtype=np.float64), color=theme.rgba(theme.STEEL, 0.6), linewidth=0.8 * scale)
        ax.plot(self.x, self.y, color=theme.SILVER, linewidth=1.2 * scale)
        if marks:
            at = np.asarray(marks, dtype=np.float64)
            ax.scatter(at, np.interp(at, self.x, self.y), s=10 * scale, color=theme.STEEL, linewidths=0, zorder=3)
        self.rule = ax.axvline(float(self.x[0]), color=theme.rgba(theme.TEXT, 0.35), linewidth=0.8 * scale)
        (self.marker,) = ax.plot([self.x[0]], [self.y[0]], "o", color=theme.TEXT, markersize=5 * scale, zorder=4)
        if label:
            ax.text(0.0, 1.06, theme.tracked(label), transform=ax.transAxes, va="bottom", ha="left", color=theme.SILVER,
                    fontproperties=theme.font("label", 7.5 * scale))  # fmt: skip
        if xlabel:
            ax.set_xlabel(xlabel, color=theme.SUBTLE, fontproperties=theme.font("readout", 7 * scale), labelpad=2)

    def mark(self, x: float) -> None:
        """Moves the marker and the rule to `x`, on the curve."""
        y = float(np.interp(x, self.x, self.y))
        self.marker.set_data([x], [y])
        self.rule.set_xdata([x, x])
