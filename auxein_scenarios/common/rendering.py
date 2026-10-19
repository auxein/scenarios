"""Offline rendering helpers shared by the scenarios: matplotlib figures to MP4 through ffmpeg.

- **ffmpeg without a system install.** The `imageio-ffmpeg` wheel ships a static ffmpeg binary (macOS arm64, Linux x86-64), and
  matplotlib's `FFMpegWriter` is pointed at it (`animation.ffmpeg_path`), so `uv sync` is enough to render on a Mac and in CI.
- **No pyplot.** Figures are built with matplotlib's object API on an Agg canvas: no global figure state, no GUI backend, and
  nothing that depends on the machine's display, so rendering works the same in a terminal, in CI and in a worker.
- **Time.** A video shows simulated time sped up by `speedup` at `fps` frames per second; `frame_records` picks, for each frame,
  the recorded state nearest to that frame's simulated time, so a trajectory can be recorded every step or every k-th.
- **`drawn_text`**: the text a figure shows, for tests that check labels without comparing pixels.
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

Frame = TypeVar("Frame")
"""What `write_mp4` hands `update` per frame: a record index, or anything a montage needs."""

__all__ = ["drawn_text", "figure", "frame_records", "grid_shape", "write_mp4"]


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


def drawn_text(fig: Figure) -> list[str]:
    """Every visible, non-empty text of `fig` (its own and its axes'), in drawing order: what a viewer can read in a frame."""
    texts: list[Text] = [*fig.texts]
    for ax in fig.axes:
        if ax.get_visible():
            texts += [*ax.texts, ax.title, ax.xaxis.label, ax.yaxis.label]
            texts += [t for t in ax.get_xticklabels() + ax.get_yticklabels()]
    return [t.get_text() for t in texts if t.get_visible() and t.get_text() and (t.axes is None or t.axes.get_visible())]
