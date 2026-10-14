"""Drawing box-pushing worlds (design doc §7): clips of recorded trajectories and galleries of scenario layouts.

Rendering is **offline and draws a recorded trajectory** (`world.TrajectoryRecorder`), so a clip shows exactly what the physics
computed. A clip shows one or more scenarios side by side, one panel each:

- the arena's walls, the obstacles, and the target zone (a dashed circle);
- the box, with an arrow along its own x axis (its heading): orange while moving, grey while static friction holds it, green
  once delivered;
- the agents as discs with a tick along their heading, highlighted while they touch the box;
- a corner caption: the scenario, the step, the simulated time, whether the box is at rest, how many agents touch it, and
  "delivered" once it is.

Milestone clips of trained champions, the montage and the charts (M3) build on these.
"""

# matplotlib's stubs type many methods' **kwargs as unknown (set_xticks, plot, text, savefig, grab_frame...), and imageio-ffmpeg
# ships none: strict mode would flag every call. Only that check is off here; everything else stays strict.
# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false

import math
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
from auxein.backend import HostArray
from matplotlib.axes import Axes
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrow, Polygon, Rectangle
from matplotlib.text import Text

from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.world import Trajectory
from auxein_scenarios.common.geometry import obb_contact, obb_corners
from auxein_scenarios.common.rendering import figure, frame_records, grid_shape, save_png, write_mp4

__all__ = ["TOUCH_TOLERANCE", "render_clip", "render_gallery", "touching"]

TOUCH_TOLERANCE = 0.02
"""An agent touches the box when its disc is within this distance of the box's surface (contacts overlap slightly, or not at
all for a light touch)."""

AGENT, AGENT_TOUCHING = "#4c72b0", "#dd8452"
BOX_MOVING, BOX_RESTING, BOX_DELIVERED = "#e1812c", "#8c8c8c", "#3a923a"
OBSTACLE, TARGET, WALL = "#5f5f5f", "#3a923a", "#222222"


def touching(trajectory: Trajectory, record: int, candidate: int, scenario: int, params: WorldParams) -> HostArray:
    """Which agents touch the box in one recorded state: a boolean array over the team."""
    t, n, s = record, candidate, scenario
    angle = trajectory.box_angle[t, n, s]
    _, _, distance, _, _ = obb_contact(
        trajectory.agent_x[t, n, s],
        trajectory.agent_y[t, n, s],
        np.asarray(trajectory.box_x[t, n, s]),
        np.asarray(trajectory.box_y[t, n, s]),
        np.asarray(math.cos(angle)),
        np.asarray(math.sin(angle)),
        params.box.width / 2.0,
        params.box.height / 2.0,
    )
    return np.asarray(distance) <= params.agent.radius + TOUCH_TOLERANCE


def _box_outline(x: float, y: float, angle: float, params: WorldParams) -> HostArray:
    cx, cy = obb_corners(
        np.asarray(x),
        np.asarray(y),
        np.asarray(math.cos(angle)),
        np.asarray(math.sin(angle)),
        params.box.width / 2.0,
        params.box.height / 2.0,
    )
    return np.stack([np.asarray(cx), np.asarray(cy)], axis=-1)


def _draw_static(ax: Axes, layout: Mapping[str, object], params: WorldParams) -> None:
    half = params.arena.half
    ax.set_xlim(-half - 0.3, half + 0.3)
    ax.set_ylim(-half - 0.3, half + 0.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(Rectangle((-half, -half), 2 * half, 2 * half, fill=False, edgecolor=WALL, linewidth=2.0))
    for x, y, r in cast("list[list[float]]", layout["circles"]):
        ax.add_patch(Circle((x, y), r, color=OBSTACLE))
    for x, y, hx, hy in cast("list[list[float]]", layout["rects"]):
        ax.add_patch(Rectangle((x - hx, y - hy), 2 * hx, 2 * hy, color=OBSTACLE))
    tx, ty = cast("list[float]", layout["target"])
    ax.add_patch(Circle((tx, ty), params.simulation.target_radius, fill=False, linestyle="--", edgecolor=TARGET, linewidth=1.5))
    ax.plot([tx], [ty], marker="+", color=TARGET, markersize=8)


class _Panel:
    """One scenario's panel: the static scene, and the artists that move, updated in place for every frame."""

    def __init__(self, ax: Axes, layout: Mapping[str, object], params: WorldParams, label: str, agents: int) -> None:
        self.ax, self.params, self.label = ax, params, label
        _draw_static(ax, layout, params)
        self.box = Polygon(np.zeros((4, 2)), closed=True, facecolor=BOX_RESTING, edgecolor="black", linewidth=1.0, zorder=3)
        ax.add_patch(self.box)
        self.arrow: FancyArrow | None = None
        diameter = 2 * params.agent.radius
        self.agents = EllipseCollection(
            widths=np.full(agents, diameter),
            heights=np.full(agents, diameter),
            angles=np.zeros(agents),
            units="xy",
            offsets=np.zeros((agents, 2)),
            offset_transform=ax.transData,
            facecolors=AGENT,
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
        )
        ax.add_collection(self.agents)
        self.ticks = LineCollection([], colors="white", linewidths=1.0, zorder=5)
        ax.add_collection(self.ticks)
        self.caption: Text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            zorder=6,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    def update(self, trajectory: Trajectory, record: int, candidate: int, scenario: int, dt: float) -> None:
        t, n, s, p = record, candidate, scenario, self.params
        bx, by, angle = float(trajectory.box_x[t, n, s]), float(trajectory.box_y[t, n, s]), float(trajectory.box_angle[t, n, s])
        resting, delivered = bool(trajectory.box_at_rest[t, n, s]), bool(trajectory.done[t, n, s])
        self.box.set_xy(_box_outline(bx, by, angle, p))
        self.box.set_facecolor(BOX_DELIVERED if delivered else BOX_RESTING if resting else BOX_MOVING)
        if self.arrow is not None:
            self.arrow.remove()
        length = 0.4 * p.box.width
        self.arrow = FancyArrow(
            bx, by, length * math.cos(angle), length * math.sin(angle), width=0.06, head_width=0.25, color="black", zorder=4
        )
        self.ax.add_patch(self.arrow)
        x, y, heading = trajectory.agent_x[t, n, s], trajectory.agent_y[t, n, s], trajectory.agent_heading[t, n, s]
        touch = touching(trajectory, t, n, s, p)
        self.agents.set_offsets(np.stack([x, y], axis=-1))
        self.agents.set_facecolor([AGENT_TOUCHING if hit else AGENT for hit in touch])
        r = p.agent.radius
        ends = np.stack([np.stack([x, y], axis=-1), np.stack([x + r * np.cos(heading), y + r * np.sin(heading)], axis=-1)], axis=1)
        self.ticks.set_segments(list(ends))
        step = int(trajectory.step[t])
        lines = [
            self.label,
            f"step {step:4d}  t {step * dt:5.1f} s",
            f"box {'at rest' if resting else 'moving'}  touching {int(np.sum(touch))}",
        ]
        if delivered:
            lines.append("DELIVERED")
        self.caption.set_text("\n".join(lines))


def _panels(
    layouts: Sequence[Mapping[str, object]],
    params: WorldParams,
    labels: Sequence[str] | None,
    columns: int | None,
    title: str,
    subtitle: str,
    panel_size: float,
    dpi: int,
) -> tuple[Figure, list[_Panel]]:
    rows, cols = grid_shape(len(layouts), columns)
    top = 0.86 if subtitle else 0.93
    fig = figure(cols * panel_size, rows * panel_size / top, dpi)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=top, wspace=0.03, hspace=0.03)
    if title:
        # about 11 characters per inch at this font size: wrap the subtitle to the figure's width
        wrapped = textwrap.fill(subtitle, width=max(40, int(cols * panel_size * 11))) if subtitle else ""
        fig.suptitle(title + (f"\n{wrapped}" if wrapped else ""), fontsize=11)
    agents = len(cast("list[object]", layouts[0]["agents"]))
    panels: list[_Panel] = []
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        if i >= len(layouts):
            ax.set_axis_off()
            continue
        label = labels[i] if labels is not None else f"scenario {i}"
        panels.append(_Panel(ax, layouts[i], params, label, agents))
    return fig, panels


def render_clip(
    trajectory: Trajectory,
    layouts: Sequence[Mapping[str, object]],
    params: WorldParams,
    path: str | Path,
    *,
    scenarios: Sequence[int] | None = None,
    candidate: int = 0,
    labels: Sequence[str] | None = None,
    title: str = "",
    subtitle: str = "",
    fps: float = 30.0,
    speedup: float = 2.0,
    columns: int | None = None,
    panel_size: float = 5.0,
    dpi: int = 100,
) -> Path:
    """An MP4 of a recorded trajectory: one panel per scenario (all of them by default) of candidate `candidate`.

    `layouts` are the params of every scenario of the trajectory (for the static scene). `title` and `subtitle` go above the
    panels: say there what to look for. Simulated time runs `speedup` times faster than real time."""
    chosen = list(range(len(layouts))) if scenarios is None else list(scenarios)
    names = [labels[i] for i in chosen] if labels is not None else [f"scenario {i}" for i in chosen]
    fig, panels = _panels([layouts[i] for i in chosen], params, names, columns, title, subtitle, panel_size, dpi)
    frames = frame_records(trajectory.step, params.simulation.dt, fps, speedup)

    def update(record: int) -> None:
        for panel, scenario in zip(panels, chosen, strict=True):
            panel.update(trajectory, record, candidate, scenario, params.simulation.dt)

    return write_mp4(fig, update, frames, path, fps)


def render_gallery(
    layouts: Sequence[Mapping[str, object]],
    params: WorldParams,
    path: str | Path,
    *,
    labels: Sequence[str] | None = None,
    title: str = "",
    columns: int | None = None,
    panel_size: float = 3.0,
    dpi: int = 100,
) -> Path:
    """A PNG grid of scenario layouts at their start: obstacles, the target zone, the box and the agents."""
    fig, panels = _panels(layouts, params, labels, columns, title, "", panel_size, dpi)
    for panel, layout in zip(panels, layouts, strict=True):
        agents = np.asarray(layout["agents"], dtype=np.float64).reshape(1, 1, 1, -1, 3)
        box = np.asarray(layout["box"], dtype=np.float64).reshape(1, 1, 1, 3)
        start = Trajectory(
            step=np.zeros(1, dtype=np.int64),
            agent_x=agents[..., 0],
            agent_y=agents[..., 1],
            agent_heading=agents[..., 2],
            box_x=box[..., 0],
            box_y=box[..., 1],
            box_angle=box[..., 2],
            box_at_rest=np.ones((1, 1, 1), dtype=bool),
            done=np.zeros((1, 1, 1), dtype=bool),
        )
        panel.update(start, 0, 0, 0, params.simulation.dt)
        difficulty, attempts = layout.get("difficulty", ""), layout.get("attempts", "")
        panel.caption.set_text(
            f"{panel.label}\n{difficulty}, {len(cast('list[object]', layout['agents']))} agents"
            + (f", {attempts} draws" if attempts else "")
        )
    return save_png(fig, path)
