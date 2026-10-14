"""Drawing box-pushing worlds (design doc §7): clips of recorded trajectories and galleries of scenario layouts.

Rendering is **offline and draws a recorded trajectory** (`world.TrajectoryRecorder`), so a clip shows exactly what the physics
computed. Everything is drawn in the scenarios' theme (`common.theme`): monochrome silver on near-black, state shown with
brightness, never with hue. A clip has a header (a small tracked label, a light headline, a muted lead saying what to look
for) above one panel per scenario:

- the floor, lit at its centre; the arena's walls; the obstacles in slate; the target zone as a dashed silver ring, filled
  once the box is delivered;
- the box: outlined in steel while static friction holds it, filled bright white while it moves, a tick along its own x axis
  (its heading);
- the agents as silver chevrons along their heading, inside a faint outline of their disc, bright white while they touch the
  box; agents within `LINK_RANGE` of each other joined by faint lines;
- a corner caption: the scenario, the step and simulated time, and the box's state (at rest, moving with how many agents
  touching it, or delivered).

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
from matplotlib.collections import EllipseCollection, LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle

from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.world import Trajectory
from auxein_scenarios.common import theme
from auxein_scenarios.common.geometry import obb_contact, obb_corners
from auxein_scenarios.common.rendering import figure, frame_records, grid_shape, save_png, write_mp4

__all__ = ["LINK_RANGE", "TOUCH_TOLERANCE", "render_clip", "render_gallery", "touching"]

TOUCH_TOLERANCE = 0.02
"""An agent touches the box when its disc is within this distance of the box's surface (contacts overlap slightly, or not at
all for a light touch)."""

LINK_RANGE = 1.5
"""Agents closer than this (centre to centre, in units) are joined by a line, fainter as they are farther apart."""

GUTTER = 0.12
"""Inches between panels and around them."""


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


def _chevrons(x: HostArray, y: HostArray, heading: HostArray, radius: float) -> HostArray:
    """The agents' glyphs: `(N, 4, 2)` polygons, the theme's chevron scaled to the radius and turned to the heading."""
    shape = np.asarray(theme.CHEVRON, dtype=np.float64) * radius  # (4, 2)
    c, s = np.cos(heading)[:, None], np.sin(heading)[:, None]
    px = x[:, None] + c * shape[None, :, 0] - s * shape[None, :, 1]
    py = y[:, None] + s * shape[None, :, 0] + c * shape[None, :, 1]
    return np.stack([px, py], axis=-1)


def _links(x: HostArray, y: HostArray) -> tuple[list[HostArray], list[tuple[float, float, float, float]]]:
    """The lines between agents closer than `LINK_RANGE`, with their colours (fainter with distance)."""
    dx, dy = x[:, None] - x[None, :], y[:, None] - y[None, :]
    distance = np.hypot(dx, dy)
    i, j = np.nonzero(np.triu(distance < LINK_RANGE, k=1))
    segments = [np.array([[x[a], y[a]], [x[b], y[b]]]) for a, b in zip(i.tolist(), j.tolist(), strict=True)]
    colours = [
        theme.rgba(theme.SILVER, theme.LINK_ALPHA * (1.0 - float(distance[a, b]) / LINK_RANGE))
        for a, b in zip(i.tolist(), j.tolist(), strict=True)
    ]
    return segments, colours


def _draw_static(ax: Axes, layout: Mapping[str, object], params: WorldParams) -> Circle:
    """The scene that does not move; returns the target zone's fill, shown on delivery."""
    half = params.arena.half
    margin = 0.3
    theme.style_axes(ax)
    ax.set_xlim(-half - margin, half + margin)
    ax.set_ylim(-half - margin, half + margin)
    ax.set_aspect("equal")
    theme.floor(ax, (-half, half, -half, half))
    ax.add_patch(
        Rectangle((-half, -half), 2 * half, 2 * half, fill=False, edgecolor=theme.rgba(theme.STEEL, 0.55), linewidth=1.2, zorder=1)
    )
    for x, y, r in cast("list[list[float]]", layout["circles"]):
        ax.add_patch(Circle((x, y), r, facecolor=theme.SLATE, edgecolor=theme.LINE, linewidth=0.8, zorder=1))
    for x, y, hx, hy in cast("list[list[float]]", layout["rects"]):
        ax.add_patch(Rectangle((x - hx, y - hy), 2 * hx, 2 * hy, facecolor=theme.SLATE, edgecolor=theme.LINE, linewidth=0.8, zorder=1))
    tx, ty = cast("list[float]", layout["target"])
    radius = params.simulation.target_radius
    filled = Circle((tx, ty), radius, facecolor=theme.rgba(theme.SILVER, 0.16), edgecolor="none", zorder=1, visible=False)
    ax.add_patch(filled)
    ax.add_patch(
        Circle((tx, ty), radius, fill=False, linestyle=(0, (4, 3)), edgecolor=theme.rgba(theme.SILVER, 0.75), linewidth=1.0, zorder=1)
    )
    ax.plot([tx - 0.25, tx + 0.25], [ty, ty], color=theme.SUBTLE, linewidth=0.8, zorder=1)
    ax.plot([tx, tx], [ty - 0.25, ty + 0.25], color=theme.SUBTLE, linewidth=0.8, zorder=1)
    return filled


class _Panel:
    """One scenario's panel: the static scene, and the artists that move, updated in place for every frame."""

    def __init__(self, ax: Axes, layout: Mapping[str, object], params: WorldParams, label: str, agents: int, size: float) -> None:
        self.ax, self.params, self.label = ax, params, label
        self.target_fill = _draw_static(ax, layout, params)
        self.links = LineCollection([], linewidths=0.6, zorder=2)
        ax.add_collection(self.links)
        self.box = Polygon(np.zeros((4, 2)), closed=True, linewidth=1.4, zorder=3)
        ax.add_patch(self.box)
        self.heading = Line2D([], [], linewidth=1.6, solid_capstyle="round", zorder=4)
        ax.add_line(self.heading)
        diameter = 2 * params.agent.radius
        self.discs = EllipseCollection(
            widths=np.full(agents, diameter),
            heights=np.full(agents, diameter),
            angles=np.zeros(agents),
            units="xy",
            offsets=np.zeros((agents, 2)),
            offset_transform=ax.transData,
            facecolors="none",
            edgecolors=[theme.rgba(theme.STEEL, 0.28)],
            linewidths=0.5,
            zorder=4,
        )
        ax.add_collection(self.discs)
        self.chevrons = PolyCollection([], linewidths=0, zorder=5)
        ax.add_collection(self.chevrons)
        # the caption scales with the panel: sizes are for a 5-inch panel
        scale = size / 5.0
        self.caption_label = ax.text(
            0.03,
            0.97,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            color=theme.SILVER,
            fontproperties=theme.font("label", 7.5 * scale),
            zorder=6,
        )
        self.caption_readout = ax.text(
            0.03,
            0.925,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            color=theme.SUBTLE,
            fontproperties=theme.font("readout", 7.5 * scale),
            zorder=6,
        )
        self.caption_status = ax.text(
            0.03, 0.885, "", transform=ax.transAxes, va="top", ha="left", fontproperties=theme.font("label", 7.0 * scale), zorder=6
        )

    def caption(self, readout: str, status: str, bright: bool) -> None:
        self.caption_label.set_text(theme.tracked(self.label))
        self.caption_readout.set_text(readout)
        self.caption_status.set_text(theme.tracked(status) if status else "")
        self.caption_status.set_color(theme.TEXT if bright else theme.SUBTLE)

    def update(self, trajectory: Trajectory, record: int, candidate: int, scenario: int, dt: float) -> None:
        t, n, s, p = record, candidate, scenario, self.params
        bx, by, angle = float(trajectory.box_x[t, n, s]), float(trajectory.box_y[t, n, s]), float(trajectory.box_angle[t, n, s])
        resting, delivered = bool(trajectory.box_at_rest[t, n, s]), bool(trajectory.done[t, n, s])
        moving = not resting and not delivered
        self.box.set_xy(_box_outline(bx, by, angle, p))
        if delivered:
            self.box.set_facecolor(theme.SILVER)
            self.box.set_edgecolor(theme.TEXT)
        elif moving:
            self.box.set_facecolor(theme.TEXT)
            self.box.set_edgecolor(theme.TEXT)
        else:
            self.box.set_facecolor(theme.SLATE)
            self.box.set_edgecolor(theme.STEEL)
        self.target_fill.set_visible(delivered)
        length = 0.4 * p.box.width
        self.heading.set_data([bx, bx + length * math.cos(angle)], [by, by + length * math.sin(angle)])
        self.heading.set_color(theme.BACKGROUND if (moving or delivered) else theme.STEEL)

        x, y, heading = trajectory.agent_x[t, n, s], trajectory.agent_y[t, n, s], trajectory.agent_heading[t, n, s]
        touch = touching(trajectory, t, n, s, p)
        self.discs.set_offsets(np.stack([x, y], axis=-1))
        self.chevrons.set_verts(list(_chevrons(x, y, heading, p.agent.radius)))
        self.chevrons.set_facecolor([theme.TEXT if hit else theme.rgba(theme.SILVER, 0.8) for hit in touch])
        segments, colours = _links(x, y)
        self.links.set_segments(segments)
        self.links.set_color(colours)

        step = int(trajectory.step[t])
        readout = f"step {step:d}  ·  {step * dt:.1f} s"
        if delivered:
            self.caption(readout, "delivered", True)
        elif moving:
            self.caption(readout, f"moving · {int(np.sum(touch))} touching", True)
        else:
            self.caption(readout, f"at rest · {int(np.sum(touch))} touching", False)


def _header(fig: Figure, width: float, kicker: str, title: str, lead: str) -> None:
    """Draws the header (label, headline, lead) at the top of `fig`, `width` inches wide (`_header_height` tall)."""
    left, top = 0.32, 0.36
    headline_lines = textwrap.wrap(title, width=max(20, int(width * 6.2))) if title else []
    lead_lines = textwrap.wrap(lead, width=max(30, int(width * 13))) if lead else []
    total = fig.get_size_inches()[1]

    def place(y_from_top: float) -> float:
        return 1.0 - y_from_top / total

    y = top
    if kicker:
        fig.text(left / width, place(y), theme.tracked(kicker), color=theme.SILVER, fontproperties=theme.font("label", 8.5), va="top")
        y += 0.3
    for line in headline_lines:
        fig.text(left / width, place(y), line, color=theme.TEXT, fontproperties=theme.font("headline", 23), va="top")
        y += 0.42
    if lead_lines:
        y += 0.12
        for line in lead_lines:
            fig.text(left / width, place(y), line, color=theme.MUTED, fontproperties=theme.font("lead", 11), va="top")
            y += 0.21


def _header_height(width: float, kicker: str, title: str, lead: str) -> float:
    headline = len(textwrap.wrap(title, width=max(20, int(width * 6.2)))) if title else 0
    body = len(textwrap.wrap(lead, width=max(30, int(width * 13)))) if lead else 0
    return 0.36 + (0.3 if kicker else 0.0) + 0.42 * headline + (0.12 if body else 0.0) + 0.21 * body + 0.3


def _panels(
    layouts: Sequence[Mapping[str, object]],
    params: WorldParams,
    labels: Sequence[str] | None,
    columns: int | None,
    kicker: str,
    title: str,
    subtitle: str,
    panel_size: float,
    dpi: int,
) -> tuple[Figure, list[_Panel]]:
    rows, cols = grid_shape(len(layouts), columns)
    width = cols * panel_size + (cols + 1) * GUTTER
    header = _header_height(width, kicker, title, subtitle) if (kicker or title or subtitle) else GUTTER
    height = header + rows * panel_size + rows * GUTTER
    fig = figure(width, height, dpi)
    fig.set_facecolor(theme.BACKGROUND)
    if kicker or title or subtitle:
        _header(fig, width, kicker, title, subtitle)
    agents = len(cast("list[object]", layouts[0]["agents"]))
    panels: list[_Panel] = []
    for i, layout in enumerate(layouts):
        row, col = divmod(i, cols)
        left = GUTTER + col * (panel_size + GUTTER)
        bottom = height - header - (row + 1) * panel_size - row * GUTTER
        ax = fig.add_axes((left / width, bottom / height, panel_size / width, panel_size / height))
        label = labels[i] if labels is not None else f"scenario {i}"
        panels.append(_Panel(ax, layout, params, label, agents, panel_size))
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
    kicker: str = "",
    title: str = "",
    subtitle: str = "",
    fps: float = 30.0,
    speedup: float = 2.0,
    columns: int | None = None,
    panel_size: float = 5.0,
    dpi: int = 100,
) -> Path:
    """An MP4 of a recorded trajectory: one panel per scenario (all of them by default) of candidate `candidate`.

    `layouts` are the params of every scenario of the trajectory (for the static scene). `kicker` (a short label), `title`
    (the headline) and `subtitle` (the lead: say there what to look for) go above the panels. Simulated time runs `speedup`
    times faster than real time."""
    chosen = list(range(len(layouts))) if scenarios is None else list(scenarios)
    names = [labels[i] for i in chosen] if labels is not None else [f"scenario {i}" for i in chosen]
    fig, panels = _panels([layouts[i] for i in chosen], params, names, columns, kicker, title, subtitle, panel_size, dpi)
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
    kicker: str = "",
    title: str = "",
    columns: int | None = None,
    panel_size: float = 3.0,
    dpi: int = 100,
) -> Path:
    """A PNG grid of scenario layouts at their start: obstacles, the target zone, the box and the agents."""
    fig, panels = _panels(layouts, params, labels, columns, kicker, title, "", panel_size, dpi)
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
        team = len(cast("list[object]", layout["agents"]))
        draws = f"  ·  {attempts} draw{'s' if attempts != 1 else ''}" if attempts else ""
        panel.caption(f"{difficulty}  ·  {team} agents{draws}", "", False)
    return save_png(fig, path)
