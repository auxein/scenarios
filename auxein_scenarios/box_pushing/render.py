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

A panel shows one **cell**, a candidate on a scenario: by default one candidate on every scenario (the demos), or, for brains,
every candidate on one scenario (`cells`).

**The sensing overlay** (M1b, design doc §7), optional, draws what chosen agents sensed at each step (`environment.SensingTrace`,
recorded by the same step loop that scores an episode), again with brightness only:

- the watched agent bright white, inside its sensing circle `r` (dashed), with ticks at the sector boundaries;
- its rays, faint, each ending in a dot where it hit: bright white on the box, steel on a wall or an obstacle, none when nothing
  was in range; rays on the box drawn brighter;
- the box cue, a line to the box's centre, while the box is in range;
- the teammates it senses, joined to it by lines as bright as they weigh (`1 − d / r`): M1a's faint links, now meaning "sensed".
  With the overlay on, these replace the proximity links.

**Montages** (M3, design doc §7) draw trained champions at chosen generations, at a fixed video size (1920 × 1080 by
default), in the same panels:

- **`render_sequence_montage`**: one segment per milestone, one after another, each opened by a title card ("Generation 50",
  its score and deliveries), the milestone's champion on the showcase scenarios, a header that names the generation, and a
  small fitness-over-generations curve whose marker moves to the milestone being shown;
- **`render_grid_montage`**: a few milestones side by side on **one** scenario, simultaneously: the clearest before and after,
  with the same curve marking the generations shown.

Every panel of a clip or montage of a run carries a footer saying plainly what it is (a training scenario, or a held-out one
never seen in training) and how its episode ended.
"""

# matplotlib's stubs type many methods' **kwargs as unknown (set_xticks, plot, text, savefig, grab_frame...), and imageio-ffmpeg
# ships none: strict mode would flag every call. Only that check is off here; everything else stays strict.
# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false

import math
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
from auxein.backend import HostArray
from matplotlib.axes import Axes
from matplotlib.collections import EllipseCollection, LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.text import Text

from auxein_scenarios.box_pushing.config import SensingParams, WorldParams
from auxein_scenarios.box_pushing.environment import TOUCH_TOLERANCE, SensingTrace
from auxein_scenarios.box_pushing.sensors import HIT_BOX, HIT_OBSTACLE
from auxein_scenarios.box_pushing.world import Trajectory
from auxein_scenarios.common import theme
from auxein_scenarios.common.geometry import obb_contact, obb_corners
from auxein_scenarios.common.rendering import CurveInset, TitleCard, figure, frame_records, grid_shape, save_png, write_mp4

__all__ = [
    "LINK_RANGE",
    "TOUCH_TOLERANCE",
    "Curve",
    "Segment",
    "render_clip",
    "render_gallery",
    "render_grid_montage",
    "render_sequence_montage",
    "touching",
]

Cell = tuple[int, int]
"""A panel's content: a candidate (row of the batch) on a scenario (column)."""

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


class _Overlay:
    """What one watched agent sensed, drawn over a panel (see the module docstring)."""

    def __init__(self, ax: Axes, sensing: SensingParams, agent: int, rays: int, half: float) -> None:
        self.sensing, self.agent = sensing, agent
        self.circle = Circle(
            (0.0, 0.0), sensing.range, fill=False, linestyle=(0, (2, 3)), edgecolor=theme.rgba(theme.SILVER, 0.35), linewidth=0.8, zorder=2
        )
        ax.add_patch(self.circle)
        self.ticks = LineCollection([], linewidths=0.8, colors=[theme.rgba(theme.SILVER, 0.35)], zorder=2)
        ax.add_collection(self.ticks)
        self.rays = LineCollection([], linewidths=0.6, zorder=2)
        ax.add_collection(self.rays)
        self.cue = Line2D([], [], linewidth=1.0, color=theme.rgba(theme.TEXT, 0.7), zorder=3)
        ax.add_line(self.cue)
        self.sensed = LineCollection([], linewidths=0.9, zorder=2)
        ax.add_collection(self.sensed)
        self.hits = ax.scatter(np.zeros(rays), np.zeros(rays), s=np.zeros(rays), linewidths=0, zorder=6)
        # the sensing circle reaches past the walls: keep the overlay inside the arena
        arena = Rectangle((-half, -half), 2 * half, 2 * half, transform=ax.transData)
        for artist in (self.circle, self.ticks, self.rays, self.cue, self.sensed, self.hits):
            artist.set_clip_path(arena)

    def update(
        self, senses: SensingTrace, t: int, n: int, s: int, x: HostArray, y: HostArray, heading: HostArray, box: tuple[float, float]
    ) -> None:
        a = senses.agents.index(self.agent)
        ax_, ay_ = float(x[self.agent]), float(y[self.agent])
        self.circle.set_center((ax_, ay_))
        r = self.sensing.range
        width = 2.0 * math.pi / self.sensing.sectors
        bounds = float(heading[self.agent]) + width * (np.arange(self.sensing.sectors) + 0.5)
        self.ticks.set_segments(
            [
                np.array([[ax_ + 0.82 * r * math.cos(b), ay_ + 0.82 * r * math.sin(b)], [ax_ + r * math.cos(b), ay_ + r * math.sin(b)]])
                for b in bounds
            ]
        )
        ends_x, ends_y, hit = senses.ray_x[t, n, s, a], senses.ray_y[t, n, s, a], senses.ray_hit[t, n, s, a]
        self.rays.set_segments([np.array([[ax_, ay_], [ex, ey]]) for ex, ey in zip(ends_x.tolist(), ends_y.tolist(), strict=True)])
        self.rays.set_color([theme.rgba(theme.TEXT, 0.45) if h == HIT_BOX else theme.rgba(theme.SILVER, 0.16) for h in hit.tolist()])
        self.hits.set_offsets(np.stack([ends_x, ends_y], axis=-1))
        self.hits.set_sizes([9.0 if h == HIT_BOX else 5.0 if h == HIT_OBSTACLE else 0.0 for h in hit.tolist()])
        self.hits.set_facecolor([theme.TEXT if h == HIT_BOX else theme.STEEL for h in hit.tolist()])
        if bool(senses.box_on[t, n, s, a]):
            self.cue.set_data([ax_, box[0]], [ay_, box[1]])
        else:
            self.cue.set_data([], [])
        weight = senses.teammate_weight[t, n, s, a]
        others = np.nonzero(weight > 0.0)[0].tolist()
        self.sensed.set_segments([np.array([[ax_, ay_], [float(x[j]), float(y[j])]]) for j in others])
        self.sensed.set_color([theme.rgba(theme.SILVER, 0.15 + 0.5 * float(weight[j])) for j in others])


class _Panel:
    """One scenario's panel: the static scene, and the artists that move, updated in place for every frame."""

    def __init__(
        self,
        ax: Axes,
        layout: Mapping[str, object],
        params: WorldParams,
        label: str,
        agents: int,
        size: float,
        overlays: Sequence[_Overlay] = (),
    ) -> None:
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
        self.overlays = list(overlays)
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
        # the footer: what the panel is (training or held-out, said plainly) and how its episode ended
        self.footer_note = ax.text(
            0.03, 0.075, "", transform=ax.transAxes, va="bottom", ha="left", color=theme.SILVER,
            fontproperties=theme.font("label", 6.5 * scale), zorder=6,
        )  # fmt: skip
        self.footer_outcome = ax.text(
            0.03, 0.03, "", transform=ax.transAxes, va="bottom", ha="left", color=theme.SUBTLE,
            fontproperties=theme.font("readout", 7.0 * scale), zorder=6,
        )  # fmt: skip

    def footer(self, note: str, outcome: str) -> None:
        self.footer_note.set_text(theme.tracked(note) if note else "")
        self.footer_outcome.set_text(outcome)

    def caption(self, readout: str, status: str, bright: bool) -> None:
        self.caption_label.set_text(theme.tracked(self.label))
        self.caption_readout.set_text(readout)
        self.caption_status.set_text(theme.tracked(status) if status else "")
        self.caption_status.set_color(theme.TEXT if bright else theme.SUBTLE)

    def update(
        self, trajectory: Trajectory, record: int, candidate: int, scenario: int, dt: float, senses: SensingTrace | None = None
    ) -> None:
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
        watched = {overlay.agent for overlay in self.overlays}
        self.chevrons.set_facecolor([theme.TEXT if hit or i in watched else theme.rgba(theme.SILVER, 0.8) for i, hit in enumerate(touch)])
        if self.overlays and senses is not None:
            self.links.set_segments([])
            for overlay in self.overlays:
                overlay.update(senses, t, n, s, x, y, heading, (bx, by))
        else:
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


def _panel_at(
    fig: Figure,
    rect: tuple[float, float, float, float],
    layout: Mapping[str, object],
    params: WorldParams,
    label: str,
    size: float,
    watch: Sequence[int] = (),
    sensing: SensingParams | None = None,
) -> _Panel:
    """A panel in `rect` (figure fractions), `size` inches wide (for its caption's type)."""
    ax = fig.add_axes(rect)
    agents = len(cast("list[object]", layout["agents"]))
    chosen = sensing or SensingParams()
    overlays = [_Overlay(ax, chosen, agent, chosen.rays, params.arena.half) for agent in watch]
    return _Panel(ax, layout, params, label, agents, size, overlays)


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
    watch: Sequence[int] = (),
    sensing: SensingParams | None = None,
) -> tuple[Figure, list[_Panel]]:
    rows, cols = grid_shape(len(layouts), columns)
    width = cols * panel_size + (cols + 1) * GUTTER
    header = _header_height(width, kicker, title, subtitle) if (kicker or title or subtitle) else GUTTER
    height = header + rows * panel_size + rows * GUTTER
    fig = figure(width, height, dpi)
    fig.set_facecolor(theme.BACKGROUND)
    if kicker or title or subtitle:
        _header(fig, width, kicker, title, subtitle)
    panels: list[_Panel] = []
    for i, layout in enumerate(layouts):
        row, col = divmod(i, cols)
        left = GUTTER + col * (panel_size + GUTTER)
        bottom = height - header - (row + 1) * panel_size - row * GUTTER
        label = labels[i] if labels is not None else f"scenario {i}"
        rect = (left / width, bottom / height, panel_size / width, panel_size / height)
        panels.append(_panel_at(fig, rect, layout, params, label, panel_size, watch, sensing))
    return fig, panels


def render_clip(
    trajectory: Trajectory,
    layouts: Sequence[Mapping[str, object]],
    params: WorldParams,
    path: str | Path,
    *,
    scenarios: Sequence[int] | None = None,
    candidate: int = 0,
    cells: Sequence[Cell] | None = None,
    labels: Sequence[str] | None = None,
    senses: SensingTrace | None = None,
    sensing: SensingParams | None = None,
    kicker: str = "",
    title: str = "",
    subtitle: str = "",
    fps: float = 30.0,
    speedup: float = 2.0,
    columns: int | None = None,
    panel_size: float = 5.0,
    dpi: int = 100,
    footers: Sequence[tuple[str, str]] | None = None,
    records: int | None = None,
) -> Path:
    """An MP4 of a recorded trajectory, one panel per cell (a candidate on a scenario).

    The cells are `cells` if given, else candidate `candidate` on `scenarios` (all of them by default); `labels` name the
    panels, in the order of the cells (by default the scenarios'). `layouts` are the params of every scenario of the trajectory
    (for the static scene). With `senses` (recorded by `environment.WatchRecorder` alongside the trajectory, with the world's
    `sensing`), the sensing overlay is drawn for its watched agents. `kicker` (a short label), `title` (the headline) and
    `subtitle` (the lead: say there what to look for) go above the panels. Simulated time runs `speedup` times faster than real
    time. `footers`, one `(note, outcome)` per cell, are written at the bottom of the panels (what the panel is, how its
    episode ended); `records` shows only the first records (a candidate whose episodes ended before the batch's)."""
    if cells is None:
        cells = [(candidate, scenario) for scenario in (range(len(layouts)) if scenarios is None else scenarios)]
    chosen = list(cells)
    names = list(labels) if labels is not None else [f"scenario {s}" for _, s in chosen]
    if senses is not None and senses.ray_x.shape[0] != trajectory.records:
        raise ValueError(f"the senses have {senses.ray_x.shape[0]} records and the trajectory {trajectory.records}: record both together")
    watch = senses.agents if senses is not None else ()
    fig, panels = _panels([layouts[s] for _, s in chosen], params, names, columns, kicker, title, subtitle, panel_size, dpi, watch, sensing)
    for panel, (note, outcome) in zip(panels, footers or [("", "")] * len(panels), strict=True):
        panel.footer(note, outcome)
    frames = frame_records(trajectory.step[: records or trajectory.records], params.simulation.dt, fps, speedup)

    def update(record: int) -> None:
        for panel, (row, scenario) in zip(panels, chosen, strict=True):
            panel.update(trajectory, record, row, scenario, params.simulation.dt, senses)

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


# --- montages (M3) ---


@dataclass(frozen=True)
class Curve:
    """The inset's data: `y` over `x` (the best score over generations), an optional `faint` second line (the generation's
    mean) and quiet `marks` (the milestones' generations)."""

    x: Sequence[float]
    y: Sequence[float]
    faint: Sequence[float] | None = None
    marks: Sequence[float] = ()
    label: str = "fitness over generations"


@dataclass(frozen=True)
class Segment:
    """One milestone of a sequence montage: its candidate's row in the trajectory and how many records to show, the title
    card's and header's text, one footer per showcase panel, and where its marker sits on the curve (its generation)."""

    candidate: int
    records: int
    label: str
    headline: str
    lead: str
    at: float
    footers: Sequence[tuple[str, str]] = field(default_factory=list[tuple[str, str]])


GUTTER_MONTAGE = 0.25
"""Inches between panels in a montage, and at its edges."""


def _montage_figure(size: tuple[int, int], dpi: int = 100) -> tuple[Figure, float, float]:
    width, height = size[0] / dpi, size[1] / dpi
    fig = figure(width, height, dpi)
    fig.set_facecolor(theme.BACKGROUND)
    return fig, width, height


def _montage_header(fig: Figure, width: float, height: float, kicker: str) -> tuple[Text, Text]:
    """The header texts of a montage, at its top: a fixed label, a headline and a lead that segments change."""
    left = GUTTER_MONTAGE * 2 / width
    top = 1.0 - 0.38 / height
    fig.text(left, top, theme.tracked(kicker), color=theme.SILVER, fontproperties=theme.font("label", 10), va="top")
    headline = fig.text(left, top - 0.32 / height, "", color=theme.TEXT, fontproperties=theme.font("headline", 30), va="top")
    lead = fig.text(left, top - 0.9 / height, "", color=theme.MUTED, fontproperties=theme.font("lead", 13), va="top")
    return headline, lead


def _row_of_panels(
    fig: Figure,
    width: float,
    height: float,
    layouts: Sequence[Mapping[str, object]],
    labels: Sequence[str],
    params: WorldParams,
    top: float,
    bottom: float,
    watch: Sequence[int],
    sensing: SensingParams | None,
) -> list[_Panel]:
    """Panels for `layouts`, centred in the band between `top` and `bottom` (inches from the figure's bottom), as large as fit."""
    count = len(layouts)
    rows, cols = grid_shape(count, min(count, 4))
    g = GUTTER_MONTAGE
    size = min((width - (cols + 1) * g) / cols, (top - bottom - (rows - 1) * g) / rows)
    x0 = (width - cols * size - (cols - 1) * g) / 2.0
    y0 = bottom + (top - bottom - rows * size - (rows - 1) * g) / 2.0
    panels: list[_Panel] = []
    for i, layout in enumerate(layouts):
        row, col = divmod(i, cols)
        left = x0 + col * (size + g)
        lower = y0 + (rows - 1 - row) * (size + g)
        panels.append(
            _panel_at(fig, (left / width, lower / height, size / width, size / height), layout, params, labels[i], size, watch, sensing)
        )
    return panels


def _inset(fig: Figure, width: float, height: float, curve: Curve, band: float) -> CurveInset:
    g = GUTTER_MONTAGE * 2
    rect = ((g + 0.35) / width, 0.45 / height, (width - 2 * g - 0.35) / width, (band - 0.75) / height)
    return CurveInset(fig, rect, curve.x, curve.y, faint=curve.faint, marks=curve.marks, label=curve.label, xlabel="generation", scale=1.3)


def render_sequence_montage(
    trajectory: Trajectory,
    layouts: Sequence[Mapping[str, object]],
    params: WorldParams,
    path: str | Path,
    *,
    scenarios: Sequence[int],
    labels: Sequence[str],
    segments: Sequence[Segment],
    curve: Curve,
    kicker: str,
    senses: SensingTrace | None = None,
    sensing: SensingParams | None = None,
    size: tuple[int, int] = (1920, 1080),
    fps: float = 30.0,
    speedup: float = 4.0,
    card_seconds: float = 1.5,
) -> Path:
    """The milestones one after another (module docstring): for each segment a title card for `card_seconds`, then its
    candidate on `scenarios` (indices into the trajectory's scenarios, whose `layouts` and `labels` they are), with the curve's
    marker at the segment's generation."""
    fig, width, height = _montage_figure(size)
    headline, lead = _montage_header(fig, width, height, kicker)
    band = min(2.2, 0.22 * height)
    watch = senses.agents if senses is not None else ()
    panels = _row_of_panels(fig, width, height, [layouts[s] for s in scenarios], labels, params, height - 1.7, band, watch, sensing)
    inset = _inset(fig, width, height, curve, band)
    card = TitleCard(fig, scale=height / 10.8)
    dt = params.simulation.dt
    frames: list[tuple[int, int]] = []  # (segment, record), record -1 for the title card
    for i, segment in enumerate(segments):
        frames += [(i, -1)] * max(1, round(card_seconds * fps))
        frames += [(i, r) for r in frame_records(trajectory.step[: segment.records], dt, fps, speedup, hold=0.6)]
    shown: list[int] = [-1]

    def update(frame: tuple[int, int]) -> None:
        i, record = frame
        segment = segments[i]
        if record < 0:
            card.show(segment.label, segment.headline, segment.lead)
            return
        card.hide()
        if shown[0] != i:
            shown[0] = i
            headline.set_text(segment.headline)
            lead.set_text(segment.lead)
            inset.mark(segment.at)
            for panel, (note, outcome) in zip(panels, segment.footers or [("", "")] * len(panels), strict=True):
                panel.footer(note, outcome)
        for panel, scenario in zip(panels, scenarios, strict=True):
            panel.update(trajectory, record, segment.candidate, scenario, dt, senses)

    return write_mp4(fig, update, frames, path, fps)


def render_grid_montage(
    trajectory: Trajectory,
    layouts: Sequence[Mapping[str, object]],
    params: WorldParams,
    path: str | Path,
    *,
    scenario: int,
    candidates: Sequence[int],
    labels: Sequence[str],
    footers: Sequence[tuple[str, str]],
    curve: Curve,
    kicker: str,
    title: str,
    lead: str,
    records: int | None = None,
    size: tuple[int, int] = (1920, 1080),
    fps: float = 30.0,
    speedup: float = 3.0,
) -> Path:
    """`candidates` (rows of the trajectory) side by side on one scenario, simultaneously (module docstring)."""
    fig, width, height = _montage_figure(size)
    headline, lead_text = _montage_header(fig, width, height, kicker)
    headline.set_text(title)
    lead_text.set_text(lead)
    band = min(2.2, 0.22 * height)
    panels = _row_of_panels(fig, width, height, [layouts[scenario]] * len(candidates), labels, params, height - 1.7, band, (), None)
    for panel, (note, outcome) in zip(panels, footers, strict=True):
        panel.footer(note, outcome)
    inset = _inset(fig, width, height, curve, band)
    inset.mark(float(curve.marks[-1]) if curve.marks else float(curve.x[-1]))
    frames = frame_records(trajectory.step[: records or trajectory.records], params.simulation.dt, fps, speedup)

    def update(record: int) -> None:
        for panel, candidate in zip(panels, candidates, strict=True):
            panel.update(trajectory, record, candidate, scenario, params.simulation.dt)

    return write_mp4(fig, update, frames, path, fps)
