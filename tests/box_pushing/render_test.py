"""Rendering: panels from different trajectories (different worlds and team sizes) to MP4 through the bundled ffmpeg, with
their captions; touching agents; the frames picked from a trajectory."""

from pathlib import Path

import numpy as np
from auxein.backend import Backend

from auxein_scenarios.box_pushing import scripted
from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.render import Panel, render_panels, touching
from auxein_scenarios.box_pushing.scenarios import layout_params
from auxein_scenarios.common.rendering import frame_records, grid_shape
from tests.box_pushing.support import FACING_X, line_behind, run

PARAMS = WorldParams()


def test_panels_from_two_trajectories_to_mp4(tmp_path: Path) -> None:
    """Two teams of different sizes, in worlds with different step limits, each its own trajectory, side by side: the shorter
    episode holds its last state while the other plays on."""
    small = [layout_params([0.0, 0.0, FACING_X], [3.0, 0.0], line_behind(0.0, 4, PARAMS, spread=0.75), circles=[[0.0, 5.0, 1.0]])]
    large = [layout_params([2.0, 2.0, 0.3], [8.0, 8.0], line_behind(2.0, 6, PARAMS, spread=0.75), rects=[[-5.0, -5.0, 1.0, 2.0]])]
    backend = Backend("numpy", "cpu", "float64")
    _, short, _ = run(backend, small, scripted.push(), 10)
    _, long, _ = run(backend, large, scripted.push(), 20)
    panels = [
        Panel(short, 0, 0, small[0], PARAMS, "brain a", "scenario-0000 · full", "not delivered"),
        Panel(
            long, 0, 0, large[0], PARAMS.replace(simulation={"max_steps": 300}), "brain b", "scenario-0000 · full", "delivered at step 9"
        ),
    ]
    path = render_panels(
        panels, tmp_path / "video.mp4", title="smoke", subtitle="a few frames", fps=10, speedup=1.0, panel_size=2.0, dpi=60
    )
    assert path.exists() and path.stat().st_size > 1000
    assert path.read_bytes()[4:8] == b"ftyp"  # an MP4 container


def test_touching_counts_agents_against_the_box() -> None:
    layouts = [layout_params([0.0, 0.0, FACING_X], [9.0, 0.0], line_behind(0.0, 3, PARAMS) + [[-5.0, 5.0, 0.0]])]
    _, trajectory, _ = run(Backend("numpy", "cpu", "float64"), layouts, scripted.idle, 1)
    assert touching(trajectory, 0, 0, 0, PARAMS).tolist() == [True, True, True, False]


def test_frame_records_and_grid() -> None:
    steps = np.arange(0, 101)
    frames = frame_records(steps, dt=0.05, fps=10, speedup=2.0, hold=0.5)
    # 5 s of simulated time at 2x is 2.5 s of video: 26 frames, every 4th step, then 5 held frames of the last record
    assert frames[:4] == [0, 4, 8, 12] and frames[25] == 100 and frames[26:] == [100] * 5 and len(frames) == 31
    sparse = frame_records(np.arange(0, 101, 10), dt=0.05, fps=10, speedup=1.0, hold=0.0)
    assert sparse[:4] == [0, 0, 0, 1]  # frames every 2 steps, records every 10: the nearest record (step 6 is nearer 10)
    assert grid_shape(6) == (2, 3) and grid_shape(1) == (1, 1) and grid_shape(3, columns=1) == (3, 1)
