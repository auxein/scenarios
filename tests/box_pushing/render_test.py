"""Rendering smoke tests: a few frames to MP4 through the bundled ffmpeg, a gallery PNG, and the command-line tool."""

import math
from pathlib import Path

import numpy as np
from auxein.backend import Backend

from auxein_scenarios.box_pushing import cli, demos, scenarios, scripted
from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.render import render_clip, render_gallery, touching
from auxein_scenarios.box_pushing.scenarios import layout_params
from auxein_scenarios.common.rendering import frame_records, grid_shape
from tests.box_pushing.support import FACING_X, line_behind, run

PARAMS = WorldParams()


def test_clip_to_mp4(tmp_path: Path) -> None:
    layouts = [
        layout_params([0.0, 0.0, FACING_X], [3.0, 0.0], line_behind(0.0, 6, PARAMS, spread=0.75), circles=[[0.0, 5.0, 1.0]]),
        layout_params([2.0, 2.0, 0.3], [8.0, 8.0], line_behind(2.0, 6, PARAMS, spread=0.75), rects=[[-5.0, -5.0, 1.0, 2.0]]),
    ]
    _, trajectory, _ = run(Backend("numpy", "cpu", "float64"), layouts, scripted.push(), 20)
    path = render_clip(
        trajectory,
        layouts,
        PARAMS,
        tmp_path / "clip.mp4",
        title="smoke",
        subtitle="a few frames",
        fps=10,
        speedup=1.0,
        panel_size=2.0,
        dpi=60,
    )
    assert path.exists() and path.stat().st_size > 1000
    assert path.read_bytes()[4:8] == b"ftyp"  # an MP4 container


def test_touching_counts_agents_against_the_box() -> None:
    layouts = [layout_params([0.0, 0.0, FACING_X], [9.0, 0.0], line_behind(0.0, 3, PARAMS) + [[-5.0, 5.0, 0.0]])]
    _, trajectory, _ = run(Backend("numpy", "cpu", "float64"), layouts, scripted.idle, 1)
    assert touching(trajectory, 0, 0, 0, PARAMS).tolist() == [True, True, True, False]


def test_gallery_png(tmp_path: Path) -> None:
    generated = scenarios.training_set(10, PARAMS, seed=5, mix=((scenarios.FULL, 3),))
    path = render_gallery(
        [dict(s.params) for s in generated],
        PARAMS,
        tmp_path / "gallery.png",
        labels=list(generated.ids),
        title="gallery",
        panel_size=2.0,
        dpi=50,
    )
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_frame_records_and_grid() -> None:
    steps = np.arange(0, 101)
    frames = frame_records(steps, dt=0.05, fps=10, speedup=2.0, hold=0.5)
    # 5 s of simulated time at 2x is 2.5 s of video: 26 frames, every 4th step, then 5 held frames of the last record
    assert frames[:4] == [0, 4, 8, 12] and frames[25] == 100 and frames[26:] == [100] * 5 and len(frames) == 31
    sparse = frame_records(np.arange(0, 101, 10), dt=0.05, fps=10, speedup=1.0, hold=0.0)
    assert sparse[:4] == [0, 0, 0, 1]  # frames every 2 steps, records every 10: the nearest record (step 6 is nearer 10)
    assert grid_shape(6) == (2, 3) and grid_shape(1) == (1, 1) and grid_shape(3, columns=1) == (3, 1)


def test_cli_gallery(tmp_path: Path) -> None:
    assert cli.main(["scenarios", "--difficulty", "easy", "--seed", "4", "--count", "2", "--agents", "5", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "scenarios-easy-seed4.png").exists()


def test_every_demo_builds() -> None:
    for name in demos.DEMOS:
        if name == "delivery":
            continue  # it simulates to choose its scenarios; rendered by hand (box-pushing demo delivery)
        demo = demos.build(name, PARAMS)
        assert demo.layouts and demo.steps > 0 and demo.title
        teams = {len(p["agents"]) for p in demo.layouts}  # type: ignore[arg-type]
        assert len(teams) == 1, f"{name}: every panel needs the same team size"
        for layout in demo.layouts:
            agents = np.asarray(layout["agents"])
            assert np.all(np.abs(agents[:, :2]) < PARAMS.arena.half), name


def test_cli_threshold_demo(tmp_path: Path) -> None:
    assert cli.main(["demo", "threshold", "--out", str(tmp_path), "--fps", "5", "--speedup", "8"]) == 0
    assert (tmp_path / "threshold.mp4").stat().st_size > 1000
    assert math.isfinite(PARAMS.derived.static_force)
