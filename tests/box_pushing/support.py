"""Helpers for the box-pushing tests: hand-built layouts, simulations on a backend, host copies."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from auxein.backend import Backend

from auxein_scenarios.box_pushing import scenarios
from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.world import Controller, Layout, State, Trajectory, TrajectoryRecorder, initial_state, simulate

FACING_X = math.pi / 2
"""The box's angle with its long side vertical, so that it is pushed on its long face along +x."""


def host(backend: Backend, array: Any) -> np.ndarray:
    return np.asarray(backend.to_numpy(array), dtype=np.float64)


def line_behind(box_x: float, count: int, params: WorldParams, *, spread: float | None = None, gap: float = 0.0) -> list[list[float]]:
    """`count` agents touching the long face of a box at `(box_x, 0)` turned to face +x, evenly spread across the face (centred),
    facing +x."""
    r = params.agent.radius
    face = box_x - params.box.height / 2.0 - r - gap
    half = (params.box.width / 2.0 - r) if spread is None else spread
    ys = [0.0] if count == 1 else np.linspace(-half, half, count).tolist()
    return [[face, float(y), 0.0] for y in ys]


def run(
    backend: Backend,
    layouts: Sequence[Mapping[str, object]],
    controller: Controller,
    steps: int,
    params: WorldParams | None = None,
    *,
    n: int = 1,
    circles: int | None = None,
    rects: int | None = None,
    start: State | None = None,
    stop_when_done: bool = False,
) -> tuple[State, Trajectory, Layout]:
    """Simulates the layouts for `n` candidates, recording every step."""
    p = params or WorldParams()
    layout = scenarios.layout_arrays(layouts, backend, circles=circles, rects=rects)
    recorder = TrajectoryRecorder()
    state = start if start is not None else initial_state(layout, n)
    end = simulate(state, layout, p, controller, steps, hook=recorder, stop_when_done=stop_when_done)
    return end, recorder.trajectory(), layout
