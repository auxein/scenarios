"""The penalty contacts, pair by pair, against hand-computed configurations: touching, overlapping, separated; forces only
pushing; torques consistent with the contact point; padded obstacles contributing nothing. On every backend."""

import math
from typing import Any

import numpy as np
import pytest
from auxein.backend import Backend

from auxein_scenarios.box_pushing import contacts
from auxein_scenarios.box_pushing.contacts import BoxKinematics

K, C, R = 200.0, 6.0, 0.25


def tol(backend: Backend) -> float:
    return 1e-9 if backend.precision == "float64" else 1e-4


def arr(backend: Backend, values: Any) -> Any:
    return backend.asarray(np.asarray(values, dtype=np.float64))


def host(backend: Backend, *arrays: Any) -> list[np.ndarray]:
    return [np.asarray(backend.to_numpy(a), dtype=np.float64) for a in arrays]


def box(
    backend: Backend, x: float = 0.0, y: float = 0.0, angle: float = 0.0, vx: float = 0.0, vy: float = 0.0, spin: float = 0.0
) -> BoxKinematics:
    return BoxKinematics(
        x=arr(backend, x),
        y=arr(backend, y),
        c=arr(backend, math.cos(angle)),
        s=arr(backend, math.sin(angle)),
        vx=arr(backend, vx),
        vy=arr(backend, vy),
        spin=arr(backend, spin),
        hx=1.0,
        hy=0.6,
    )


def test_penalty_only_pushes(backend: Backend) -> None:
    overlap = arr(backend, [-0.1, 0.0, 0.05, 0.05, 0.05])
    growth = arr(backend, [0.0, 0.0, 0.0, 1.0, -10.0])  # the last one separates fast: the damper would pull
    got = host(backend, contacts.penalty(overlap, growth, K, C))[0]
    np.testing.assert_allclose(got, [0.0, 0.0, K * 0.05, K * 0.05 + C, 0.0], atol=tol(backend) * K)


def test_agent_agent_overlapping_touching_separated(backend: Backend) -> None:
    # pairs (0, 1) overlap by 0.1 along x; agent 2 exactly touches agent 1; agent 3 is far away
    x = arr(backend, [0.0, 0.4, 0.9, 5.0])
    y = arr(backend, [0.0, 0.0, 0.0, 5.0])
    zero = arr(backend, [0.0, 0.0, 0.0, 0.0])
    fx, fy = host(backend, *contacts.agent_agent(x, y, zero, zero, R, K, C))
    np.testing.assert_allclose(fx, [-K * 0.1, K * 0.1, 0.0, 0.0], atol=tol(backend) * K)
    np.testing.assert_allclose(fy, 0.0, atol=tol(backend) * K)


def test_agent_agent_conserves_momentum_and_handles_coincident_agents(backend: Backend) -> None:
    rng = np.random.default_rng(0)
    x, y = arr(backend, rng.uniform(-1, 1, (3, 30))), arr(backend, rng.uniform(-1, 1, (3, 30)))
    vx, vy = arr(backend, rng.normal(size=(3, 30))), arr(backend, rng.normal(size=(3, 30)))
    fx, fy = host(backend, *contacts.agent_agent(x, y, vx, vy, R, K, C))
    assert np.abs(fx).max() > 1.0  # there are contacts
    np.testing.assert_allclose(fx.sum(axis=-1), 0.0, atol=tol(backend) * 1e3)
    np.testing.assert_allclose(fy.sum(axis=-1), 0.0, atol=tol(backend) * 1e3)
    # two agents at the same point: equal and opposite forces, no NaN
    same = arr(backend, [1.0, 1.0])
    zero = arr(backend, [0.0, 0.0])
    fx, fy = host(backend, *contacts.agent_agent(same, same, zero, zero, R, K, C))
    np.testing.assert_allclose(fx, [-K * 2 * R, K * 2 * R], atol=tol(backend) * K)
    assert np.all(np.isfinite(fy))


def test_agent_circles_and_padding(backend: Backend) -> None:
    x, y = arr(backend, [1.6, 5.0]), arr(backend, [0.0, 5.0])
    zero = arr(backend, [0.0, 0.0])
    # a circle of radius 1.5 at the origin: the first agent overlaps it by 0.15; the second slot is padding at the same place
    cx, cy, cr = arr(backend, [0.0, 1.6]), arr(backend, [0.0, 0.0]), arr(backend, [1.5, 3.0])
    valid = arr(backend, [1.0, 0.0])
    fx, fy = host(backend, *contacts.agent_circles(x, y, zero, zero, R, cx, cy, cr, valid, K, C))
    np.testing.assert_allclose(fx, [K * 0.15, 0.0], atol=tol(backend) * K)
    np.testing.assert_allclose(fy, [0.0, 0.0], atol=tol(backend) * K)
    # padded with more invalid slots: the same numbers, to the bit
    mx, my = arr(backend, [0.0, 1.6, 0.0, 0.0]), arr(backend, [0.0, 0.0, 0.0, 9.0])
    mr, mv = arr(backend, [1.5, 3.0, 2.0, 1.0]), arr(backend, [1.0, 0.0, 0.0, 0.0])
    gx, gy = host(backend, *contacts.agent_circles(x, y, zero, zero, R, mx, my, mr, mv, K, C))
    np.testing.assert_array_equal(gx, fx)
    np.testing.assert_array_equal(gy, fy)


def test_agent_rects(backend: Backend) -> None:
    # a 2 × 4 rectangle at (0, 0): agent 0 overlaps its right face by 0.05, agent 1 is inside near its top face
    x, y = arr(backend, [1.2, 0.0]), arr(backend, [0.5, 1.9])
    zero = arr(backend, [0.0, 0.0])
    fx, fy = host(
        backend,
        *contacts.agent_rects(
            x,
            y,
            zero,
            zero,
            R,
            arr(backend, [0.0]),
            arr(backend, [0.0]),
            arr(backend, [1.0]),
            arr(backend, [2.0]),
            arr(backend, [1.0]),
            K,
            C,
        ),
    )
    np.testing.assert_allclose(fx, [K * 0.05, 0.0], atol=tol(backend) * K)
    np.testing.assert_allclose(fy, [0.0, K * (R + 0.1)], atol=tol(backend) * K)


def test_agent_walls(backend: Backend) -> None:
    x, y = arr(backend, [-9.9, 9.8, 0.0]), arr(backend, [0.0, 9.9, 0.0])
    zero = arr(backend, [0.0, 0.0, 0.0])
    fx, fy = host(backend, *contacts.agent_walls(x, y, zero, zero, R, 10.0, K, C))
    np.testing.assert_allclose(fx, [K * 0.15, -K * 0.05, 0.0], atol=tol(backend) * K)
    np.testing.assert_allclose(fy, [0.0, -K * 0.15, 0.0], atol=tol(backend) * K)


def test_agent_box_force_reaction_and_torque(backend: Backend) -> None:
    """An agent pressing on the box's left face above its centre pushes it to +x and turns it clockwise (negative torque)."""
    b = box(backend)
    x, y = arr(backend, [[-1.2, 5.0]]), arr(backend, [[0.4, 5.0]])
    zero = arr(backend, [[0.0, 0.0]])
    fx, fy, bfx, bfy, torque = host(backend, *contacts.agent_box(x, y, zero, zero, R, b, K, C))
    overlap = R - 0.2
    np.testing.assert_allclose(fx[0], [-K * overlap, 0.0], atol=tol(backend) * K)
    np.testing.assert_allclose(fy[0], 0.0, atol=tol(backend) * K)
    np.testing.assert_allclose([bfx[0], bfy[0]], [K * overlap, 0.0], atol=tol(backend) * K)
    # applied at (-1, 0.4): r × f = (-1)(0) - (0.4)(K overlap)
    np.testing.assert_allclose(torque[0], -0.4 * K * overlap, atol=tol(backend) * K)


def test_agent_box_rotated(backend: Backend) -> None:
    """The box turned a quarter turn: its long side is vertical. An agent below it presses its end."""
    b = box(backend, angle=math.pi / 2)
    x, y = arr(backend, [[0.0]]), arr(backend, [[-1.2]])
    zero = arr(backend, [[0.0]])
    fx, fy, bfx, bfy, torque = host(backend, *contacts.agent_box(x, y, zero, zero, R, b, K, C))
    np.testing.assert_allclose([fx[0, 0], fy[0, 0], bfx[0], bfy[0], torque[0]], [0.0, -K * 0.05, 0.0, K * 0.05, 0.0], atol=tol(backend) * K)


def test_box_circles(backend: Backend) -> None:
    # a circle of radius 0.5 whose centre is 0.4 to the right of the box's right face: overlap 0.1, pushing the box to -x
    b = box(backend)
    fx, fy, torque = host(
        backend,
        *contacts.box_circles(
            b, arr(backend, [[1.4, 0.0]]), arr(backend, [[0.3, 0.0]]), arr(backend, [[0.5, 9.0]]), arr(backend, [[1.0, 0.0]]), K, C
        ),
    )
    np.testing.assert_allclose([fx[0], fy[0]], [-K * 0.1, 0.0], atol=tol(backend) * K)
    # applied at (1, 0.3): r × f = 1 * 0 - 0.3 * (-K 0.1)
    np.testing.assert_allclose(torque[0], 0.3 * K * 0.1, atol=tol(backend) * K)


def test_box_rects_corner_into_face(backend: Backend) -> None:
    """The box turned 45°: its right corner pokes 0.1 into a wall-like rectangle's left face, pushing the box to -x."""
    angle = math.pi / 4
    b = box(backend, angle=angle)
    corner = 1.0 * math.cos(angle) + 0.6 * math.sin(angle)  # the farthest x of the box's corners
    face = corner - 0.1
    fx, fy, torque = host(
        backend,
        *contacts.box_rects(
            b,
            arr(backend, [[face + 1.0]]),
            arr(backend, [[0.0]]),
            arr(backend, [[1.0]]),
            arr(backend, [[3.0]]),
            arr(backend, [[1.0]]),
            K,
            C,
        ),
    )
    np.testing.assert_allclose([fx[0], fy[0]], [-K * 0.1, 0.0], atol=tol(backend) * K)
    # the corner is at (corner, y_c) with y_c = sin(45°) - 0.6 cos(45°): torque = -y_c * fx
    yc = 1.0 * math.sin(angle) - 0.6 * math.cos(angle)
    np.testing.assert_allclose(torque[0], -yc * (-K * 0.1), atol=tol(backend) * K)


def test_box_rects_edge_to_edge(backend: Backend) -> None:
    """The box's right face 0.05 inside a tall rectangle's left face: two corners penetrate, the push is along -x, no torque."""
    b = box(backend)
    fx, fy, torque = host(
        backend,
        *contacts.box_rects(
            b, arr(backend, [[1.95]]), arr(backend, [[0.0]]), arr(backend, [[1.0]]), arr(backend, [[3.0]]), arr(backend, [[1.0]]), K, C
        ),
    )
    np.testing.assert_allclose([fx[0], fy[0], torque[0]], [-2 * K * 0.05, 0.0, 0.0], atol=tol(backend) * K)


def test_box_rects_obstacle_corner_into_box(backend: Backend) -> None:
    """The box turned 45°; a square's corner pokes 0.1 into the middle of its top face: the box is pushed straight away from
    the corner, along the face's normal, with no torque (the contact is on the face's centre line)."""
    c45 = math.sqrt(0.5)
    b = box(backend, angle=math.pi / 4)
    nx, ny = -c45, c45  # the top face's outward normal
    px, py = 0.5 * nx, 0.5 * ny  # the square's bottom-right corner, 0.1 inside the face at distance 0.6
    fx, fy, torque = host(
        backend,
        *contacts.box_rects(
            b,
            arr(backend, [[px - 0.3]]),
            arr(backend, [[py + 0.3]]),
            arr(backend, [[0.3]]),
            arr(backend, [[0.3]]),
            arr(backend, [[1.0]]),
            K,
            C,
        ),
    )
    np.testing.assert_allclose([fx[0], fy[0], torque[0]], [-K * 0.1 * nx, -K * 0.1 * ny, 0.0], atol=tol(backend) * K)


def test_box_rects_corner_to_corner_counts_both_corners(backend: Backend) -> None:
    """Corners overlapping each other: the box's corner is inside the square and the square's corner inside the box, and the
    model counts both: the overlap [0.55, 1] × [0.5, 0.6] is pushed out twice through the shallow faces."""
    b = box(backend)
    fx, fy, _ = host(
        backend,
        *contacts.box_rects(
            b, arr(backend, [[0.85]]), arr(backend, [[0.8]]), arr(backend, [[0.3]]), arr(backend, [[0.3]]), arr(backend, [[1.0]]), K, C
        ),
    )
    np.testing.assert_allclose([fx[0], fy[0]], [0.0, -2 * K * 0.1], atol=tol(backend) * K)


def test_box_rects_padding_and_separation(backend: Backend) -> None:
    b = box(backend)
    far = contacts.box_rects(
        b, arr(backend, [[5.0]]), arr(backend, [[5.0]]), arr(backend, [[1.0]]), arr(backend, [[1.0]]), arr(backend, [[1.0]]), K, C
    )
    padded = contacts.box_rects(
        b, arr(backend, [[0.0]]), arr(backend, [[0.0]]), arr(backend, [[1.0]]), arr(backend, [[1.0]]), arr(backend, [[0.0]]), K, C
    )
    for got in (host(backend, *far), host(backend, *padded)):
        np.testing.assert_array_equal(np.concatenate(got), 0.0)


def test_box_walls(backend: Backend) -> None:
    b = box(backend, x=10.0 - 0.95)  # its right face 0.05 past the wall at x = 10: two corners
    fx, fy, torque = host(backend, *contacts.box_walls(b, 10.0, K, C))
    np.testing.assert_allclose([fx.item(), fy.item(), torque.item()], [-2 * K * 0.05, 0.0, 0.0], atol=tol(backend) * K)


@pytest.mark.parametrize("closing", [1.0, -1.0])
def test_damping_acts_along_the_normal(backend: Backend, closing: float) -> None:
    """Closing in, the damper adds to the spring; moving apart, it subtracts (and the force never pulls)."""
    x, y = arr(backend, [0.0, 0.45]), arr(backend, [0.0, 0.0])
    vx, vy = arr(backend, [closing, -closing]), arr(backend, [0.0, 0.0])
    fx, _ = host(backend, *contacts.agent_agent(x, y, vx, vy, R, K, C))
    expected = max(K * 0.05 + C * 2 * closing, 0.0)
    np.testing.assert_allclose(fx, [-expected, expected], atol=tol(backend) * K)
