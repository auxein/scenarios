"""The geometry helpers against hand-computed configurations, on every backend."""

import math

import numpy as np
import pytest
from auxein.backend import Backend

from auxein_scenarios.common.geometry import (
    aabb_contact,
    from_frame,
    half_plane_distance,
    obb_contact,
    obb_corners,
    rotate,
    to_frame,
    wrap_angle,
)


def tolerance(backend: Backend) -> float:
    return 1e-12 if backend.precision == "float64" else 1e-6


def values(backend: Backend, *arrays: object) -> list[np.ndarray]:
    return [np.asarray(backend.to_numpy(a), dtype=np.float64) for a in arrays]


def test_rotation_and_frames_are_inverse(backend: Backend) -> None:
    a = backend.asarray(np.array([0.3, -1.2, 2.0]))
    x, y = backend.asarray(np.array([1.0, 0.5, -2.0])), backend.asarray(np.array([0.0, 3.0, 1.5]))
    c, s = backend.asarray(np.cos(backend.to_numpy(a))), backend.asarray(np.sin(backend.to_numpy(a)))
    ox, oy = backend.asarray(np.array([1.0, -1.0, 0.5])), backend.asarray(np.array([2.0, 0.0, -0.5]))
    lx, ly = to_frame(x, y, ox, oy, c, s)
    bx, by = from_frame(lx, ly, ox, oy, c, s)
    got = values(backend, bx, by)
    np.testing.assert_allclose(got[0], [1.0, 0.5, -2.0], atol=tolerance(backend))
    np.testing.assert_allclose(got[1], [0.0, 3.0, 1.5], atol=tolerance(backend))
    rx, ry = values(backend, *rotate(backend.asarray(np.array([1.0])), backend.asarray(np.array([0.0])), c[:1], s[:1]))
    np.testing.assert_allclose([rx[0], ry[0]], [math.cos(0.3), math.sin(0.3)], atol=tolerance(backend))


def test_wrap_angle(backend: Backend) -> None:
    got = values(backend, wrap_angle(backend.asarray(np.array([0.0, 3 * math.pi / 2, -5 * math.pi / 2, 7.0]))))[0]
    np.testing.assert_allclose(got, [0.0, -math.pi / 2, -math.pi / 2, 7.0 - 2 * math.pi], atol=10 * tolerance(backend))


@pytest.mark.parametrize(
    ("point", "normal", "distance", "closest"),
    [
        ((3.0, 0.0), (1.0, 0.0), 2.0, (1.0, 0.0)),  # outside a face
        ((2.0, 2.0), (math.sqrt(0.5), math.sqrt(0.5)), math.sqrt(2.0), (1.0, 1.0)),  # outside a corner
        ((0.0, -1.5), (0.0, -1.0), 0.5, (0.0, -1.0)),
        ((0.5, 0.1), (1.0, 0.0), -0.5, (1.0, 0.1)),  # inside: the nearest face is x
        ((-0.2, 0.9), (0.0, 1.0), -0.1, (-0.2, 1.0)),  # inside: the nearest face is y
        ((1.0, 0.3), (1.0, 0.0), 0.0, (1.0, 0.3)),  # on the boundary
    ],
)
def test_aabb_contact(
    backend: Backend, point: tuple[float, float], normal: tuple[float, float], distance: float, closest: tuple[float, float]
) -> None:
    def a(v: float) -> object:
        return backend.asarray(np.array([v]))

    got = values(backend, *aabb_contact(a(point[0] + 2.0), a(point[1] - 1.0), a(2.0), a(-1.0), a(1.0), a(1.0)))
    np.testing.assert_allclose([g[0] for g in got], [*normal, distance, closest[0] + 2.0, closest[1] - 1.0], atol=tolerance(backend))


def test_obb_contact_rotated(backend: Backend) -> None:
    """A 4 × 1 rectangle turned a quarter turn: a point 3 units above its centre is 1 unit from its end."""

    def a(v: float) -> object:
        return backend.asarray(np.array([v]))

    got = values(backend, *obb_contact(a(1.0), a(5.0), a(1.0), a(2.0), a(0.0), a(1.0), 2.0, 0.5))
    np.testing.assert_allclose([g[0] for g in got], [0.0, 1.0, 1.0, 1.0, 4.0], atol=10 * tolerance(backend))
    # beside it: 0.5 from its long side, which faces -x and +x in the world
    got = values(backend, *obb_contact(a(2.0), a(2.5), a(1.0), a(2.0), a(0.0), a(1.0), 2.0, 0.5))
    np.testing.assert_allclose([g[0] for g in got], [1.0, 0.0, 0.5, 1.5, 2.5], atol=10 * tolerance(backend))


def test_obb_corners(backend: Backend) -> None:
    angle = math.pi / 6
    xs, ys = obb_corners(
        backend.asarray(np.array([1.0])),
        backend.asarray(np.array([-1.0])),
        backend.asarray(np.array([math.cos(angle)])),
        backend.asarray(np.array([math.sin(angle)])),
        1.0,
        0.5,
    )
    gx, gy = values(backend, xs, ys)
    assert gx.shape == (1, 4)
    local = [(1.0, 0.5), (-1.0, 0.5), (-1.0, -0.5), (1.0, -0.5)]
    expected_x = [1.0 + math.cos(angle) * u - math.sin(angle) * v for u, v in local]
    expected_y = [-1.0 + math.sin(angle) * u + math.cos(angle) * v for u, v in local]
    np.testing.assert_allclose(gx[0], expected_x, atol=10 * tolerance(backend))
    np.testing.assert_allclose(gy[0], expected_y, atol=10 * tolerance(backend))


def test_half_plane_distance(backend: Backend) -> None:
    got = values(
        backend, half_plane_distance(backend.asarray(np.array([3.0, -1.0])), backend.asarray(np.array([0.0, 0.0])), 1.0, 0.0, 2.0)
    )[0]
    np.testing.assert_allclose(got, [1.0, -3.0], atol=tolerance(backend))
