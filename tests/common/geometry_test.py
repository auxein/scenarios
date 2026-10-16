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
    ray_aabb,
    ray_circle,
    ray_inside_aabb,
    ray_obb,
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


# --- ray casts ---

MAX = 5.0


def rays(backend: Backend, origins: list[tuple[float, float]], angles: list[float]) -> tuple[object, object, object, object]:
    """Origins and unit directions as backend arrays, one ray per entry."""
    o = np.asarray(origins, dtype=np.float64)
    a = np.asarray(angles, dtype=np.float64)
    return tuple(backend.asarray(v) for v in (o[:, 0], o[:, 1], np.cos(a), np.sin(a)))  # type: ignore[return-value]


def test_ray_circle(backend: Backend) -> None:
    ox, oy, dx, dy = rays(
        backend,
        [(-3.0, 0.0), (-3.0, 1.0), (-3.0, 1.0 + 1e-3), (3.0, 0.0), (0.5, 0.0), (-8.0, 0.0), (-3.0, 0.0)],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2],
    )
    c = backend.asarray(np.zeros(7))
    one = backend.asarray(np.ones(7))
    (got,) = values(backend, ray_circle(ox, oy, dx, dy, c, c, one, MAX))
    # straight in: 2; grazing the top: tangent at 3; just above: no hit; pointing away: no hit; from inside: 0;
    # the circle beyond the maximum range: no hit; passing beside: no hit
    np.testing.assert_allclose(got, [2.0, 3.0, MAX, MAX, 0.0, MAX, MAX], atol=1e-5)


def test_ray_aabb(backend: Backend) -> None:
    # a 2 × 1 rectangle centred at (2, 0): x in [1, 3], y in [-0.5, 0.5]
    ox, oy, dx, dy = rays(
        backend,
        [(0.0, 0.0), (0.0, 0.5), (0.0, 0.6), (2.0, 3.0), (2.0, 0.0), (4.0, 0.0), (0.0, -2.0), (-5.0, 0.0)],
        [0.0, 0.0, 0.0, -math.pi / 2, 0.0, 0.0, math.atan2(1.5, 1.0), 0.0],
    )
    z = backend.asarray(np.zeros(8))
    cx, hx, hy = backend.asarray(np.full(8, 2.0)), backend.asarray(np.full(8, 1.0)), backend.asarray(np.full(8, 0.5))
    (got,) = values(backend, ray_aabb(ox, oy, dx, dy, cx, z, hx, hy, MAX))
    # straight in: 1; along the top edge: grazing hits at 1; just above: no hit; from above, down: 2.5; from inside: 0;
    # pointing away: no hit; diagonally into the bottom-left corner (1, -0.5): its distance √(1 + 2.25); beyond range: no hit
    np.testing.assert_allclose(got, [1.0, 1.0, MAX, 2.5, 0.0, MAX, math.sqrt(3.25), MAX], atol=1e-5)


def test_ray_obb_is_the_aabb_in_the_box_frame(backend: Backend) -> None:
    # a 2 × 1 box at the origin turned by 90°: it spans x in [-0.5, 0.5], y in [-1, 1]
    ox, oy, dx, dy = rays(backend, [(-3.0, 0.0), (-3.0, 0.9), (-3.0, 1.1), (0.0, -4.0), (0.0, 0.0)], [0.0, 0.0, 0.0, math.pi / 2, 1.0])
    z = backend.asarray(np.zeros(5))
    c, s = backend.asarray(np.zeros(5)), backend.asarray(np.ones(5))
    (got,) = values(backend, ray_obb(ox, oy, dx, dy, z, z, c, s, 1.0, 0.5, MAX))
    np.testing.assert_allclose(got, [2.5, 2.5, MAX, 3.0, 0.0], atol=1e-5)
    # at 45°: a point (x, 0) is (x/√2, −x/√2) in the box's frame, inside while |x|/√2 ≤ 0.5 (the half-height binds first),
    # so a ray along the x axis from (−4, 0) enters at x = −0.5 √2
    angle = math.pi / 4
    ox, oy, dx, dy = rays(backend, [(-4.0, 0.0)], [0.0])
    zero = backend.asarray(np.zeros(1))
    (got,) = values(backend, ray_obb(ox, oy, dx, dy, zero, zero, zero + math.cos(angle), zero + math.sin(angle), 1.0, 0.5, MAX))
    np.testing.assert_allclose(got, [4.0 - 0.5 * math.sqrt(2.0)], atol=1e-5)


def test_ray_inside_aabb_sees_the_walls_from_within(backend: Backend) -> None:
    ox, oy, dx, dy = rays(
        backend, [(0.0, 0.0), (8.0, 0.0), (9.0, 9.0), (0.0, 0.0), (11.0, 0.0)], [0.0, 0.0, math.pi / 4, math.pi / 2, math.pi]
    )
    (got,) = values(backend, ray_inside_aabb(ox, oy, dx, dy, 10.0, MAX))
    # the wall 10 away is beyond range; 2 away; into the corner diagonally: √2; up 10: beyond range; from outside: 0
    np.testing.assert_allclose(got, [MAX, 2.0, math.sqrt(2.0), MAX, 0.0], atol=1e-5)


def test_rays_broadcast_over_a_trailing_ray_axis(backend: Backend) -> None:
    """Rays `(agents, R)` against one circle: each agent's fan of rays."""
    angles = np.linspace(0.0, 2 * math.pi, 8, endpoint=False)
    ox = backend.asarray(np.array([[-3.0], [0.0]]))
    oy = backend.asarray(np.array([[0.0], [-3.0]]))
    dx, dy = backend.asarray(np.cos(angles)[None]), backend.asarray(np.sin(angles)[None])
    zero, one = backend.asarray(np.zeros(1)), backend.asarray(np.ones(1))
    (got,) = values(backend, ray_circle(ox, oy, dx, dy, zero, zero, one, MAX))
    assert got.shape == (2, 8)
    assert got[0, 0] == pytest.approx(2.0, abs=1e-5) and got[1, 2] == pytest.approx(2.0, abs=1e-5)
    assert (got[0, 1:] == MAX).sum() >= 6
