"""2-D geometry on batched arrays, written against the Array API namespace of its inputs (numpy or torch, any precision, any device).

Points and vectors are passed as **separate `x` and `y` arrays**, not as arrays with a trailing axis of 2: the physics builds
pairwise arrays such as `(n, s, N, N)` for agent pairs, and a trailing coordinate axis would double their size and force a
`stack` on every operation. Every function broadcasts its arguments like any element-wise operation, so the same code serves
one point, a team of agents or a whole population of teams, with no Python loops.

Orientations are passed as their cosine and sine (`c`, `s`): the physics computes them once per substep and reuses them for
every frame change, instead of calling `cos` and `sin` in every helper.

Everything is element-wise and branch-free (both sides of a case are computed and chosen with `where`), so that results do not
depend on the batch an entry is part of, and divisions are guarded so that no branch produces a NaN that a `where` would then
have to discard (a NaN would still poison gradients and some backends' reductions).
"""

from auxein.backend import Array, backend_of

__all__ = [
    "aabb_contact",
    "from_frame",
    "half_plane_distance",
    "obb_contact",
    "obb_corners",
    "rotate",
    "to_frame",
    "wrap_angle",
]


def rotate(x: Array, y: Array, c: Array, s: Array) -> tuple[Array, Array]:
    """The vector `(x, y)` rotated by the angle whose cosine and sine are `c` and `s`."""
    return c * x - s * y, s * x + c * y


def to_frame(x: Array, y: Array, ox: Array, oy: Array, c: Array, s: Array) -> tuple[Array, Array]:
    """The point `(x, y)` in the frame with origin `(ox, oy)` and orientation `(c, s)`: translated, then rotated by minus the angle."""
    dx, dy = x - ox, y - oy
    return c * dx + s * dy, c * dy - s * dx


def from_frame(x: Array, y: Array, ox: Array, oy: Array, c: Array, s: Array) -> tuple[Array, Array]:
    """The inverse of `to_frame`: a point given in the frame `(ox, oy, c, s)`, in the world frame."""
    rx, ry = rotate(x, y, c, s)
    return rx + ox, ry + oy


def wrap_angle(angle: Array) -> Array:
    """An angle wrapped to `[-π, π]`, through `atan2` so that it is exact at any magnitude and on any backend."""
    xp = backend_of(angle).xp
    return xp.atan2(xp.sin(angle), xp.cos(angle))


def half_plane_distance(x: Array, y: Array, nx: Array, ny: Array, offset: Array | float) -> Array:
    """The signed distance of `(x, y)` from the line `n · p = offset`, positive on the side `n` points to (`n` is a unit vector)."""
    return nx * x + ny * y - offset


def aabb_contact(x: Array, y: Array, cx: Array, cy: Array, hx: Array, hy: Array) -> tuple[Array, Array, Array, Array, Array]:
    """The contact of the point `(x, y)` with the axis-aligned rectangle of centre `(cx, cy)` and half-sides `(hx, hy)`.

    Returns `(nx, ny, distance, qx, qy)`:

    - `distance`, the **signed distance** from the rectangle's surface: positive outside, negative inside (minus the depth to
      the nearest face), zero on the boundary;
    - `(nx, ny)`, the unit **outward normal** at the contact: from the closest surface point toward the point outside, along
      the nearest face's normal inside, so that moving the point along it always reduces its penetration;
    - `(qx, qy)`, the **closest point on the surface**, where a contact force is applied.

    Outside, the closest point is the point clamped to the rectangle. Inside, it is the projection onto the nearest face, the
    direction of least penetration; the choice between two faces at equal depth is the x face, for determinism.
    """
    xp = backend_of(x).xp
    lx, ly = x - cx, y - cy
    kx, ky = xp.clip(lx, -hx, hx), xp.clip(ly, -hy, hy)
    dx, dy = lx - kx, ly - ky
    outside_distance = xp.sqrt(dx * dx + dy * dy)
    inside = outside_distance == 0.0
    safe = xp.where(inside, 1.0, outside_distance)
    # inside (or on the boundary): the face of least depth; a point exactly on an axis goes toward the positive face
    sx = xp.where(lx >= 0.0, 1.0, -1.0)
    sy = xp.where(ly >= 0.0, 1.0, -1.0)
    depth_x, depth_y = hx - xp.abs(lx), hy - xp.abs(ly)
    use_x = depth_x <= depth_y
    nx = xp.where(inside, xp.where(use_x, sx, 0.0), dx / safe)
    ny = xp.where(inside, xp.where(use_x, 0.0, sy), dy / safe)
    distance = xp.where(inside, -xp.minimum(depth_x, depth_y), outside_distance)
    qx = xp.where(inside, xp.where(use_x, sx * hx, lx), kx) + cx
    qy = xp.where(inside, xp.where(use_x, ly, sy * hy), ky) + cy
    return nx, ny, distance, qx, qy


def obb_contact(
    x: Array, y: Array, cx: Array, cy: Array, c: Array, s: Array, hx: Array, hy: Array
) -> tuple[Array, Array, Array, Array, Array]:
    """`aabb_contact` for an oriented rectangle: centre `(cx, cy)`, orientation `(c, s)` and half-sides `(hx, hy)` along its own
    axes. The point is taken into the rectangle's frame, and the normal and the closest point are brought back to the world's."""
    lx, ly = to_frame(x, y, cx, cy, c, s)
    zero = 0.0 * lx
    nx, ny, distance, qx, qy = aabb_contact(lx, ly, zero, zero, hx, hy)
    wnx, wny = rotate(nx, ny, c, s)
    wqx, wqy = from_frame(qx, qy, cx, cy, c, s)
    return wnx, wny, distance, wqx, wqy


def obb_corners(cx: Array, cy: Array, c: Array, s: Array, hx: Array, hy: Array) -> tuple[Array, Array]:
    """The four corners of an oriented rectangle, as `x` and `y` arrays with a new trailing axis of 4, counter-clockwise from
    the corner at `(+hx, +hy)` in the rectangle's own frame."""
    xp = backend_of(cx).xp
    lx = xp.stack([hx + 0.0 * cx, -hx + 0.0 * cx, -hx + 0.0 * cx, hx + 0.0 * cx], axis=-1)
    ly = xp.stack([hy + 0.0 * cy, hy + 0.0 * cy, -hy + 0.0 * cy, -hy + 0.0 * cy], axis=-1)
    return from_frame(lx, ly, cx[..., None], cy[..., None], c[..., None], s[..., None])
