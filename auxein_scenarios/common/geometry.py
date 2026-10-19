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

**Ray casts** (for the agents' range rays): a ray starts at `(ox, oy)` and runs along the unit direction `(dx, dy)`; every
`ray_*` function returns the distance along it to the first point of a shape, **0 when the ray starts inside the shape**, and
`max_range` when the shape is not hit within `[0, max_range]` ("no hit"). The sensor takes the minimum over the shapes, so a
shape that is farther than another simply loses, and a padded obstacle slot (`valid` 0) returns `max_range` and never wins:
the minimum with it is exact, so padding does not change a reading to the last bit. Rays are typically shaped `(..., R)`, with
the shapes broadcast against them.
"""

from auxein.backend import Array, backend_of

__all__ = [
    "aabb_contact",
    "ray_aabb",
    "ray_circle",
    "ray_inside_aabb",
    "ray_obb",
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


def _slab(o: Array, d: Array, low: Array, high: Array) -> tuple[Array, Array]:
    """The parameter interval `[enter, exit]` in which the ray `o + t d` (one axis) is within `[low, high]`; empty (enter >
    exit) when the ray runs parallel to the slab outside it. Infinite bounds come only from `where`, never from arithmetic."""
    xp = backend_of(o).xp
    parallel = d == 0.0
    inv = 1.0 / xp.where(parallel, 1.0, d)
    t1, t2 = (low - o) * inv, (high - o) * inv
    within = (o >= low) & (o <= high)
    infinity = 0.0 * t1 + xp.inf  # an array, so that the nested `where` keeps the inputs' dtype and device
    enter = xp.where(parallel, xp.where(within, -infinity, infinity), xp.minimum(t1, t2))
    leave = xp.where(parallel, xp.where(within, infinity, -infinity), xp.maximum(t1, t2))
    return enter, leave


def ray_aabb(ox: Array, oy: Array, dx: Array, dy: Array, cx: Array, cy: Array, hx: Array, hy: Array, max_range: float) -> Array:
    """The distance along the ray to the axis-aligned rectangle of centre `(cx, cy)` and half-sides `(hx, hy)` (slab method).

    A ray grazing an edge or a corner hits it; a ray starting inside or on the boundary returns 0."""
    xp = backend_of(ox).xp
    enter_x, leave_x = _slab(ox, dx, cx - hx, cx + hx)
    enter_y, leave_y = _slab(oy, dy, cy - hy, cy + hy)
    enter, leave = xp.maximum(enter_x, enter_y), xp.minimum(leave_x, leave_y)
    hit = (enter <= leave) & (leave >= 0.0) & (enter <= max_range)
    return xp.where(hit, xp.clip(enter, 0.0, None), max_range)


def ray_obb(
    ox: Array,
    oy: Array,
    dx: Array,
    dy: Array,
    cx: Array,
    cy: Array,
    c: Array,
    s: Array,
    hx: Array | float,
    hy: Array | float,
    max_range: float,
) -> Array:
    """`ray_aabb` for an oriented rectangle (centre `(cx, cy)`, orientation `(c, s)`, half-sides along its own axes): the
    ray is taken into the rectangle's frame, where distances are unchanged."""
    lx, ly = to_frame(ox, oy, cx, cy, c, s)
    ldx, ldy = c * dx + s * dy, c * dy - s * dx
    zero = 0.0 * lx
    return ray_aabb(lx, ly, ldx, ldy, zero, zero, zero + hx, zero + hy, max_range)


def ray_circle(ox: Array, oy: Array, dx: Array, dy: Array, cx: Array, cy: Array, r: Array, max_range: float) -> Array:
    """The distance along the ray to the circle of centre `(cx, cy)` and radius `r`: the smaller root of `|o + t d − c| = r`.

    A tangent ray hits (one double root); a ray starting inside or on the circle returns 0."""
    xp = backend_of(ox).xp
    px, py = ox - cx, oy - cy
    b = px * dx + py * dy
    c = px * px + py * py - r * r
    discriminant = b * b - c
    t = -b - xp.sqrt(xp.clip(discriminant, 0.0, None))
    inside = c <= 0.0
    hit = inside | ((discriminant >= 0.0) & (t >= 0.0) & (t <= max_range))
    return xp.where(hit, xp.where(inside, 0.0, t), max_range)


def ray_inside_aabb(ox: Array, oy: Array, dx: Array, dy: Array, half: float, max_range: float) -> Array:
    """The distance along a ray from inside the square `[-half, half]²` to its boundary: the walls of the arena, seen from
    within (a ray always leaves the square, so it hits unless the walls are beyond `max_range`). From outside, 0."""
    xp = backend_of(ox).xp
    _, leave_x = _slab(ox, dx, -half + 0.0 * ox, half + 0.0 * ox)
    _, leave_y = _slab(oy, dy, -half + 0.0 * oy, half + 0.0 * oy)
    leave = xp.clip(xp.minimum(leave_x, leave_y), 0.0, None)
    outside = (xp.abs(ox) > half) | (xp.abs(oy) > half)
    return xp.where(outside, 0.0, xp.clip(leave, None, max_range))
