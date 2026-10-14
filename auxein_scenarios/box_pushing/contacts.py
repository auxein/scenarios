"""The penalty contact model (design doc §2.6): forces and torques from overlaps, for every contact pair at once.

A contact between two bodies that overlap by `δ > 0` produces a spring–damper force `k δ + c δ̇` along the contact normal,
**only pushing**: the magnitude is clamped at zero, so a damper never pulls bodies together as they separate. `δ̇` is the rate at
which the overlap grows, minus the relative normal velocity. The force is applied at the contact point, so on the box it makes
a torque about the box's centre.

The pairs, each a function here, returning forces (and, for the box, torques about its centre) in the world frame:

| Pair | Function | Note |
|---|---|---|
| disc–disc | `agent_agent`, `agent_circles` | agents among themselves (the `N²` pair), agents against circles |
| disc–axis-aligned rectangle | `agent_rects` | the closest point by clamping |
| disc–oriented rectangle | `agent_box` | the same, in the box's frame; the reaction goes to the box |
| oriented box–circle | `box_circles` | the circle as a disc against the box |
| oriented box–axis-aligned rectangle | `box_rects` | corner penetration both ways (below) |
| everything–wall | `agent_walls`, `box_walls` | half-planes; the box by its four corners |

**Dense arrays reduced with sums, never scatters** (design doc §2.1), so results are deterministic on every device. Agent pairs
are `(..., N, N)` arrays summed over the partner axis. Obstacles are padded to a fixed count per batch with a validity mask, so
layouts with different numbers of obstacles share arrays; their contributions are **accumulated slot by slot**, in a Python loop
over the (few) obstacle slots, never over candidates, scenarios or agents. That makes padding exact: a padded slot adds an exact
zero after the real ones, so a scenario gives the same numbers, to the last bit, whether its obstacles are padded to 6 or to 10.

**The box against rectangles** is the hardest pair: two rectangles at an angle meet corner-to-edge or edge-to-edge. It is
computed as corner penetration both ways: each of the box's corners inside the obstacle is pushed out through the obstacle's
nearest face, and each of the obstacle's corners inside the box pushes the box away through the box's nearest face, at the
corner. That is exact for corner–edge contacts and covers edge–edge ones (two corners penetrate); a crossing with no corner
inside either body is missed, but it can only be reached through a corner contact first, which the stiff box contact resists.
"""

from dataclasses import dataclass

from auxein.backend import Array, backend_of

from auxein_scenarios.common.geometry import aabb_contact, obb_contact, obb_corners

__all__ = [
    "BoxKinematics",
    "agent_agent",
    "agent_box",
    "agent_circles",
    "agent_rects",
    "agent_walls",
    "box_circles",
    "box_rects",
    "box_walls",
    "penalty",
]


@dataclass(frozen=True)
class BoxKinematics:
    """The box's pose and velocity, as the contacts need them: arrays of the batch's leading shape (e.g. `(n, s)`).

    `c` and `s` are the cosine and sine of its angle, computed once per substep; `hx` and `hy` its half-sides (numbers)."""

    x: Array
    y: Array
    c: Array
    s: Array
    vx: Array
    vy: Array
    spin: Array
    hx: float
    hy: float

    def point_velocity(self, px: Array, py: Array) -> tuple[Array, Array]:
        """The velocity of the box's material point at `(px, py)`: `v + ω × r`. `px`, `py` may have extra trailing axes."""
        x, y, vx, vy, w = (_trailing(a, px) for a in (self.x, self.y, self.vx, self.vy, self.spin))
        return vx - w * (py - y), vy + w * (px - x)

    def torque(self, px: Array, py: Array, fx: Array, fy: Array) -> Array:
        """The torque about the box's centre of a force `(fx, fy)` applied at `(px, py)`: `r × f`."""
        rx, ry = px - _trailing(self.x, px), py - _trailing(self.y, px)
        return rx * fy - ry * fx


def _trailing(a: Array, like: Array) -> Array:
    """`a` with trailing axes of length 1 added until it has as many axes as `like`, so that it broadcasts against it."""
    extra = like.ndim - a.ndim
    return a if extra <= 0 else a[(..., *([None] * extra))]


def penalty(overlap: Array, growth: Array, stiffness: float, damping: float) -> Array:
    """The magnitude of a penalty contact: `max(0, k δ + c δ̇)` where the bodies overlap (`δ > 0`), else 0.

    `growth` is `δ̇`, the rate at which the overlap grows. Clamping at zero keeps the force pushing only: a damper would
    otherwise pull separating bodies together at the end of a contact."""
    xp = backend_of(overlap).xp
    return xp.where(overlap > 0.0, xp.clip(stiffness * overlap + damping * growth, 0.0, None), 0.0)


def agent_agent(x: Array, y: Array, vx: Array, vy: Array, radius: float, stiffness: float, damping: float) -> tuple[Array, Array]:
    """The contact forces between the agents of each team: `(..., N)` positions and velocities in, `(..., N)` forces out.

    Every pair is computed on `(..., N, N)` arrays and summed over partners. The force on `i` from `j` is exactly minus the
    force on `j` from `i` (the same magnitude, the opposite normal), so contacts conserve momentum to rounding. Two agents at
    exactly the same point get the normal ±x by index, which keeps that antisymmetry."""
    xp = backend_of(x).xp
    n_agents = x.shape[-1]
    dx = x[..., :, None] - x[..., None, :]
    dy = y[..., :, None] - y[..., None, :]
    distance = xp.sqrt(dx * dx + dy * dy)
    index = xp.arange(n_agents, device=backend_of(x).device)
    order = index[:, None] - index[None, :]
    other = xp.astype(order != 0, x.dtype)  # 0 on the diagonal: an agent does not touch itself
    fallback = xp.astype(xp.sign(order), x.dtype)
    apart = distance > 0.0
    safe = xp.where(apart, distance, 1.0)
    nx = xp.where(apart, dx / safe, fallback)
    ny = xp.where(apart, dy / safe, 0.0)
    growth = -((vx[..., :, None] - vx[..., None, :]) * nx + (vy[..., :, None] - vy[..., None, :]) * ny)
    magnitude = penalty(2.0 * radius - distance, growth, stiffness, damping) * other
    return xp.sum(magnitude * nx, axis=-1), xp.sum(magnitude * ny, axis=-1)


def agent_circles(
    x: Array,
    y: Array,
    vx: Array,
    vy: Array,
    radius: float,
    cx: Array,
    cy: Array,
    cr: Array,
    valid: Array,
    stiffness: float,
    damping: float,
) -> tuple[Array, Array]:
    """The forces on agents `(..., N)` from circular obstacles `(..., O)` (centres, radii, validity), padded slots included."""
    xp = backend_of(x).xp
    fx, fy = 0.0 * x, 0.0 * y
    for slot in range(cx.shape[-1]):
        ox, oy, r, ok = (a[..., slot, None] for a in (cx, cy, cr, valid))
        dx, dy = x - ox, y - oy
        distance = xp.sqrt(dx * dx + dy * dy)
        apart = distance > 0.0
        safe = xp.where(apart, distance, 1.0)
        nx, ny = xp.where(apart, dx / safe, 1.0), xp.where(apart, dy / safe, 0.0)
        magnitude = penalty(radius + r - distance, -(vx * nx + vy * ny), stiffness, damping) * ok
        fx, fy = fx + magnitude * nx, fy + magnitude * ny
    return fx, fy


def agent_rects(
    x: Array,
    y: Array,
    vx: Array,
    vy: Array,
    radius: float,
    cx: Array,
    cy: Array,
    hx: Array,
    hy: Array,
    valid: Array,
    stiffness: float,
    damping: float,
) -> tuple[Array, Array]:
    """The forces on agents `(..., N)` from axis-aligned rectangular obstacles `(..., O)` (centres, half-sides, validity)."""
    fx, fy = 0.0 * x, 0.0 * y
    for slot in range(cx.shape[-1]):
        ox, oy, ex, ey, ok = (a[..., slot, None] for a in (cx, cy, hx, hy, valid))
        nx, ny, distance, _, _ = aabb_contact(x, y, ox, oy, ex, ey)
        magnitude = penalty(radius - distance, -(vx * nx + vy * ny), stiffness, damping) * ok
        fx, fy = fx + magnitude * nx, fy + magnitude * ny
    return fx, fy


def agent_walls(
    x: Array, y: Array, vx: Array, vy: Array, radius: float, half: float, stiffness: float, damping: float
) -> tuple[Array, Array]:
    """The forces on agents from the arena's four walls, at `±half` on both axes."""
    left = penalty(radius - (x + half), -vx, stiffness, damping)
    right = penalty(radius - (half - x), vx, stiffness, damping)
    bottom = penalty(radius - (y + half), -vy, stiffness, damping)
    top = penalty(radius - (half - y), vy, stiffness, damping)
    return left - right, bottom - top


def agent_box(
    x: Array, y: Array, vx: Array, vy: Array, radius: float, box: BoxKinematics, stiffness: float, damping: float
) -> tuple[Array, Array, Array, Array, Array]:
    """The contacts between agents `(..., N)` and the box `(...)`: `(agent fx, agent fy, box fx, box fy, box torque)`.

    The force on each agent is along the box's outward normal at the closest point of its surface; the box gets the opposite
    force at that point, summed over the agents, with its torque."""
    xp = backend_of(x).xp
    bx, by, c, s = (a[..., None] for a in (box.x, box.y, box.c, box.s))
    nx, ny, distance, qx, qy = obb_contact(x, y, bx, by, c, s, box.hx, box.hy)
    pvx, pvy = box.point_velocity(qx, qy)
    growth = -((vx - pvx) * nx + (vy - pvy) * ny)
    magnitude = penalty(radius - distance, growth, stiffness, damping)
    fx, fy = magnitude * nx, magnitude * ny
    torque = box.torque(qx, qy, -fx, -fy)
    return fx, fy, -xp.sum(fx, axis=-1), -xp.sum(fy, axis=-1), xp.sum(torque, axis=-1)


def box_circles(
    box: BoxKinematics, cx: Array, cy: Array, cr: Array, valid: Array, stiffness: float, damping: float
) -> tuple[Array, Array, Array]:
    """The force and torque on the box from circular obstacles `(..., O)`: each circle is a disc pressing on the box's surface."""
    fx, fy, torque = 0.0 * box.x, 0.0 * box.y, 0.0 * box.x
    for slot in range(cx.shape[-1]):
        ox, oy, r, ok = (a[..., slot] for a in (cx, cy, cr, valid))
        nx, ny, distance, qx, qy = obb_contact(ox, oy, box.x, box.y, box.c, box.s, box.hx, box.hy)
        pvx, pvy = box.point_velocity(qx, qy)
        magnitude = penalty(r - distance, pvx * nx + pvy * ny, stiffness, damping) * ok
        gx, gy = -magnitude * nx, -magnitude * ny
        fx, fy, torque = fx + gx, fy + gy, torque + box.torque(qx, qy, gx, gy)
    return fx, fy, torque


def box_rects(
    box: BoxKinematics, cx: Array, cy: Array, hx: Array, hy: Array, valid: Array, stiffness: float, damping: float
) -> tuple[Array, Array, Array]:
    """The force and torque on the box from axis-aligned rectangular obstacles `(..., O)`, by corner penetration both ways."""
    xp = backend_of(box.x).xp
    fx, fy, torque = 0.0 * box.x, 0.0 * box.y, 0.0 * box.x
    corners_x, corners_y = obb_corners(box.x, box.y, box.c, box.s, box.hx, box.hy)  # (..., 4)
    for slot in range(cx.shape[-1]):
        ox, oy, ex, ey, ok = (a[..., slot] for a in (cx, cy, hx, hy, valid))
        # the box's corners inside the obstacle: pushed out through the obstacle's nearest face
        nx, ny, distance, _, _ = aabb_contact(corners_x, corners_y, ox[..., None], oy[..., None], ex[..., None], ey[..., None])
        pvx, pvy = box.point_velocity(corners_x, corners_y)
        magnitude = penalty(-distance, -(pvx * nx + pvy * ny), stiffness, damping) * ok[..., None]
        gx, gy = magnitude * nx, magnitude * ny
        torque = torque + xp.sum(box.torque(corners_x, corners_y, gx, gy), axis=-1)
        fx, fy = fx + xp.sum(gx, axis=-1), fy + xp.sum(gy, axis=-1)
        # the obstacle's corners inside the box: they push the box away through the box's nearest face, at the corner
        ocx = xp.stack([ox + ex, ox - ex, ox - ex, ox + ex], axis=-1)
        ocy = xp.stack([oy + ey, oy + ey, oy - ey, oy - ey], axis=-1)
        bx, by, c, s = (a[..., None] for a in (box.x, box.y, box.c, box.s))
        nx, ny, distance, _, _ = obb_contact(ocx, ocy, bx, by, c, s, box.hx, box.hy)
        pvx, pvy = box.point_velocity(ocx, ocy)
        magnitude = penalty(-distance, pvx * nx + pvy * ny, stiffness, damping) * ok[..., None]
        gx, gy = -magnitude * nx, -magnitude * ny
        torque = torque + xp.sum(box.torque(ocx, ocy, gx, gy), axis=-1)
        fx, fy = fx + xp.sum(gx, axis=-1), fy + xp.sum(gy, axis=-1)
    return fx, fy, torque


def box_walls(box: BoxKinematics, half: float, stiffness: float, damping: float) -> tuple[Array, Array, Array]:
    """The force and torque on the box from the four walls, through its four corners."""
    xp = backend_of(box.x).xp
    px, py = obb_corners(box.x, box.y, box.c, box.s, box.hx, box.hy)
    pvx, pvy = box.point_velocity(px, py)
    gx = penalty(-(px + half), -pvx, stiffness, damping) - penalty(px - half, pvx, stiffness, damping)
    gy = penalty(-(py + half), -pvy, stiffness, damping) - penalty(py - half, pvy, stiffness, damping)
    return xp.sum(gx, axis=-1), xp.sum(gy, axis=-1), xp.sum(box.torque(px, py, gx, gy), axis=-1)
