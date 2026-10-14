"""The visual theme of every video, gallery and chart the scenarios produce: monochrome silver on near-black.

One place for the look, so that the demos of M1a, the milestone clips and the montage of M3 and the charts all match:

- **Colour.** A near-black background and a strictly monochrome scale of whites, silvers and steel greys, with no accent
  colour. **State is shown with brightness, never with hue**: what is active (a moving box, an agent touching it) is bright
  white; what is idle is steel; what is background is slate.
- **Agents** are drawn as chevrons pointing along their heading, inside a faint outline of their true (collision) disc, and
  agents close to each other are joined by faint lines whose opacity grows as they get closer.
- **Type.** Headlines in Inter Tight Light, large and tight; small uppercase labels in Inter Bold with wide letter spacing
  (`tracked`); body text and readouts in Inter. The fonts are bundled (`fonts/`, SIL Open Font License), so a video looks the
  same on every machine, CI included, instead of falling back to whatever sans-serif matplotlib finds.
- **Depth.** The floor of a scene is lit slightly at its centre and falls off to the background at its edges (`floor`).
"""

# matplotlib's stubs type imshow's **kwargs as unknown: only that check is off here.
# pyright: reportUnknownMemberType=false

from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.font_manager import FontProperties

__all__ = [
    "BACKGROUND",
    "CHEVRON",
    "FONTS",
    "LINE",
    "LINK_ALPHA",
    "MUTED",
    "SILVER",
    "SLATE",
    "STEEL",
    "SUBTLE",
    "TEXT",
    "Role",
    "floor",
    "font",
    "rgba",
    "style_axes",
    "tracked",
]

BACKGROUND = "#07080a"
"""Near-black: the figure, the floor's edges."""
SLATE = "#151a21"
"""Obstacles and other solid scenery: just above the background."""
LINE = "#2a313b"
"""Hairlines and edges of scenery (a touch lighter than the site's #20262f, which disappears at video size)."""
STEEL = "#8a929c"
"""Idle things: a box at rest, walls, outlines."""
SUBTLE = "#727d89"
"""Quiet text: readouts, secondary captions."""
MUTED = "#aab4bf"
"""Body text: the lead under a headline."""
SILVER = "#c7cdd4"
"""Agents, labels, the target."""
TEXT = "#f4f6f8"
"""Bright white: headlines, and everything active (a moving box, an agent touching it, a delivery)."""

LINK_ALPHA = 0.22
"""The opacity of the line between two agents at zero distance; it falls linearly to 0 at the linking range."""

CHEVRON = ((1.0, 0.0), (-0.67, 0.5), (-0.33, 0.0), (-0.67, -0.5))
"""An agent's glyph, in units of its radius, pointing along +x (its heading)."""

FONTS = Path(__file__).parent / "fonts"
"""The bundled font files."""

Role = Literal["headline", "lead", "label", "body", "readout"]
_FILES: dict[Role, str] = {
    "headline": "InterTight-Light.ttf",
    "lead": "Inter-Light.ttf",
    "label": "Inter-Bold.ttf",
    "body": "Inter-Regular.ttf",
    "readout": "Inter-Regular.ttf",
}


@cache
def _font_file(role: Role) -> str:
    path = FONTS / _FILES[role]
    if not path.exists():
        raise FileNotFoundError(f"the theme's font {path.name} is missing from {FONTS}")
    return str(path)


def font(role: Role, size: float) -> FontProperties:
    """The font of a text role at `size` points, from the bundled files (no system font lookup)."""
    return FontProperties(fname=_font_file(role), size=size)


def tracked(text: str, *, upper: bool = True) -> str:
    """`text` with wide letter spacing, for labels: thin spaces between letters (three between words), uppercased by default.

    matplotlib has no letter-spacing, so the spacing is made of U+2009 thin spaces, which the bundled fonts have."""
    words = (text.upper() if upper else text).split(" ")
    return "   ".join(" ".join(word) for word in words)


def rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    """A theme colour with an opacity."""
    return to_rgba(color, alpha)


def style_axes(ax: Axes) -> None:
    """An axes in the theme: the background colour, no ticks, no frame."""
    ax.set_facecolor(BACKGROUND)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def floor(ax: Axes, extent: tuple[float, float, float, float], *, glow: str = "#0f1217", resolution: int = 512) -> None:
    """A floor lit at its centre: `glow` there, falling smoothly to the background at the corners, under everything else."""
    centre, edge = np.asarray(to_rgb(glow)), np.asarray(to_rgb(BACKGROUND))
    u = np.linspace(-1.0, 1.0, resolution)
    distance: npt.NDArray[np.float64] = np.sqrt(u[None, :] ** 2 + u[:, None] ** 2) / np.sqrt(2.0)
    weight = np.clip(1.0 - distance, 0.0, 1.0) ** 1.5
    # the gradient spans only a few 8-bit levels, so it would show as rings: a fixed, fine dither breaks them up
    dither = np.random.default_rng(0).uniform(-0.5, 0.5, (resolution, resolution, 1)) / 255.0
    image = np.clip(edge + (centre - edge) * weight[..., None] + dither, 0.0, 1.0)
    ax.imshow(image, extent=extent, origin="lower", interpolation="nearest", zorder=0, aspect="auto")
