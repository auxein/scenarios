"""The theme: the bundled fonts load with the right faces, labels are tracked, and the floor is drawn without error."""

from matplotlib.ft2font import FT2Font

from auxein_scenarios.common import theme
from auxein_scenarios.common.rendering import figure


def test_bundled_fonts_have_the_expected_faces() -> None:
    expected = {"headline": ("Inter Tight", "Light"), "lead": ("Inter", "Light"), "label": ("Inter", "Bold"), "body": ("Inter", "Regular")}
    for role, (family, style) in expected.items():
        properties = theme.font(role, 10)  # type: ignore[arg-type]
        face = FT2Font(str(properties.get_file()))
        assert (face.family_name, face.style_name) == (family, style), role
        assert ord(" ") in face.get_charmap()  # the thin space that `tracked` relies on
    assert (theme.FONTS / "OFL-Inter.txt").exists() and (theme.FONTS / "OFL-InterTight.txt").exists()


def test_tracked_labels() -> None:
    assert theme.tracked("at rest") == "A T   R E S T"
    assert theme.tracked("ok", upper=False) == "o k"


def test_floor_and_axes() -> None:
    fig = figure(2.0, 2.0, 50)
    ax = fig.add_subplot()
    theme.style_axes(ax)
    theme.floor(ax, (-1.0, 1.0, -1.0, 1.0), resolution=32)
    assert len(ax.images) == 1 and ax.get_xticks().size == 0
    assert theme.rgba(theme.SILVER, 0.5)[3] == 0.5
