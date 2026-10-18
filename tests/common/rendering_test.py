"""The montage pieces of `common.rendering`: sizes, the text a frame shows, the title card and the curve inset."""

import numpy as np
import pytest

from auxein_scenarios.common.rendering import CurveInset, TitleCard, drawn_text, figure, parse_size


def test_parse_size() -> None:
    assert parse_size("1920x1080") == (1920, 1080)
    assert parse_size("1280X720") == (1280, 720)
    for bad in ("1920", "1920x", "x1080", "1921x1080", "100x100", "big"):
        with pytest.raises(ValueError):
            parse_size(bad)


def test_the_title_card_shows_and_hides() -> None:
    fig = figure(6.4, 3.6)
    fig.text(0.1, 0.9, "under the card")
    card = TitleCard(fig)
    assert "Generation 50" not in drawn_text(fig)
    card.show("milestone 3 of 9", "Generation 50", "score 1.287")
    shown = drawn_text(fig)
    assert "Generation 50" in shown and "score 1.287" in shown
    card.hide()
    assert drawn_text(fig) == ["under the card"]


def test_the_inset_marker_moves_along_the_curve() -> None:
    fig = figure(6.4, 3.6)
    inset = CurveInset(fig, (0.1, 0.1, 0.8, 0.3), [1, 2, 3, 4], [0.0, 1.0, 1.0, 3.0], marks=[1, 4], label="best so far")
    inset.mark(2.5)
    x, y = (np.asarray(v, dtype=np.float64).tolist() for v in inset.marker.get_data())
    assert (x, y) == ([2.5], [1.0])
    assert any("B" in t for t in drawn_text(fig)), "its label is drawn"
