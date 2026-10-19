"""`common.rendering`'s `drawn_text`: the text a frame shows, which tests read instead of comparing pixels."""

from auxein_scenarios.common.rendering import drawn_text, figure


def test_drawn_text_reads_the_figure_and_its_axes() -> None:
    fig = figure(6.4, 3.6)
    fig.text(0.1, 0.9, "a headline")
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.6))
    ax.text(0.5, 0.5, "a caption")
    hidden = ax.text(0.5, 0.2, "hidden")
    hidden.set_visible(False)
    shown = drawn_text(fig)
    assert "a headline" in shown and "a caption" in shown and "hidden" not in shown
    ax.set_visible(False)
    assert "a caption" not in drawn_text(fig)
