"""Milestone clips, both montages, the charts and the held-out showcase picks (design doc §7), on a smoke run.

Clips render a few frames each (a large speed-up, a low frame rate); what they say is checked from the text drawn in every
frame (`drawn_text`), not from pixels."""

import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from matplotlib.figure import Figure

from auxein_scenarios.box_pushing import render, report, showcase, train
from auxein_scenarios.box_pushing.config import load_config
from auxein_scenarios.box_pushing.milestones import find_milestones
from auxein_scenarios.common.rendering import drawn_text

ROOT = Path(__file__).resolve().parents[2]
SMOKE = load_config(ROOT / "configs" / "smoke.toml")


def quiet(_: str) -> None:
    pass


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("showcase") / "smoke"
    train.train(SMOKE, path, out=quiet, progress=False, estimate=False)
    return path


class TextSpy:
    """Wraps `render.write_mp4`: the clip is still written, and the text drawn in each frame is kept."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.frames: list[list[str]] = []
        real = render.write_mp4

        def spy(fig: Figure, update: Callable[[Any], None], frames: Sequence[Any], path: str | Path, fps: float = 30.0) -> Path:
            def recorded(value: Any) -> None:
                update(value)
                self.frames.append(drawn_text(fig))

            return real(fig, recorded, frames, path, fps)

        monkeypatch.setattr(render, "write_mp4", spy)

    @property
    def text(self) -> str:
        """Everything drawn, with tracked labels read back as plain words (`theme.tracked` spaces letters with U+2009)."""
        return "\n".join(plain(line) for frame in self.frames for line in frame)


def remove_report(run: Path) -> None:
    shutil.rmtree(run / report.REPORT, ignore_errors=True)


def plain(text: str) -> str:
    return text.replace("\u2009\u2009\u2009", " ").replace("\u2009", "")


def test_milestone_clips_label_their_panels_honestly(run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spy = TextSpy(monkeypatch)
    paths = showcase.milestone_clips(run, generations=[3], showcase=["train:0", "held:1"], fps=4, speedup=60, out=quiet)
    assert [p.name for p in paths] == ["gen-0003.mp4"] and paths[0].read_bytes()[4:8] == b"ftyp"
    text = spy.text
    milestone = find_milestones(run / train.RECORDING, [3])[0]
    assert "Generation 3" in text and f"score {milestone.score:.3f}" in text
    assert "TRAINING SCENARIO" in text  # the footers, in tracked capitals
    assert "HELD-OUT: NEVER SEEN IN TRAINING" in text
    assert "delivered at step" in text or "not delivered" in text, "each panel says how its episode ended"
    assert "TRAIN-0000 · EASY" in text and "HELD-0001 · FULL" in text


def test_the_sequence_montage(run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spy = TextSpy(monkeypatch)
    path = showcase.montage(
        run, layout="sequence", size=(640, 360), generations=[1, 3], showcase=["train:0"], fps=2, speedup=100, out=quiet
    )
    assert path.name == "montage-sequence.mp4" and path.read_bytes()[4:8] == b"ftyp"
    milestones = find_milestones(run / train.RECORDING, [1, 3])
    cards = [[plain(t) for t in frame] for frame in spy.frames if any(plain(t).startswith("MILESTONE") for t in frame)]
    assert len(cards) >= len(milestones), "every milestone is opened by a title card"
    for m in milestones:
        assert any(f"Generation {m.generation}" in t for frame in cards for t in frame)
    assert "BEST SCORE SO FAR" in spy.text, "the fitness inset"


def test_the_grid_montage(run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spy = TextSpy(monkeypatch)
    path = showcase.montage(run, layout="grid", size=(640, 360), fps=2, speedup=100, out=quiet)
    assert path.name == "montage-grid.mp4" and path.read_bytes()[4:8] == b"ftyp"
    text = spy.text
    for m in showcase.grid_milestones(find_milestones(run / train.RECORDING)):
        assert f"GENERATION {m.generation}" in text
    assert "HELD-0000 · HELD-OUT: NEVER SEEN IN TRAINING" in text, "the scenario is labelled held-out"


def test_grid_milestones() -> None:
    assert showcase.grid_milestones(list(range(9))) == [0, 3, 5, 8]  # type: ignore[arg-type]
    assert showcase.grid_milestones(list(range(3))) == [0, 1, 2]  # type: ignore[arg-type]


def test_the_charts_and_the_generalisation_chart_without_a_report(run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    real = showcase.line_chart

    def spy(path: str | Path, series: Any, **kwargs: Any) -> tuple[Path, Path]:
        calls.append({"path": Path(path), "series": series, **kwargs})
        return real(path, series, **kwargs)

    monkeypatch.setattr(showcase, "line_chart", spy)
    remove_report(run)
    written = showcase.charts(run, out=quiet)
    assert sorted(p.name for p in written) == sorted(
        f"{n}.{x}" for n in ("fitness", "delivery", "cooperation", "generalisation") for x in ("png", "svg")
    )
    assert all(p.stat().st_size > 1000 for p in written)
    without = next(c for c in calls if c["path"].name == "generalisation")
    assert "No held-out report yet" in without["message"] and not without["series"]
    fitness = next(c for c in calls if c["path"].name == "fitness")
    assert [len(s.y) for s in fitness["series"]] == [3, 3, 3], "one point per generation"

    report.write_report(report.build_report(run, out=quiet))
    calls.clear()
    showcase.charts(run, out=quiet)
    with_report = next(c for c in calls if c["path"].name == "generalisation")
    assert not with_report.get("message") and [s.label for s in with_report["series"]] == ["training (full)", "held-out (unseen)"]


def test_best_and_worst_held_out_scenarios(run: Path) -> None:
    built = report.build_report(run, out=quiet)
    final = built.final
    ranked = [(int(e.delivered), e.progress, int(e.scenario_id.split("-")[-1])) for e in final.held_out]
    expected_best = max(ranked, key=lambda r: (r[0], r[1], -r[2]))[2]
    expected_worst = min(ranked, key=lambda r: (r[0], r[1], r[2]))[2]
    # without a report, the final champion is evaluated on the spot; with one, its rows are read: the same picks
    remove_report(run)
    best, worst = showcase.held_extremes(run, out=quiet)
    assert (best.index, worst.index) == (expected_best, expected_worst) and best.set == worst.set == "held"
    report.write_report(built)
    assert showcase.held_extremes(run, out=quiet) == (best, worst)
    assert showcase.resolve_showcase(run, ["best-held", "train:1", "worst-held"], out=quiet) == [best, train.Showcase("train", 1), worst]
