"""Comparing runs on one held-out set (design doc §7.1, M2b): two smoke runs, one per generator version."""

import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from auxein_scenarios.box_pushing import compare, report, showcase, train
from auxein_scenarios.box_pushing.config import dump_config, load_config, parse_config

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent / "data"


def quiet(_: str) -> None:
    pass


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """An older generator-1 run with a shorter step limit (its own world), then a generator-2 run: the newest."""
    base = tmp_path_factory.mktemp("compare")
    old = parse_config((DATA / "v1_smoke.toml").read_text().replace("max_steps = 200", "max_steps = 150"), name="old")
    assert old.scenarios.generator == 1 and old.max_steps == 150
    train.train(old, base / "old", out=quiet, progress=False, estimate=False)
    new = load_config(ROOT / "configs" / "smoke.toml")
    train.train(new, base / "new", out=quiet, progress=False, estimate=False)
    return base / "old", base / "new"


def fingerprint(directory: Path) -> dict[str, str]:
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.rglob("*")) if p.is_file()}


def test_each_champion_runs_in_its_own_world_on_the_common_set(runs: tuple[Path, Path]) -> None:
    old, new = runs
    comparison = compare.build_comparison([old, new], out=quiet)
    assert comparison.reference.run_dir == new, "the newest run's held-out set by default"
    assert comparison.held.fingerprint == train.held_out_set(load_config(new / train.CONFIG_FILE)).fingerprint
    by_run = {c.run_dir: c for c in comparison.runs}
    assert [len(c.champions) for c in comparison.runs] == [1, 1], "the final champion of each"
    for run, champion in ((r, c.champions[0]) for r, c in by_run.items()):
        assert [e.scenario_id for e in champion.held_out] == list(comparison.held.ids)
        own = report.champions_on(run, report.find_milestones(run / train.RECORDING)[-1:], comparison.held, out=quiet)[0][0]
        assert champion.held_out == own.held_out
    # the old run keeps its 150-step limit on the common set: no episode of it runs longer
    assert max(e.steps for e in by_run[old].champions[0].held_out) <= 150
    assert max(e.steps for e in by_run[new].champions[0].held_out) <= 200
    assert {b.kind for b in comparison.baselines} == {"random brains", "senses only"}


def test_milestones_and_another_reference(runs: tuple[Path, Path]) -> None:
    old, new = runs
    comparison = compare.build_comparison([old, new], held_out=old, milestones=True, out=quiet)
    assert comparison.reference.run_dir == old and comparison.reference.config.scenarios.generator == 1
    assert [len(c.champions) for c in comparison.runs] == [len(report.find_milestones(r / train.RECORDING)) for r in (old, new)]


def test_mismatched_runs_are_refused(runs: tuple[Path, Path], tmp_path: Path) -> None:
    old, new = runs
    bigger = tmp_path / "bigger.toml"
    bigger.write_text(dump_config(load_config(new / train.CONFIG_FILE)).replace("agents = 10", "agents = 12"))
    with pytest.raises(compare.CompareError, match="team has 10 agents, the held-out layouts 12"):
        compare.build_comparison([old, new], held_out=bigger, out=quiet)
    wider = tmp_path / "wider.toml"
    wider.write_text(dump_config(load_config(new / train.CONFIG_FILE)).replace("size = 20.0", "size = 24.0"))
    with pytest.raises(compare.CompareError, match="world.arena"):
        compare.build_comparison([new], held_out=wider, out=quiet)
    with pytest.raises(compare.CompareError, match="not a box-pushing run"):
        compare.build_comparison([tmp_path], out=quiet)


def test_the_files_are_deterministic_and_the_bands_add_up(runs: tuple[Path, Path], tmp_path: Path) -> None:
    old, new = runs
    before = {r: fingerprint(r / train.RECORDING) for r in runs}
    first = compare.write_comparison(compare.build_comparison([old, new], out=quiet), tmp_path / "a")
    second = compare.write_comparison(compare.build_comparison([old, new], out=quiet), tmp_path / "b")
    assert (first / compare.CSV_FILE).read_text() == (second / compare.CSV_FILE).read_text()
    assert {r: fingerprint(r / train.RECORDING) for r in runs} == before, "nothing is written into the recordings"
    with open(first / compare.CSV_FILE) as file:
        rows = list(csv.DictReader(file))
    assert [r["run"] for r in rows] == ["old", "new", "baseline", "baseline"]
    assert [r["generator"] for r in rows[:2]] == ["1", "2"]
    for row in rows:
        bands = sum(int(row[f"held_{name}_episodes"]) for name, _, _ in compare.BANDS)
        assert bands == int(row["held_episodes"]), "the distance bands cover every episode"
        assert sum(int(row[f"held_{name}_delivered"]) for name, _, _ in compare.BANDS) == int(row["held_delivered"])
    assert "# Comparison on a common held-out set" in (first / compare.MARKDOWN_FILE).read_text()
    default = compare.write_comparison(compare.build_comparison([old, new], out=quiet))
    assert default == new / report.REPORT, "by default under the reference run's report/"


def test_showcase_with_two_runs_reports_both_and_compares(runs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """The clips, montages and charts are stubbed here (they have their own tests): this checks the order and the outputs."""
    called: list[str] = []

    def stub(name: str) -> Any:
        return lambda run, **_: called.append(f"{name} {Path(run).name}")

    for name in ("milestone_clips", "montage", "charts"):
        monkeypatch.setattr(showcase, name, stub(name))
    old, new = runs
    for run in runs:
        (run / report.REPORT / report.MARKDOWN_FILE).unlink(missing_ok=True)
    target = showcase.showcase_runs([old, new], out=quiet)
    assert target == new / report.REPORT
    for run in runs:
        assert (run / report.REPORT / report.MARKDOWN_FILE).exists()
    assert target is not None and (target / compare.MARKDOWN_FILE).exists() and (target / compare.CSV_FILE).exists()
    assert called[:4] == ["milestone_clips old", "montage old", "montage old", "charts old"]
    assert showcase.showcase_runs([new], out=quiet) is None, "one run: no comparison"


def test_runs_started_in_the_same_second_go_by_their_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """`started_at` has whole seconds, so two runs can tie (two quick runs in a row, as in these tests): the later one given wins."""
    same = datetime(2026, 10, 19, 12, 0, 0)
    monkeypatch.setattr(compare, "_started", lambda _: same)
    assert compare._newest([Path("a"), Path("b")]) == Path("b")  # pyright: ignore[reportPrivateUsage]
