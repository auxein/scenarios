"""Generator versions (design doc §4.2, M2b): generator 2 keeps the box clear of walls and obstacles; generator 1, what a
config without `scenarios.generator` means, is bit for bit the generator before M2b, so recorded runs stay reproducible."""

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from auxein.recording.reader import open_run

from auxein_scenarios.box_pushing import report, scenarios, train
from auxein_scenarios.box_pushing.config import WorldParams, load_config, parse_config
from auxein_scenarios.box_pushing.scenarios import EASY, EASY_2, FULL, FULL_2, Difficulty, box_clearance, generate_layout

DATA = Path(__file__).parent / "data"
ROOT = Path(__file__).resolve().parents[2]
PARAMS = WorldParams()
BOX_RADIUS = math.hypot(PARAMS.box.width, PARAMS.box.height) / 2.0


def clearance(layout: dict[str, object]) -> float:
    """The box's clearance, computed here independently of the generator: from its centre to the nearest wall and obstacle
    surface (circles: centre distance less the radius; rectangles: the exact distance to an axis-aligned rectangle), less the
    circumscribed radius, which bounds the box's footprint in any orientation."""
    x, y, _ = (float(v) for v in layout["box"])  # type: ignore[union-attr]
    half = PARAMS.arena.half
    nearest = min(half - abs(x), half - abs(y))
    for cx, cy, r in layout["circles"]:  # type: ignore[union-attr]
        nearest = min(nearest, math.hypot(x - cx, y - cy) - r)
    for rx, ry, hx, hy in layout["rects"]:  # type: ignore[union-attr]
        dx, dy = max(abs(x - rx) - hx, 0.0), max(abs(y - ry) - hy, 0.0)
        nearest = min(nearest, math.hypot(dx, dy))
    return nearest - BOX_RADIUS


@pytest.mark.parametrize("difficulty", [EASY_2, FULL_2], ids=["easy", "full"])
def test_generator_2_keeps_the_box_clear(difficulty: Difficulty) -> None:
    assert difficulty.box_clearance == scenarios.BOX_CLEARANCE == 1.0
    for seed in range(1000):
        layout = generate_layout(np.random.default_rng(seed), difficulty, 3, PARAMS)
        assert clearance(layout) >= 1.0 - 1e-9, f"seed {seed}: clearance {clearance(layout):.3f}"
        assert box_clearance(layout, PARAMS) == pytest.approx(clearance(layout), abs=1e-9)
        assert scenarios.is_valid(layout, PARAMS)


def test_violating_layouts_are_redrawn() -> None:
    """Generator 1 starts boxes where generator 2 would not; generator 2 redraws them (more attempts), and a larger clearance
    is kept just as well."""
    old = [generate_layout(np.random.default_rng(seed), FULL, 3, PARAMS) for seed in range(300)]
    assert min(clearance(layout) for layout in old) < 1.0, "generator 1 has boxes too close to walls or obstacles"
    new = [generate_layout(np.random.default_rng(seed), FULL_2, 3, PARAMS) for seed in range(300)]
    attempts = [np.mean([int(layout["attempts"]) for layout in batch]) for batch in (old, new)]  # type: ignore[call-overload]
    assert attempts[1] > attempts[0]
    wide = Difficulty("full", obstacles=(2, 6), box_target_distance=(6.0, 12.0), spawn_radius=None, box_clearance=2.5)
    for seed in range(100):
        assert clearance(generate_layout(np.random.default_rng(seed), wide, 3, PARAMS)) >= 2.5 - 1e-9


def test_the_target_and_the_spawns_are_unchanged_in_distribution() -> None:
    """A loose check: only the box's start is constrained, so the box–target distances and the agents' spawns are drawn as
    before (their means within a few percent), and the box only moves in from the walls."""
    for one, two in ((EASY, EASY_2), (FULL, FULL_2)):
        stats: list[tuple[float, float, float]] = []
        for difficulty in (one, two):
            distance, spawn, box = [], [], []
            for seed in range(400):
                layout = generate_layout(np.random.default_rng(seed), difficulty, 5, PARAMS)
                (bx, by, _), target = layout["box"], layout["target"]  # type: ignore[misc]
                distance.append(math.dist((bx, by), target))  # type: ignore[arg-type]
                agents = np.asarray(layout["agents"])
                spawn.append(float(np.mean(np.hypot(agents[:, 0] - bx, agents[:, 1] - by))))
                box.append(max(abs(bx), abs(by)))
            stats.append((float(np.mean(distance)), float(np.mean(spawn)), float(np.max(box))))
        (d1, s1, b1), (d2, s2, b2) = stats
        lo, hi = one.box_target_distance
        assert lo <= d2 <= hi and abs(d2 - d1) < 0.05 * (lo + hi), (one.name, d1, d2)
        assert abs(s2 - s1) < 0.08 * s1, (one.name, s1, s2)
        assert b2 <= PARAMS.arena.half - BOX_RADIUS - 1.0 + 1e-9 < b1


# --- generator 1 is the generator before M2b ---


PINS = json.loads((DATA / "v1_fingerprints.json").read_text())


def rounded(value: object) -> object:
    """Floats to 9 decimals, recursively: the layouts use `math.cos` and `math.sin`, whose last bit depends on the platform's
    libm, so the pins taken on macOS compare on any platform only once rounded (the exact fingerprints match on macOS)."""
    if isinstance(value, float):
        return round(value, 9) + 0.0
    if isinstance(value, list | tuple):
        return [rounded(v) for v in value]  # type: ignore[union-attr]
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}  # type: ignore[union-attr]
    return value


def params_sha(scenario_set: object) -> str:
    params = [rounded(dict(s.params)) for s in scenario_set]  # type: ignore[attr-defined]
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def same_platform_sets(config: object) -> bool:
    """Whether this platform rebuilds the smoke preset's sets bit for bit as pinned (true on macOS arm64, where they were)."""
    pinned = PINS["smoke"]
    return (train.training_set(config).fingerprint, train.held_out_set(config).fingerprint) == (  # type: ignore[arg-type]
        pinned["training"],
        pinned["held_out"],
    )


@pytest.mark.parametrize("preset", ["smoke", "full"])
def test_a_config_without_generator_rebuilds_the_sets_from_before_m2b(preset: str) -> None:
    """Configs from before M2b (the presets as they were, kept in `data/`) mean generator 1, and rebuild the training and
    held-out sets they did: the ids, seeds and params pinned from `main` before M2b (floats to 9 decimals, `rounded`)."""
    pinned = PINS[preset]
    config = load_config(DATA / f"v1_{preset}.toml")
    assert config.scenarios.generator == 1
    training, held = train.training_set(config), train.held_out_set(config)
    assert (params_sha(training), params_sha(held)) == (pinned["training_params_rounded_sha"], pinned["held_params_rounded_sha"])
    assert "difficulty" in training[0].params and "box_clearance" not in training[0].params


def test_generator_1_and_2_differ_and_are_written_out() -> None:
    v1 = parse_config("[scenarios]\neasy = 2\nfull = 2\nheld_out = 4\n[team]\nagents = 10")
    v2 = parse_config("[scenarios]\neasy = 2\nfull = 2\nheld_out = 4\ngenerator = 2\n[team]\nagents = 10")
    assert (v1.scenarios.generator, v2.scenarios.generator) == (1, 2)
    assert train.training_set(v1).fingerprint != train.training_set(v2).fingerprint
    assert train.held_out_set(v1).fingerprint != train.held_out_set(v2).fingerprint
    with pytest.raises(Exception, match="scenarios.generator"):
        parse_config("[scenarios]\ngenerator = 3")


def rows(path: Path) -> list[dict[str, str]]:
    with open(path) as file:
        return list(csv.DictReader(file))


def same_rows(a: list[dict[str, str]], b: list[dict[str, str]]) -> bool:
    """Equal CSV rows: text exactly, numbers to 1e-6 (the files carry six decimals)."""

    def equal(x: str, y: str) -> bool:
        try:
            return math.isclose(float(x), float(y), abs_tol=1e-6) or (math.isnan(float(x)) and math.isnan(float(y)))
        except ValueError:
            return x == y

    return len(a) == len(b) and all(r.keys() == s.keys() and all(equal(r[k], s[k]) for k in r) for r, s in zip(a, b, strict=True))


def test_a_run_recorded_before_m2b_reports_and_resumes(tmp_path: Path) -> None:
    """A smoke run recorded on `main` before M2b (`data/pre_m2b_smoke`, 2 generations, its config without `generator`): its
    report gives the CSV it gave then, and it resumes (Auxein compares the evaluator, whose scenario set's fingerprint is the
    generator's), its config then naming generator 1.

    A recording reproduces only where its scenarios do bit for bit: elsewhere a float's last bit differs (`rounded`), the
    fingerprint with it, and Auxein rightly refuses to resume. So this runs where the fixture was recorded (macOS arm64); the
    test below checks the same on any platform with a run recorded here."""
    run = tmp_path / "pre_m2b"
    shutil.copytree(DATA / "pre_m2b_smoke", run)
    if not same_platform_sets(load_config(run / train.CONFIG_FILE)):
        pytest.skip("this platform's libm rebuilds the fixture's layouts with other last bits (recorded on macOS arm64)")
    before = rows(run / report.REPORT / report.SUMMARY_FILE)
    episodes = rows(run / report.REPORT / report.EPISODES_FILE)
    assert "generator" not in (run / train.CONFIG_FILE).read_text()
    written = report.write_report(report.build_report(run, out=lambda _: None))
    assert same_rows(rows(written / report.SUMMARY_FILE), before)
    assert same_rows(rows(written / report.EPISODES_FILE), episodes)
    result, _ = train.resume(run, generations=3, out=lambda _: None, progress=False)
    assert result.evaluations_used == 4 + 2 * 4
    assert load_config(run / train.CONFIG_FILE).scenarios.generator == 1, "the rewritten config says generator 1"


def test_a_run_without_generator_resumes_as_generator_1(tmp_path: Path) -> None:
    """On any platform: a run whose config has no `generator` (the pre-M2b smoke preset) records, resumes, and keeps
    generator 1 in its rewritten config."""
    run = tmp_path / "legacy"
    train.train(load_config(DATA / "v1_smoke.toml").with_generations(1), run, out=lambda _: None, progress=False, estimate=False)
    assert load_config(run / train.CONFIG_FILE).scenarios.generator == 1
    train.resume(run, generations=2, out=lambda _: None, progress=False)
    with open_run(run / train.RECORDING) as reader:
        assert reader.generations() == 2
    assert load_config(run / train.CONFIG_FILE).scenarios.generator == 1


def test_a_new_run_writes_generator_2(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "smoke.toml").with_generations(1)
    lines: list[str] = []
    train.train(config, tmp_path / "new", out=lines.append, progress=False, estimate=False)
    assert "generator = 2" in (tmp_path / "new" / train.CONFIG_FILE).read_text()
    assert any(line.startswith("scenarios: generator 2") for line in lines)
