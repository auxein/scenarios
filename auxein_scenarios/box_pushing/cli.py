"""The box-pushing command-line tool (`uv run box-pushing ...`).

Commands for watching the world:

- `box-pushing demo <name>|all [--agents N] [--out DIR]` simulates and renders the demo clips (`demos.py`) to MP4 (M1a; the
  `sensing` demo from M1b);
- `box-pushing scenarios --difficulty easy|full --seed S --count C [--agents N] [--out DIR]` draws a gallery of generated
  layouts to PNG (M1a);
- `box-pushing brains [--count C] [--seed S] [--difficulty easy|full] [--agents N] [--overlay]` renders random brains acting on
  their own senses, one panel per brain: the clueless start (M1b);
- `box-pushing senses [--count C] [--seed S] [--difficulty easy|full] [--agents N] [--overlay]` renders the senses-only
  controller on generated scenarios (M1b).

And for training (M2a, `train.py`):

- `box-pushing train CONFIG [--run-dir DIR]` trains a run defined by a TOML config (`configs/` has presets) into
  `runs/<name>-<timestamp>/` by default; Ctrl-C stops it resumably;
- `box-pushing train --resume DIR [--generations N] [--config CONFIG]` continues an interrupted run, or extends it (only the
  budget may change);
- `box-pushing watch DIR [--generation K] [--scenarios train:0 held:0 ...] [--overlay]` renders the best brain so far, with
  the run's backend; safe while the run is training.

And for showing a run (M3, `report.py`, `showcase.py`), all written under the run directory:

- `box-pushing report DIR [--generations 1,50,400] [--exploit TEXT]`: the held-out report of the milestones' champions and
  two baselines, into `DIR/report/`;
- `box-pushing milestones DIR [--generations ...] [--showcase train:0 held:3 best-held ...] [--overlay]`: one clip per
  milestone, into `DIR/clips/milestones/`;
- `box-pushing montage DIR [--layout sequence|grid] [--size 1920x1080] [--generations ...] [--showcase ...]`: into
  `DIR/clips/montage-<layout>.mp4`;
- `box-pushing charts DIR`: fitness, delivery, cooperation and generalisation charts, into `DIR/charts/`;
- `box-pushing showcase DIR`: all four, in that order (the report first), printing where everything is.

The watching commands write to `out/` by default and the training ones to `runs/`, both ignored by git.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from auxein.backend import Backend

from auxein_scenarios.box_pushing import demos, scenarios, showcase, train
from auxein_scenarios.box_pushing import report as held_out_report
from auxein_scenarios.box_pushing.config import ConfigError, WorldParams, load_config
from auxein_scenarios.box_pushing.milestones import MilestoneError, parse_generations
from auxein_scenarios.box_pushing.render import render_gallery
from auxein_scenarios.common.rendering import parse_size

__all__ = ["main"]

DIFFICULTIES = {"easy": scenarios.EASY, "full": scenarios.FULL}


def _backend(args: argparse.Namespace) -> Backend:
    precision = args.precision or ("float32" if args.device != "cpu" or args.backend == "torch" else "float64")
    return Backend(args.backend, args.device, precision)


def _demo(args: argparse.Namespace) -> int:
    names = list(demos.DEMOS) if args.name == "all" else [args.name]
    params, backend = WorldParams(), _backend(args)
    for name in names:
        demo = demos.build(name, params, agents=args.agents, backend=backend)
        path = demos.run(demo, args.out, params, backend=backend, fps=args.fps, speedup=args.speedup)
        print(f"{name}: {path}")
        for note in demo.notes:
            print(f"  note: {note}")
    return 0


def _scenarios(args: argparse.Namespace) -> int:
    params = WorldParams()
    difficulty = DIFFICULTIES[args.difficulty]
    generated = scenarios.training_set(args.agents, params, seed=args.seed, mix=((difficulty, args.count),))
    path = Path(args.out) / f"scenarios-{args.difficulty}-seed{args.seed}.png"
    render_gallery(
        [dict(s.params) for s in generated],
        params,
        path,
        labels=list(generated.ids),
        kicker="Scenarios",
        title=f"{args.count} {args.difficulty} scenarios, seed {args.seed}, {args.agents} agents",
    )
    print(path)
    return 0


def _report(watched: demos.Watched, cells: str) -> None:
    """Prints the clip's path and each panel's measurements."""
    print(watched.path)
    m = watched.measurements
    rows = m["delivered"].shape[0] * m["delivered"].shape[1]
    flat = {name: values.reshape(rows) for name, values in m.items()}
    for i, label in enumerate(watched.labels):
        progress = (flat["initial_distance"][i] - flat["final_distance"][i]) / flat["initial_distance"][i]
        delivered = f"delivered at step {int(flat['steps'][i])}" if flat["delivered"][i] > 0.5 else "not delivered"
        print(
            f"  {label}: {delivered}, progress {progress:+.2f}, approach {flat['approach'][i]:.2f}, "
            f"most touching the box {int(flat['max_pushers'][i])}"
        )
    print(f"  ({cells})")


def _brains(args: argparse.Namespace) -> int:
    watched = demos.watch_brains(
        args.out,
        count=args.count,
        seed=args.seed,
        difficulty=DIFFICULTIES[args.difficulty],
        agents=args.agents,
        overlay=args.overlay,
        backend=_backend(args),
        fps=args.fps,
        speedup=args.speedup,
    )
    _report(watched, "one random brain per panel, all on the same scenario")
    return 0


def _senses(args: argparse.Namespace) -> int:
    watched = demos.watch_senses(
        args.out,
        count=args.count,
        seed=args.seed,
        difficulty=DIFFICULTIES[args.difficulty],
        agents=args.agents,
        overlay=args.overlay,
        backend=_backend(args),
        fps=args.fps,
        speedup=args.speedup,
    )
    _report(watched, "the senses-only controller, one scenario per panel")
    return 0


def _train(args: argparse.Namespace) -> int:
    try:
        if args.resume:
            config = load_config(args.config) if args.config else None
            train.resume(args.resume, generations=args.generations, config=config)
        else:
            if not args.config:
                raise ConfigError("give a config to train (e.g. configs/pilot.toml), or --resume DIR")
            if args.generations is not None:
                raise ConfigError("--generations extends a run on --resume; for a new run, set budget.generations in the config")
            train.train(load_config(args.config), args.run_dir)
    except KeyboardInterrupt:
        where = args.resume or "the run directory printed above"
        print(f"\ninterrupted: Auxein checkpoints between generations, so the run resumes: uv run box-pushing train --resume {where}")
        return 130
    except (ConfigError, train.TrainingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def _watch(args: argparse.Namespace) -> int:
    try:
        showcase = [train.Showcase.parse(text) for text in args.scenarios] if args.scenarios else None
        train.watch(args.run_dir, generation=args.generation, showcase=showcase, overlay=args.overlay, fps=args.fps, speedup=args.speedup)
    except (ConfigError, ValueError, train.TrainingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


SHOWING_ERRORS = (ConfigError, MilestoneError, ValueError, train.TrainingError, FileNotFoundError)


def _showing(handler: Callable[[argparse.Namespace], None]) -> Callable[[argparse.Namespace], int]:
    """Runs an M3 command, printing its errors (a missing run, a bad generation list) instead of a traceback."""

    def run(args: argparse.Namespace) -> int:
        try:
            handler(args)
        except SHOWING_ERRORS as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        return 0

    return run


def _generations(args: argparse.Namespace) -> list[int] | None:
    return parse_generations(args.generations) if args.generations else None


def _report_command(args: argparse.Namespace) -> None:
    built = held_out_report.build_report(args.run_dir, generations=_generations(args))
    target = held_out_report.write_report(built, args.exploit)
    r = built.recommendation(args.exploit)
    final = built.final
    print(f"{target / held_out_report.MARKDOWN_FILE}")
    print(
        f"  final champion (generation {final.generation}): held-out {final.held().delivered}/{final.held().episodes} delivered, "
        f"full training {final.train('full').delivered}/{final.train('full').episodes}, gap {final.gap:+.2f}"
    )
    print(f"  recommendation: {r.next_milestone} ({r.reason})")


def _milestones_command(args: argparse.Namespace) -> None:
    showcase.milestone_clips(
        args.run_dir, generations=_generations(args), showcase=args.showcase, overlay=args.overlay, fps=args.fps, speedup=args.speedup
    )


def _montage_command(args: argparse.Namespace) -> None:
    showcase.montage(
        args.run_dir, layout=args.layout, size=parse_size(args.size), generations=_generations(args), showcase=args.showcase, fps=args.fps
    )


def _charts_command(args: argparse.Namespace) -> None:
    showcase.charts(args.run_dir)


def _showcase_command(args: argparse.Namespace) -> None:
    showcase.showcase(
        args.run_dir, generations=_generations(args), showcase_tokens=args.showcase, exploit=args.exploit, size=parse_size(args.size)
    )


def _run_options(command: argparse.ArgumentParser, *, showcase_scenarios: bool = True) -> None:
    command.add_argument("run_dir", help="the run directory (runs/<name>-<timestamp>)")
    command.add_argument(
        "--generations", default=None, help="milestone generations, e.g. 1,50,400 (default: 1, 2, 5, 10, 20, 50, ... and the last)"
    )
    if showcase_scenarios:
        command.add_argument(
            "--showcase",
            nargs="+",
            default=None,
            help="showcase scenarios: train:<i>, held:<i>, best-held, worst-held (default: the first easy and full, 2 held-out)",
        )


def _clip_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--out", default="out", help="output directory (default: out/)")
    command.add_argument("--fps", type=float, default=30.0, help="frames per second (default: 30)")
    command.add_argument("--speedup", type=float, default=2.0, help="simulated seconds per video second (default: 2)")
    command.add_argument("--backend", choices=["numpy", "torch"], default="numpy", help="array backend (default: numpy)")
    command.add_argument("--device", default="cpu", help="cpu, mps or cuda (torch only; default: cpu)")
    command.add_argument("--precision", choices=["float32", "float64"], default=None, help="default: float64 on numpy CPU, else float32")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="box-pushing", description="Cooperative box-pushing: watch the world, train, and show a run.")
    commands = root.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="render demo clips of the physics, driven by scripted controllers")
    demo.add_argument("name", choices=[*demos.DEMOS, "all"], help="the demo to render, or all of them")
    demo.add_argument("--agents", type=int, default=0, help="team size for crowd, walls and delivery (default: theirs)")
    _clip_options(demo)
    demo.set_defaults(handler=_demo)

    brains = commands.add_parser("brains", help="render random brains acting on their own senses: the clueless start")
    brains.add_argument("--count", type=int, default=6, help="how many random brains, one panel each (default: 6)")
    brains.add_argument("--seed", type=int, default=1, help="seeds the brains and the scenario (default: 1)")
    brains.add_argument("--difficulty", choices=sorted(DIFFICULTIES), default="easy")
    brains.add_argument("--agents", type=int, default=20)
    brains.add_argument("--overlay", action="store_true", help="draw what agent 0 senses in every panel")
    _clip_options(brains)
    brains.set_defaults(handler=_brains)

    senses = commands.add_parser("senses", help="render the senses-only controller on generated scenarios")
    senses.add_argument("--count", type=int, default=4, help="how many scenarios, one panel each (default: 4)")
    senses.add_argument("--seed", type=int, default=1)
    senses.add_argument("--difficulty", choices=sorted(DIFFICULTIES), default="easy")
    senses.add_argument("--agents", type=int, default=20)
    senses.add_argument("--overlay", action="store_true", help="draw what agent 0 senses in every panel")
    _clip_options(senses)
    senses.set_defaults(handler=_senses)

    trainer = commands.add_parser("train", help="train a run from a TOML config, or resume one")
    trainer.add_argument("config", nargs="?", help="the run's config (configs/smoke.toml, pilot.toml, full.toml)")
    trainer.add_argument("--run-dir", default=None, help="where the run goes (default: runs/<name>-<timestamp>)")
    trainer.add_argument("--resume", metavar="DIR", default=None, help="continue the run in DIR (interrupted, or to extend it)")
    trainer.add_argument("--generations", type=int, default=None, help="with --resume: the new budget, to extend the run")
    trainer.set_defaults(handler=_train)

    watcher = commands.add_parser("watch", help="render a run's best brain so far (safe while it trains)")
    watcher.add_argument("run_dir", help="the run directory")
    watcher.add_argument("--generation", type=int, default=None, help="the best up to this generation (default: the last)")
    watcher.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="showcase scenarios, train:<i> or held:<i> (default: the first easy and full, 2 held-out)",
    )
    watcher.add_argument("--overlay", action="store_true", help="draw what agent 0 senses")
    watcher.add_argument("--fps", type=float, default=30.0, help="frames per second (default: 30)")
    watcher.add_argument("--speedup", type=float, default=2.0, help="simulated seconds per video second (default: 2)")
    watcher.set_defaults(handler=_watch)

    reporter = commands.add_parser("report", help="the held-out report: milestone champions and baselines on unseen scenarios")
    _run_options(reporter, showcase_scenarios=False)
    reporter.add_argument("--exploit", default=None, help="an exploit seen in the clips (e.g. 'ramming'): recommends fixing it first")
    reporter.set_defaults(handler=_showing(_report_command))

    clips = commands.add_parser("milestones", help="one clip per milestone, of its champion on the same showcase scenarios")
    _run_options(clips)
    clips.add_argument("--overlay", action="store_true", help="draw what agent 0 senses")
    clips.add_argument("--fps", type=float, default=30.0, help="frames per second (default: 30)")
    clips.add_argument("--speedup", type=float, default=2.0, help="simulated seconds per video second (default: 2)")
    clips.set_defaults(handler=_showing(_milestones_command))

    montager = commands.add_parser("montage", help="the milestones in sequence, or side by side on one scenario")
    _run_options(montager)
    montager.add_argument("--layout", choices=["sequence", "grid"], default="sequence")
    montager.add_argument("--size", default="1920x1080", help="video size in pixels (default: 1920x1080)")
    montager.add_argument("--fps", type=float, default=30.0, help="frames per second (default: 30)")
    montager.set_defaults(handler=_showing(_montage_command))

    charter = commands.add_parser("charts", help="fitness, delivery, cooperation and generalisation charts (PNG and SVG)")
    charter.add_argument("run_dir", help="the run directory")
    charter.set_defaults(handler=_showing(_charts_command))

    shower = commands.add_parser("showcase", help="the report, milestone clips, both montages and the charts of a run")
    _run_options(shower)
    shower.add_argument("--exploit", default=None, help="an exploit seen in the clips (for the report's recommendation)")
    shower.add_argument("--size", default="1920x1080", help="the montages' size in pixels (default: 1920x1080)")
    shower.set_defaults(handler=_showing(_showcase_command))

    gallery = commands.add_parser("scenarios", help="render a gallery of generated scenario layouts")
    gallery.add_argument("--difficulty", choices=sorted(DIFFICULTIES), default="full")
    gallery.add_argument("--seed", type=int, default=1)
    gallery.add_argument("--count", type=int, default=12)
    gallery.add_argument("--agents", type=int, default=20)
    gallery.add_argument("--out", default="out", help="output directory (default: out/)")
    gallery.set_defaults(handler=_scenarios)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
