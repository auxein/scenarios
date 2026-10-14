"""The box-pushing command-line tool (`uv run box-pushing ...`).

M1a has two commands, both for watching the world:

- `box-pushing demo <name>|all [--agents N] [--out DIR]` simulates and renders the demo clips (`demos.py`) to MP4;
- `box-pushing scenarios --difficulty easy|full --seed S --count C [--agents N] [--out DIR]` draws a gallery of generated
  layouts to PNG.

Both write to `out/` by default, which git ignores. Training from a config, and rendering a recorded run, come with M2 and M3.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from auxein.backend import Backend

from auxein_scenarios.box_pushing import demos, scenarios
from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.render import render_gallery

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
        title=f"{args.count} {args.difficulty} scenarios, seed {args.seed}, {args.agents} agents",
    )
    print(path)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="box-pushing", description="Cooperative box-pushing: watch the world (M1a).")
    commands = root.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="render demo clips of the physics, driven by scripted controllers")
    demo.add_argument("name", choices=[*demos.DEMOS, "all"], help="the demo to render, or all of them")
    demo.add_argument("--agents", type=int, default=0, help="team size for crowd, walls and delivery (default: theirs)")
    demo.add_argument("--out", default="out", help="output directory (default: out/)")
    demo.add_argument("--fps", type=float, default=30.0, help="frames per second (default: 30)")
    demo.add_argument("--speedup", type=float, default=2.0, help="simulated seconds per video second (default: 2)")
    demo.add_argument("--backend", choices=["numpy", "torch"], default="numpy", help="array backend (default: numpy)")
    demo.add_argument("--device", default="cpu", help="cpu, mps or cuda (torch only; default: cpu)")
    demo.add_argument("--precision", choices=["float32", "float64"], default=None, help="default: float64 on numpy CPU, else float32")
    demo.set_defaults(handler=_demo)

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
