"""Box-pushing's verbs for the `scenarios` command (`common.cli`):

- `scenarios box-pushing train CONFIG [--run-dir DIR]` trains a run defined by a TOML config (`configs/` has presets) into
  `runs/<name>-<timestamp>/`: its recording, progress, brains, held-out results and charts (`train.py`). Ctrl-C stops it
  resumably;
- `scenarios box-pushing train --resume RUN [--generations N]` continues an interrupted run, or extends it;
- `scenarios box-pushing deploy BRAIN... [options]` runs trained brains on scenarios they have not seen, into
  `deployments/<name>-<timestamp>/`: results, a summary and one video (`deploy.py`).

The verbs import what they need when they run, so `--help` is quick.
"""

import argparse
import sys

from auxein_scenarios.common.cli import Verb

__all__ = ["VERBS"]


def _train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", nargs="?", help="a TOML config (configs/smoke.toml, pilot.toml, full.toml)")
    parser.add_argument("--resume", metavar="RUN", help="continue the run in RUN instead of starting one")
    parser.add_argument("--generations", type=int, help="with --resume: the run's generations in all (more extends it)")
    parser.add_argument("--run-dir", help="where a new run goes (default: runs/<name>-<timestamp>)")


def _train(args: argparse.Namespace) -> int:
    from auxein_scenarios.box_pushing import train
    from auxein_scenarios.box_pushing.config import ConfigError, load_config

    if (args.config is None) == (args.resume is None):
        print("error: give a CONFIG to start a run, or --resume RUN to continue one", file=sys.stderr)
        return 2
    if args.config is not None and args.generations is not None:
        print("error: --generations goes with --resume; set [evolution] generations in the config to start a run", file=sys.stderr)
        return 2
    try:
        if args.resume is not None:
            train.resume(args.resume, generations=args.generations)
        else:
            train.train(load_config(str(args.config)), args.run_dir)
    except (ConfigError, train.TrainingError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def _deploy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("brains", nargs="+", metavar="BRAIN", help="brain files (a run's brains/*.safetensors)")
    parser.add_argument("--scenarios", type=int, default=64, help="how many scenarios (default 64)")
    parser.add_argument("--difficulty", choices=("full", "easy"), default="full", help="their difficulty (default full)")
    parser.add_argument("--seed", type=int, help="their seed (default: a fresh one, new to every brain)")
    parser.add_argument("--agents", type=int, help="the team size (default: each brain's own)")
    parser.add_argument("--backend", choices=("torch", "numpy"), help="default torch, or numpy without torch")
    parser.add_argument("--device", help="mps, cuda or cpu (default: the best available)")
    parser.add_argument("--precision", choices=("float32", "float64"), help="default float32 on torch, float64 on numpy")
    parser.add_argument("--no-video", action="store_true", help="skip the video")
    parser.add_argument("--video-scenarios", type=int, default=2, help="scenarios per brain in the video (default 2)")
    parser.add_argument("--out", help="where the deployment goes (default: deployments/<name>-<timestamp>)")


def _deploy(args: argparse.Namespace) -> int:
    from auxein_scenarios.box_pushing import deploy
    from auxein_scenarios.common.cli import BackendError, backend_from

    try:
        backend, note = backend_from(args.backend, args.device, args.precision)
        if note:
            print(f"note: {note}")
        deploy.deploy(
            args.brains, count=args.scenarios, difficulty=args.difficulty, seed=args.seed, agents=args.agents, backend=backend,
            video=not args.no_video, video_scenarios=args.video_scenarios, out_dir=args.out,
        )  # fmt: skip
    except (BackendError, deploy.DeployError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


VERBS = [
    Verb("train", "evolve brains: the run, its brains, held-out results and charts", _train_arguments, _train),
    Verb("deploy", "run trained brains on scenarios they have not seen: results and one video", _deploy_arguments, _deploy),
]
