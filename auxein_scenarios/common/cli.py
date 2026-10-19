"""The `scenarios` command: `scenarios <scenario> <verb> [options]`, one entry point for every scenario.

Each scenario registers itself in `SCENARIOS` (its name, a line saying what it is, and the module that declares its verbs);
the module has `VERBS`, a list of `Verb`s, each with its own options. The parser is built from them, so `scenarios --help`
lists the scenarios and `scenarios <scenario> --help` its verbs. A scenario's module is imported only to build its parser, and
its verbs import the heavy parts (Auxein, torch, matplotlib) only when they run.

`backend_from` is the device resolution every verb that computes shares: torch float32 on the best available device by
default, numpy float64 on the CPU when torch is not installed.
"""

import argparse
import importlib
import importlib.util
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from auxein.backend import Backend

__all__ = ["SCENARIOS", "BackendError", "Verb", "backend_from", "main", "parser"]


@dataclass(frozen=True)
class Verb:
    """A scenario's command: its `name`, a one-line `help`, `arguments` adding its options to its parser, and `run`, which
    takes the parsed options and returns the exit code."""

    name: str
    help: str
    arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], int]


SCENARIOS: dict[str, tuple[str, str]] = {
    "box-pushing": ("a team of agents with one shared brain pushes a heavy box to a target", "auxein_scenarios.box_pushing.cli"),
}
"""Every scenario: its name on the command line, what it is, and the module whose `VERBS` it runs."""


class BackendError(ValueError):
    """A backend, device or precision that cannot be used here."""


def backend_from(backend: str | None = None, device: str | None = None, precision: str | None = None) -> tuple[Backend, str | None]:
    """The backend to compute on: torch float32 on the best available device (Metal, then CUDA, then the CPU) unless told
    otherwise, numpy float64 on the CPU when torch is not installed and no backend was asked for (with a note saying so)."""
    has_torch = importlib.util.find_spec("torch") is not None
    name = backend or ("torch" if has_torch else "numpy")
    note = None if backend or has_torch else "torch is not installed (`uv sync --extra torch`): computing with numpy on the CPU"
    if precision not in (None, "float32", "float64"):
        raise BackendError(f"the precision is float32 or float64, got {precision!r}")
    if name == "numpy":
        if device not in (None, "cpu"):
            raise BackendError(f"numpy runs on the CPU, not {device!r}")
        return Backend("numpy", "cpu", cast("Any", precision or "float64")), note
    if name != "torch":
        raise BackendError(f"the backend is numpy or torch, got {name!r}")
    if not has_torch:
        raise BackendError("the torch backend needs torch: `uv sync --extra torch`")
    torch = importlib.import_module("torch")
    available = {"mps": bool(torch.backends.mps.is_available()), "cuda": bool(torch.cuda.is_available()), "cpu": True}
    if device is None:
        device = next(d for d in ("mps", "cuda", "cpu") if available[d])
    elif not available.get(device.split(":")[0], False):
        raise BackendError(f"the device {device!r} is not available here: use one of {[d for d, ok in available.items() if ok]}")
    return Backend("torch", device, cast("Any", precision or "float32")), note


def parser() -> argparse.ArgumentParser:
    """The whole command line: a subparser per scenario, and under it one per verb."""
    top = argparse.ArgumentParser(prog="scenarios", description="Demonstrations built on Auxein: evolve agents, then deploy them.")
    by_scenario = top.add_subparsers(dest="scenario", metavar="SCENARIO", required=True)
    for name, (about, module) in SCENARIOS.items():
        verbs = cast("Sequence[Verb]", importlib.import_module(module).VERBS)
        scenario = by_scenario.add_parser(name, help=about, description=about)
        by_verb = scenario.add_subparsers(dest="verb", metavar="VERB", required=True)
        for verb in verbs:
            sub = by_verb.add_parser(verb.name, help=verb.help, description=verb.help)
            verb.arguments(sub)
            sub.set_defaults(run=verb.run)
    return top


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run = cast("Callable[[argparse.Namespace], int]", args.run)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
