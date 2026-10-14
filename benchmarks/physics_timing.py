"""A quick timing of the box-pushing physics core (M1a): seconds per episode step for a population of teams.

Not the full performance probe (M1b, with sensing and the policy): this times `world.step` alone (contacts, friction and
integration, `substeps` physics substeps per step) on the training set's layouts, with random actions drawn once beforehand so
that only the physics is timed. Each configuration runs `--warmup` steps first (allocation, torch's lazy initialisation), then
times `--steps` steps.

    uv run python -m benchmarks.physics_timing                       # numpy float32 and torch float32 on the CPU
    uv run python -m benchmarks.physics_timing --device mps          # and torch float32 on Metal (Apple Silicon)
    uv run python -m benchmarks.physics_timing --agents 20 40 --candidates 50 --scenarios 16

The table gives the time per episode step for `n` candidates × `s` scenarios × `N` agents, and the projection to one
generation (`n` candidates on `s` scenarios for the episode's `max_steps` steps), the number that decides whether training
runs in hours (design doc §10.3).
"""

import argparse
import importlib.util
import platform
import time
from collections.abc import Sequence

import numpy as np
from auxein.backend import Backend

from auxein_scenarios.box_pushing import scenarios
from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.world import initial_state, step


def _sync(backend: Backend) -> None:
    if backend.name == "torch" and backend.device != "cpu":
        import torch

        getattr(torch, backend.device.split(":")[0]).synchronize()


def time_step(backend: Backend, agents: int, candidates: int, n_scenarios: int, steps: int, warmup: int, params: WorldParams) -> float:
    """Seconds per episode step, averaged over `steps` steps after `warmup`."""
    mix = ((scenarios.EASY, max(1, n_scenarios // 4)), (scenarios.FULL, n_scenarios - max(1, n_scenarios // 4)))
    layouts = [s.params for s in scenarios.training_set(agents, params, seed=1, mix=mix)]
    layout = scenarios.layout_arrays(layouts, backend)
    state = initial_state(layout, candidates)
    rng = np.random.default_rng(0)
    actions = [backend.asarray(rng.uniform(-1.0, 1.0, (candidates, n_scenarios, agents, 2))) for _ in range(warmup + steps)]
    for t in range(warmup):
        state = step(state, actions[t], layout, params)
    _sync(backend)
    start = time.perf_counter()
    for t in range(warmup, warmup + steps):
        state = step(state, actions[t], layout, params)
    _sync(backend)
    return (time.perf_counter() - start) / steps


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--agents", type=int, nargs="+", default=[20, 40, 100])
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--scenarios", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default=None, help="also time torch float32 on this device (mps, cuda)")
    args = parser.parse_args(argv)

    params = WorldParams()
    backends = [Backend("numpy", "cpu", "float32")]
    if importlib.util.find_spec("torch") is not None:
        backends.append(Backend("torch", "cpu", "float32"))
        if args.device:
            backends.append(Backend("torch", args.device, "float32"))
    machine = f"{platform.machine()} {platform.system()} {platform.release()}"
    print(f"machine: {machine}, python {platform.python_version()}, numpy {np.__version__}")
    if importlib.util.find_spec("torch") is not None:
        import torch

        print(f"torch {torch.__version__}, {torch.get_num_threads()} CPU threads")
    n, s, steps = args.candidates, args.scenarios, params.simulation.max_steps
    print(f"\n{n} candidates × {s} scenarios, {params.simulation.substeps} substeps per step; a generation is {steps} steps\n")
    print("| backend | agents | s / step | agent-steps / s | s / generation |")
    print("|---|---:|---:|---:|---:|")
    for agents in args.agents:
        for backend in backends:
            label = f"{backend.name} {backend.device} {backend.precision}"
            try:
                seconds = time_step(backend, agents, n, s, args.steps, args.warmup, params)
            except (MemoryError, RuntimeError) as error:  # torch reports allocation failures as RuntimeError
                print(f"| {label} | {agents} | failed: {type(error).__name__} | | |")
                continue
            rate = n * s * agents / seconds
            print(f"| {label} | {agents} | {seconds:.3f} | {rate:,.0f} | {seconds * steps:,.0f} |", flush=True)


if __name__ == "__main__":
    main()
