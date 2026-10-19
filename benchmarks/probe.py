"""The full performance probe: seconds per generation of the whole episode loop (sensing, the brain, the physics), broken down
by part, to choose the backend and the sizes of a run.

For each backend, team size and batch (`n` candidates × `s` scenarios, the scenarios a quarter easy and the rest full, as the
training mix), random brains sampled from `MLPPolicy.space()` (the GA's first generation, which never delivers, so every step
is simulated) act on their own senses:

- **parts**: `--steps` steps of sensing (`sensors.observe`), the brain (`act_batch`) and the physics (`world.step`), each
  timed on its own, with the device synchronised around it;
- **loop**: `--steps` steps of `BoxPushingWorld.run` itself, which also tracks the measurements, checks for non-finite states
  and the early stop: the number a generation actually costs;
- **generation**: the loop's time per step × the step limit: an upper bound, since delivered scenarios still cost compute;
- **memory**, where the backend reports it: numpy's peak through `tracemalloc` (numpy reports its allocations to it); on
  Metal, from `torch.mps`, the peak of what tensors hold during the loop (`current_allocated_memory`, sampled every step) and
  what the driver holds at its end (`driver_allocated_memory`, torch's cache included, emptied before each configuration so
  that the figure is this one's), as `allocated / driver`; CUDA's peak from torch's allocator; torch on the CPU reports none.

**Warm-up.** Each backend is warmed up once before anything is timed (a small configuration run through every part), so the
first configuration does not pay for the backend's start-up (Metal compiles its kernels on first use), and each configuration
runs its whole loop once untimed, so it does not pay for kernels of its own shapes either.

    uv run python -m benchmarks.probe                       # numpy float32 and torch float32 on the CPU
    uv run python -m benchmarks.probe --device mps          # and torch float32 on Metal (Apple Silicon)
    uv run python -m benchmarks.probe --agents 20 40 --batches 50x16 100x32

It ends with **a recommendation**: on the fastest backend measured, the largest team and batch whose generation fits the budget
(`--budget`, 60 s by default: about a minute per generation keeps a 400-generation run to an overnight one).
"""

import argparse
import importlib
import importlib.util
import platform
import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from auxein.backend import Backend
from auxein.random import RandomStream

from auxein_scenarios.box_pushing import scenarios
from auxein_scenarios.box_pushing.config import WorldParams
from auxein_scenarios.box_pushing.environment import BoxPushingWorld, brain
from auxein_scenarios.box_pushing.sensors import observe
from auxein_scenarios.box_pushing.world import initial_state, step


def _torch() -> Any:
    return importlib.import_module("torch")  # optional: imported by name, so the numpy-only install type-checks


def _sync(backend: Backend) -> None:
    if backend.name == "torch" and backend.device != "cpu":
        getattr(_torch(), backend.device.split(":")[0]).synchronize()


@dataclass(frozen=True)
class Timing:
    backend: Backend
    agents: int
    n: int
    s: int
    sensing: float
    policy: float
    physics: float
    loop: float
    """Seconds per step of each part, and of the whole loop."""
    peak: float | None
    """Peak memory in MB during the loop, where the backend reports it (on Metal, what tensors held)."""
    driver: float | None = None
    """Metal only: what the driver held, in MB, at the end of the loop."""

    def generation(self, steps: int) -> float:
        return self.loop * steps

    def memory(self) -> str:
        """The memory column: `peak`, or on Metal `allocated / driver`, or `n/a`."""
        if self.peak is None:
            return "n/a"
        return f"{self.peak:,.0f}" if self.driver is None else f"{self.peak:,.0f} / {self.driver:,.0f}"


class MetalMemory:
    """Metal's memory during a loop, from `torch.mps`: the peak of what tensors hold, sampled at every step (a `StepHook`), and
    what the driver holds at the end, after emptying torch's cache at the start so that it is this loop's."""

    def __init__(self, torch: Any) -> None:
        self._mps = torch.mps
        self.high = 0.0

    def start(self) -> None:
        self._mps.empty_cache()
        self.high = float(self._mps.current_allocated_memory())

    def __call__(self, t: int, state: Any, senses: Any) -> None:
        self.high = max(self.high, float(self._mps.current_allocated_memory()))

    def report(self) -> tuple[float, float]:
        """`(peak allocated, driver)` in MB."""
        return self.high / 2**20, float(self._mps.driver_allocated_memory()) / 2**20


def probe(backend: Backend, agents: int, n: int, s: int, steps: int, warmup: int, params: WorldParams) -> Timing:
    easy = max(1, s // 4)
    mix = ((scenarios.EASY, easy), (scenarios.FULL, s - easy))
    scenario_set = scenarios.training_set(agents, params, seed=1, mix=mix)
    world = BoxPushingWorld(params, steps=warmup + steps)
    layout = world.layout(tuple(scenario_set), backend)
    policy = brain(world.sensing)
    genomes = policy.space().sample_genomes(n, RandomStream(np.random.SeedSequence(0), backend), backend)
    agents_batch = policy.decode_batch(genomes)

    # the parts, one at a time
    state = initial_state(layout, n)
    parts = np.zeros(3)
    for t in range(warmup + steps):
        clock = [time.perf_counter()]
        observations = observe(state, layout, params, world.sensing)
        _sync(backend)
        clock.append(time.perf_counter())
        actions = agents_batch.act_batch(observations)
        _sync(backend)
        clock.append(time.perf_counter())
        state = step(state, actions, layout, params)
        _sync(backend)
        clock.append(time.perf_counter())
        if t >= warmup:
            parts += np.diff(clock)
    parts /= steps

    # the whole loop, as run_batch runs it (warmed up above); then, separately, its peak memory, since tracing slows numpy
    timed_world = BoxPushingWorld(params, steps=steps)

    def loop(hook: Any = None) -> float:
        _sync(backend)
        start = time.perf_counter()
        timed_world.run(agents_batch.act_batch, layout, n, hook=hook, stop_when_done=False)
        _sync(backend)
        return time.perf_counter() - start

    loop()  # the loop's own operations at these shapes, untimed (on Metal, their kernels compile on first use)
    seconds = loop()
    peak: float | None = None
    driver: float | None = None
    if backend.name == "numpy":
        tracemalloc.start()
        loop()
        peak = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()
    elif backend.device.startswith("cuda"):
        torch = _torch()
        torch.cuda.reset_peak_memory_stats()
        loop()
        peak = torch.cuda.max_memory_allocated() / 2**20
    elif backend.device == "mps":
        memory = MetalMemory(_torch())
        memory.start()
        loop(memory)
        peak, driver = memory.report()
    return Timing(backend, agents, n, s, float(parts[0]), float(parts[1]), float(parts[2]), seconds / steps, peak, driver)


def warm_up(backend: Backend, params: WorldParams) -> None:
    """A small configuration through every part, untimed, so that the first timed one does not pay for the backend's start-up."""
    probe(backend, 10, 4, 4, 1, 1, params)


def recommend(timings: Sequence[Timing], steps: int, budget: float) -> str:
    """On the fastest backend, the largest team and then the largest batch whose generation fits `budget` seconds."""
    if not timings:
        return "nothing was measured"
    totals: dict[str, float] = {}
    for t in timings:
        label = _label(t.backend)
        totals[label] = totals.get(label, 0.0) + t.loop
    fastest = min(totals, key=lambda label: totals[label])
    mine = [t for t in timings if _label(t.backend) == fastest]
    fitting = sorted((t for t in mine if t.generation(steps) <= budget), key=lambda t: (t.agents, t.n * t.s, t.n), reverse=True)
    if not fitting:
        smallest = min(mine, key=lambda t: t.generation(steps))
        return (
            f"{fastest} is the fastest backend measured, but no configuration fits {budget:.0f} s per generation: the smallest, "
            f"{smallest.agents} agents with {smallest.n} × {smallest.s}, takes {smallest.generation(steps):.0f} s"
        )
    best = fitting[0]
    lines = [
        f"backend {fastest}: team of {best.agents} agents, population {best.n}, {best.s} scenarios per generation "
        f"({best.generation(steps):.0f} s per generation, budget {budget:.0f} s)"
    ]
    others = [t for t in fitting[1:] if t.agents != best.agents][:2]
    for t in others:
        lines.append(f"  also within budget: {t.agents} agents with {t.n} × {t.s} ({t.generation(steps):.0f} s)")
    return "\n".join(lines)


def _label(backend: Backend) -> str:
    return f"{backend.name} {backend.device} {backend.precision}"


def _batch(text: str) -> tuple[int, int]:
    n, s = text.lower().split("x")
    return int(n), int(s)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="The full performance probe: sensing, brain and physics per generation.")
    parser.add_argument("--agents", type=int, nargs="+", default=[20, 40, 100])
    parser.add_argument("--batches", type=_batch, nargs="+", default=[(50, 16), (100, 16), (100, 32)], help="n x s, e.g. 100x32")
    parser.add_argument("--steps", type=int, default=3, help="timed steps per measurement (default: 3)")
    parser.add_argument("--warmup", type=int, default=1, help="untimed steps first (default: 1)")
    parser.add_argument("--device", default=None, help="also time torch float32 on this device (mps, cuda)")
    parser.add_argument("--no-cpu-torch", action="store_true", help="skip torch on the CPU")
    parser.add_argument("--budget", type=float, default=60.0, help="seconds per generation to fit (default: 60)")
    args = parser.parse_args(argv)

    params = WorldParams()
    has_torch = importlib.util.find_spec("torch") is not None
    backends = [Backend("numpy", "cpu", "float32")]
    if has_torch and not args.no_cpu_torch:
        backends.append(Backend("torch", "cpu", "float32"))
    if has_torch and args.device:
        backends.append(Backend("torch", args.device, "float32"))
    machine = f"{platform.machine()} {platform.system()} {platform.release()}"
    print(f"machine: {machine}, python {platform.python_version()}, numpy {np.__version__}")
    if has_torch:
        torch = _torch()
        print(f"torch {torch.__version__}, {torch.get_num_threads()} CPU threads")
    limit = params.simulation.max_steps
    print(f"{params.simulation.substeps} substeps per step; a generation is {limit} steps (an upper bound); budget {args.budget:.0f} s\n")
    for backend in backends:
        warm_up(backend, params)
    print("| backend | agents | n × s | sensing | brain | physics | loop (s/step) | s / generation | memory MB |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    timings: list[Timing] = []
    for agents in args.agents:
        for n, s in args.batches:
            for backend in backends:
                label = _label(backend)
                try:
                    t = probe(backend, agents, n, s, args.steps, args.warmup, params)
                except (MemoryError, RuntimeError) as error:  # torch reports allocation failures as RuntimeError
                    print(f"| {label} | {agents} | {n} × {s} | failed: {type(error).__name__} | | | | | |", flush=True)
                    continue
                timings.append(t)
                print(
                    f"| {label} | {agents} | {n} × {s} | {t.sensing:.3f} | {t.policy:.3f} | {t.physics:.3f} | {t.loop:.3f} | "
                    f"{t.generation(limit):,.0f} | {t.memory()} |",
                    flush=True,
                )
    print("\nrecommendation:")
    print(recommend(timings, limit, args.budget))


if __name__ == "__main__":
    main()
