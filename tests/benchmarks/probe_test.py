"""The probe's fixes (M2a): every backend is warmed up before anything is timed, and Metal's memory comes from `torch.mps`.
Unit-level: no Metal in CI, so `torch.mps` is a fake here."""

from types import SimpleNamespace
from typing import Any

import pytest
from auxein.backend import Backend

from benchmarks import probe
from benchmarks.probe import MetalMemory, Timing


def test_every_backend_is_warmed_up_before_the_first_timing(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[tuple[str, str]] = []

    def fake_probe(backend: Backend, agents: int, n: int, s: int, steps: int, warmup: int, params: Any) -> Timing:
        calls.append(("probe", backend.name))
        return Timing(backend, agents, n, s, 0.01, 0.001, 0.01, 0.02, 1.0)

    monkeypatch.setattr(probe, "probe", fake_probe)
    monkeypatch.setattr(probe, "warm_up", lambda backend, params: calls.append(("warm", backend.name)))
    probe.main(["--agents", "20", "40", "--batches", "10x4"])
    names = {name for kind, name in calls if kind == "probe"}
    first_probe = next(i for i, (kind, _) in enumerate(calls) if kind == "probe")
    assert {name for kind, name in calls[:first_probe] if kind == "warm"} == names, "each backend warmed up, before any timing"
    assert sum(1 for kind, _ in calls if kind == "warm") == len(names), "once per backend"
    assert "memory MB" in capsys.readouterr().out


def test_the_real_warm_up_runs_a_small_configuration() -> None:
    probe.warm_up(Backend("numpy", "cpu", "float32"), probe.WorldParams())


class FakeMps:
    def __init__(self) -> None:
        self.allocated = [100 * 2**20, 300 * 2**20, 250 * 2**20, 200 * 2**20]
        self.emptied = 0

    def empty_cache(self) -> None:
        self.emptied += 1

    def current_allocated_memory(self) -> int:
        return self.allocated.pop(0) if len(self.allocated) > 1 else self.allocated[0]

    def driver_allocated_memory(self) -> int:
        return 512 * 2**20


def test_metal_memory_is_torch_mps_allocated_and_driver() -> None:
    fake = FakeMps()
    memory = MetalMemory(SimpleNamespace(mps=fake))
    memory.start()
    for t in range(3):
        memory(t, None, None)
    peak, driver = memory.report()
    assert fake.emptied == 1, "torch's cache is emptied first, so the driver figure is this configuration's"
    assert (peak, driver) == (300.0, 512.0)
    metal = Timing(Backend("numpy", "cpu", "float32"), 40, 100, 16, 0.0, 0.0, 0.0, 0.0, peak, driver)
    assert metal.memory() == "300 / 512"
    assert Timing(Backend("numpy", "cpu", "float32"), 40, 100, 16, 0.0, 0.0, 0.0, 0.0, 56.4).memory() == "56"
    assert Timing(Backend("numpy", "cpu", "float32"), 40, 100, 16, 0.0, 0.0, 0.0, 0.0, None).memory() == "n/a"
