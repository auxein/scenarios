"""Backend fixtures for the whole suite, mirroring Auxein's (`tests/support/fixtures.py` there).

- `backend` runs a unit test on numpy float64, numpy float32 and torch float32 (CPU).
- `corner_backend` runs an integration or physics test on the two most different configurations: numpy float64 (the
  reference) and torch float32.

The torch configurations skip cleanly when torch is not installed (the numpy-only CI job)."""

import importlib.util

import pytest
from auxein.backend import Backend

HAS_TORCH = importlib.util.find_spec("torch") is not None
_NO_TORCH = pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed (uv sync --extra torch)")


@pytest.fixture(
    params=[
        pytest.param(("numpy", "float64"), id="numpy-float64"),
        pytest.param(("numpy", "float32"), id="numpy-float32"),
        pytest.param(("torch", "float32"), id="torch-float32", marks=_NO_TORCH),
    ]
)
def backend(request: pytest.FixtureRequest) -> Backend:
    name, precision = request.param
    return Backend(name, "cpu", precision)


@pytest.fixture(
    params=[
        pytest.param(("numpy", "float64"), id="numpy-float64"),
        pytest.param(("torch", "float32"), id="torch-float32", marks=_NO_TORCH),
    ]
)
def corner_backend(request: pytest.FixtureRequest) -> Backend:
    name, precision = request.param
    return Backend(name, "cpu", precision)


def _annotate(nodeid: str, when: str, longrepr: object) -> None:
    """On GitHub Actions, a failure is also printed as an `::error` workflow command, which GitHub shows as an annotation on the
    run and the pull request, readable without opening the log."""
    import os
    import sys

    out = sys.__stdout__
    if os.environ.get("GITHUB_ACTIONS") != "true" or out is None:
        return
    text = str(longrepr).splitlines()
    lines = [line for line in text if line.startswith("E ")][:12] or text[-12:]
    message = "%0A".join(line.replace("%", "%25").replace("\r", "") for line in lines)
    out.write(f"\n::error title={nodeid} ({when})::{message}\n")  # past pytest's capture
    out.flush()


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.failed:
        _annotate(report.nodeid, report.when, report.longrepr)


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.failed:
        _annotate(report.nodeid, "collection", report.longrepr)
