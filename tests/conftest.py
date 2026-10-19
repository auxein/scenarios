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


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """On GitHub Actions, each failure is also printed as an `::error` workflow command, which GitHub shows as an annotation on
    the run and the pull request, readable without opening the log."""
    import os
    import sys

    if os.environ.get("GITHUB_ACTIONS") != "true" or not report.failed:
        return
    text = str(report.longrepr).splitlines()
    lines = [line for line in text if line.startswith("E ")][:12] or text[-12:]
    message = "%0A".join(line.replace("%", "%25").replace("\r", "") for line in lines)
    sys.__stdout__.write(f"\n::error title={report.nodeid} ({report.when})::{message}\n")  # past pytest's capture
    sys.__stdout__.flush()
