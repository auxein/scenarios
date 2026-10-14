# scenarios

Demonstrations built on [Auxein](https://github.com/auxein/auxein), a Python framework for evolving agents that act in
environments. Each scenario is a world plus everything needed to evolve agents in it, watch them and show the results: videos
for a website and talks, and simple charts.

## Scenarios

| Scenario | Status | Design |
|---|---|---|
| **Cooperative box-pushing**: a team whose shared brain is evolved by Auxein pushes a box too heavy for any one of them to a target, among obstacles | M1a: physics, scenario generator and viewer | [`docs/design/box-pushing.md`](docs/design/box-pushing.md) |

## Install

With [uv](https://docs.astral.sh/uv/), Python 3.11 or later:

```
uv sync                  # numpy only, plus the dev tools and the video tools (matplotlib, a bundled ffmpeg)
uv sync --extra torch    # with PyTorch: CPU-only wheels on Linux, the normal wheels (with Metal) on macOS
uv sync --extra cma      # with pycma, for the CMA-ES comparison runs
```

Auxein is a git dependency pinned to a release tag (`[tool.uv.sources]` in `pyproject.toml`). **To move to a newer tag**, change
`tag = "..."` there, run `uv lock --upgrade-package auxein`, and update the pinned tag and commit in `tests/smoke_test.py`
(the full checklist is in [`CLAUDE.md`](CLAUDE.md)).

## Watching the box-pushing world

The physics and the scenario generator can be watched before anything is trained: demo clips driven by scripted controllers,
and galleries of generated layouts. Output goes to `out/` (ignored by git).

```
uv run box-pushing demo all                                   # every demo clip, as MP4
uv run box-pushing demo threshold                             # one of: threshold, rotation, obstacles, crowd, walls, delivery
uv run box-pushing demo crowd --agents 100                    # the team size, for crowd, walls and delivery
uv run box-pushing scenarios --difficulty full --seed 1 --count 12 --agents 20   # a gallery of layouts, as PNG
uv run --extra torch box-pushing demo all --backend torch --device mps          # on Metal
uv run python -m benchmarks.physics_timing --device mps                         # seconds per physics step
```

## Checks

```
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest
```

After `uv sync --extra torch`, use `uv run --no-sync ...` (or `uv run --extra torch ...`), since a plain `uv run` syncs back to
the default installation. CI runs the checks on Linux, numpy only and with torch on the CPU.

## Auxein bugs found here

The scenarios are Auxein's first real user. Bugs found in Auxein are fixed in Auxein, test first, in their own PRs there, and
this repository always depends on a released tag: see "Working on Auxein from this repository" in [`CLAUDE.md`](CLAUDE.md), and
the log in [`docs/auxein-issues.md`](docs/auxein-issues.md).
