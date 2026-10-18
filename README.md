# scenarios

Demonstrations built on [Auxein](https://github.com/auxein/auxein), a Python framework for evolving agents that act in
environments. Each scenario is a world plus everything needed to evolve agents in it, watch them and show the results: videos
for a website and talks, and simple charts.

## Scenarios

| Scenario | Status | Design |
|---|---|---|
| **Cooperative box-pushing**: a team whose shared brain is evolved by Auxein pushes a box too heavy for any one of them to a target, among obstacles | M2a: fitness, the training config and command, a pilot run | [`docs/design/box-pushing.md`](docs/design/box-pushing.md) |

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

The world can be watched before anything is trained: demo clips driven by scripted controllers, galleries of generated
layouts, and policies acting on their own senses (random brains, a hand-written senses-only controller), with an optional
overlay of what one agent senses. Output goes to `out/` (ignored by git).

```
uv run box-pushing demo all                                   # every demo clip, as MP4
uv run box-pushing demo threshold                             # one of: threshold, rotation, obstacles, crowd, walls, delivery, sensing
uv run box-pushing demo crowd --agents 100                    # the team size, for crowd, walls and delivery
uv run box-pushing scenarios --difficulty full --seed 1 --count 12 --agents 20   # a gallery of layouts, as PNG
uv run box-pushing brains --overlay                           # random brains, one panel each: the clueless start
uv run box-pushing senses --difficulty full --agents 40       # the senses-only controller on generated scenarios
uv run --extra torch box-pushing demo all --backend torch --device mps          # on Metal
uv run python -m benchmarks.physics_timing --device mps                         # seconds per physics step
uv run python -m benchmarks.probe --device mps                                  # seconds per generation, and a recommendation
```

## Training

A run is defined by a TOML config (`configs/` has presets; a config lists only what it changes from the design doc's
defaults). Run directories go to `runs/` (ignored by git). Training defaults to torch on Metal, so install torch first
(`uv sync --extra torch`); `smoke.toml` runs on numpy.

```
uv run box-pushing train configs/smoke.toml                   # a few seconds, numpy: a first try
uv run box-pushing train configs/full.toml                    # the overnight run: 40 agents, population 100, 400 generations
uv run box-pushing watch runs/full-<timestamp>                # the best brain so far, as MP4 (safe while it trains)
uv run box-pushing train --resume runs/full-<timestamp>       # after Ctrl-C (which stops after the current generation)
uv run box-pushing train --resume runs/full-<timestamp> --generations 600   # extend it
```

`runs/<run>/progress.csv` is rewritten after every generation (best and mean score, success rate, `max_pushers`...), so a
run can be charted while it goes on.

## Showing a run

Once a run has trained, one command makes its held-out report, its milestone clips, both montages and its charts, all under
the run directory (with the run's backend: torch on Metal for the presets, hence `--extra torch`):

```
uv run --extra torch box-pushing showcase runs/full-<timestamp>
```

Or each step on its own:

```
uv run --extra torch box-pushing report runs/full-<timestamp>       # report/: champions and baselines on 64 unseen scenarios
uv run --extra torch box-pushing milestones runs/full-<timestamp>   # clips/milestones/gen-<k>.mp4, one per milestone
uv run --extra torch box-pushing montage runs/full-<timestamp>      # clips/montage-sequence.mp4 (--layout grid: side by side)
uv run --extra torch box-pushing charts runs/full-<timestamp>       # charts/: fitness, delivery, cooperation, generalisation
```

Milestones are generations 1, 2, 5, 10, 20, 50, 100, 200, ... and the last (`--generations 1,50,400` to choose); clips show the
same scenarios at every milestone (`--showcase train:0 held:3 best-held worst-held` to choose), and every panel says whether it
is a training scenario or a held-out one never seen in training. The report's `report.md` ends with a recommendation for the
next step.

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
