# scenarios

Demonstrations built on [Auxein](https://github.com/auxein/auxein), a Python framework for evolving agents that act in
environments. Each scenario is a world plus what it takes to evolve agents in it and to show what they learned. You train
brains, then deploy them on situations they have never seen.

## Scenarios

| Scenario | What happens | Design |
|---|---|---|
| `box-pushing` | A team of agents, every one running the same evolved brain, pushes a box too heavy for any one of them to a target among obstacles. Each agent senses only its surroundings. | [`docs/design/box-pushing.md`](docs/design/box-pushing.md) |

## Install

With [uv](https://docs.astral.sh/uv/), Python 3.11 or later:

```
uv sync                  # numpy only (training on the CPU; fine for the smoke preset and the tests)
uv sync --extra torch    # with PyTorch: Metal on Apple silicon, CUDA or the CPU elsewhere
```

## Two commands

Every scenario has the same two verbs: `uv run scenarios <scenario> train|deploy`. Add `--extra torch` to `uv run` when you
use torch, or uv removes it.

**Train** brains from a config. The presets are in `configs/`; a config lists only what it changes:

```
uv run scenarios box-pushing train configs/smoke.toml                  # seconds, numpy: a first try
uv run --extra torch scenarios box-pushing train configs/pilot.toml    # the full sizes for 25 generations: about 25 minutes on Metal
uv run --extra torch scenarios box-pushing train configs/full.toml     # 400 generations: about 6 hours on Metal
uv run --extra torch scenarios box-pushing train --resume runs/full-<timestamp>                    # after Ctrl-C
uv run --extra torch scenarios box-pushing train --resume runs/full-<timestamp> --generations 600  # extend it
```

**Deploy** trained brains on 64 new scenarios (by default), next to two baselines (random brains, a hand-written controller):

```
uv run --extra torch scenarios box-pushing deploy runs/pilot-<timestamp>/brains/final.safetensors
uv run --extra torch scenarios box-pushing deploy runs/pilot-<timestamp>/brains/{gen-0001,gen-0010,final}.safetensors
uv run --extra torch scenarios box-pushing deploy BRAIN --agents 80 --scenarios 16 --seed 7 --no-video
```

Deploying several brains of one run together shows what training changed: the video gives each brain a row, on the same
scenarios. `uv run scenarios box-pushing deploy --help` lists the options.

## What they write

```
runs/<name>-<timestamp>/          # train
  config.toml                     #   the run's config, every default filled in
  progress.csv                    #   per generation: scores, success, cooperation
  held_out.csv                    #   the saved brains judged on 64 scenarios never seen in training
  brains/                         #   gen-<k>.safetensors at generations 1, 2, 5, 10, 20, 50, … and final.safetensors
  charts/                         #   fitness, delivery, cooperation, generalisation (PNG and SVG)
  recording/                      #   Auxein's record of the run (open it with auxein.open_run)

deployments/<name>-<timestamp>/   # deploy
  deploy.toml                     #   the options (the seed among them) and each brain's provenance
  results.csv                     #   per brain and scenario: delivered, steps, progress, agents pushing
  summary.md                      #   per brain and baseline: success, progress, success by distance
  deployment.mp4                  #   a row per brain, on the same scenarios
```

A brain is a small `.safetensors` file: its layers as tensors, and what it was trained in and where it came from as metadata.
Both directories are ignored by git.

## Checks

```
uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest
uv run --extra torch pyright && uv run --extra torch pytest
```

CI runs them with numpy only and with torch, on Python 3.11 to 3.14.
