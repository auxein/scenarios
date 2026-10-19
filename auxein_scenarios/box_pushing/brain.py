"""A box-pushing brain as a file: `save_brain` writes a trained genome with everything needed to run it again, `load_brain` reads
it back and checks that this code can run it.

The tensors are the network's layers (`common.brains`). The metadata says what the brain is and where it came from:

- `format`: `FORMAT`, which names the scenario and the version of this layout, and `scenario`: `"box-pushing"`;
- `brain`: the network's spec (observation size, hidden widths, outputs, activation, output mode);
- `observation_layout`: the names and sizes of the observation's parts, in order (`sensors.observation_layout`): a brain reads
  its inputs by position, so a brain trained on another layout would read the wrong senses;
- `sensing`, `world` and `team`: the sensing range, the world's parameters and the team size it was trained with, so that it
  runs in its own world wherever it is deployed;
- `provenance`: the run (its config's name, its directory, its seed, its scenario sets), the generation and candidate, its
  training score and success, the fitness weights it was scored with, the Auxein and scenarios versions (and the commit, when
  known), and when it was saved.

**Loading refuses** a file of another format or whose observation layout differs from the code's, with a message saying which.
"""

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from auxein.environments import MLPPolicy

from auxein_scenarios.box_pushing.config import SensingParams, TrainingConfig, WorldParams
from auxein_scenarios.box_pushing.environment import ACTIONS, brain
from auxein_scenarios.box_pushing.sensors import observation_layout
from auxein_scenarios.common.brains import BrainFileError, load_mlp, save_mlp

__all__ = ["FORMAT", "SCENARIO", "Brain", "BrainFileError", "layout_of", "load_brain", "save_brain"]

SCENARIO = "box-pushing"
FORMAT = "auxein-scenarios/box-pushing/brain/1"
"""The format tag of the files this module writes and reads."""

ACTIVATION, OUTPUT = "tanh", "tanh"
"""The brain's hidden activation and output mode (`environment.brain`): thrust and turning in [-1, 1]."""


def layout_of(sensing: SensingParams) -> list[list[object]]:
    """The observation layout as `[[name, size], ...]`, in order: what a brain file records and `load_brain` compares."""
    parts = observation_layout(sensing).parts
    return [[name, part.stop - part.start] for name, part in parts.items()]


@dataclass(frozen=True)
class Brain:
    """A brain loaded from a file: its genome, its network, the world, sensing and team it was trained with, its provenance."""

    path: Path
    genome: npt.NDArray[np.float64]
    policy: MLPPolicy
    precision: str
    world: WorldParams
    sensing: SensingParams
    agents: int
    provenance: dict[str, Any]

    @property
    def label(self) -> str:
        """Its run and generation, e.g. `pilot-20261020-101500 · generation 25`, or `· final` for a run's final brain."""
        p = self.provenance
        which = "final" if p.get("final") else f"generation {p.get('generation')}"
        return f"{p.get('run', self.path.stem)} · {which}"

    @property
    def seeds(self) -> tuple[int, int]:
        """The seeds of its training and held-out sets."""
        s = cast("dict[str, int]", self.provenance["scenarios"])
        return int(s["seed"]), int(s["held_out_seed"])


def _version(name: str) -> str:
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:  # pragma: no cover - always installed here
        return "unknown"


def _commit() -> str | None:
    """The scenarios repository's commit, if this code runs from a git checkout (with `+dirty` for uncommitted changes)."""
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        status = ["git", "status", "--porcelain", "--untracked-files=no"]
        dirty = subprocess.run(status, cwd=root, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + ("+dirty" if dirty.strip() else "") if commit else None


def save_brain(
    path: str | Path,
    genome: object,
    config: TrainingConfig,
    *,
    run_dir: Path,
    generation: int,
    candidate_id: int,
    score: float,
    success: float,
    final: bool = False,
) -> Path:
    """Writes a brain of the run defined by `config` (module docstring); `score` and `success` are its training results."""
    policy = brain(config.sensing, config.brain.hidden)
    s = config.scenarios
    provenance: dict[str, object] = {
        "name": config.name,
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "run_seed": config.run.seed,
        "generation": generation,
        "candidate_id": candidate_id,
        "final": final,
        "training_score": score,
        "training_success": success,
        "fitness": {
            "progress": config.fitness.progress,
            "delivery": config.fitness.delivery,
            "time": config.fitness.time,
            "approach": config.fitness.approach,
        },  # fmt: skip
        "scenarios": {"easy": s.easy, "full": s.full, "seed": s.seed, "held_out": s.held_out, "held_out_seed": s.held_out_seed},
        "auxein_version": _version("auxein"),
        "scenarios_version": _version("auxein-scenarios"),
        "scenarios_commit": _commit(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    metadata: dict[str, object] = {
        "format": FORMAT,
        "scenario": SCENARIO,
        "brain": {
            "observation_size": policy.observation_size,
            "hidden": list(policy.hidden),
            "outputs": policy.outputs,
            "activation": ACTIVATION,
            "output": OUTPUT,
        },  # fmt: skip
        "observation_layout": layout_of(config.sensing),
        "sensing": {"range": config.sensing.range, "rays": config.sensing.rays, "sectors": config.sensing.sectors},
        "world": config.world.to_dict(),
        "team": {"agents": config.team.agents},
        "provenance": provenance,
    }
    return save_mlp(path, genome, policy.layers, config.run.precision, metadata)


def load_brain(path: str | Path) -> Brain:
    """Reads a brain written by `save_brain`, refusing one of another format or observation layout (module docstring)."""
    source = Path(path)
    loaded = load_mlp(source, FORMAT)
    m = loaded.metadata
    sensing = SensingParams(range=float(cast("dict[str, float]", m["sensing"])["range"]))
    expected, found = layout_of(sensing), m.get("observation_layout")
    if found != expected:
        raise BrainFileError(
            f"{source} was trained on another observation layout than this code's: it has {json.dumps(found)}, "
            f"this code {json.dumps(expected)}"
        )
    spec = cast("dict[str, Any]", m["brain"])
    policy = brain(sensing, tuple(int(w) for w in spec["hidden"]))
    if (spec["outputs"], spec["activation"], spec["output"]) != (ACTIONS, ACTIVATION, OUTPUT) or policy.layers != loaded.layers:
        raise BrainFileError(f"{source} holds a network this code does not build: {json.dumps(spec)} with layers {list(loaded.layers)}")
    return Brain(
        path=source,
        genome=loaded.genome,
        policy=policy,
        precision=loaded.precision,
        world=WorldParams.from_dict(cast("dict[str, Any]", m["world"])),
        sensing=sensing,
        agents=int(cast("dict[str, int]", m["team"])["agents"]),
        provenance=cast("dict[str, Any]", m["provenance"]),
    )
