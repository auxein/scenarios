"""Deploying trained brains on scenarios they have not seen: their results, the baselines', and one video.

`deploy(brains)` loads each brain file (`brain.py`) and runs it **in its own world**: the world, sensing and team size it was
trained with, from its metadata; only the team size can be changed (`agents`), and the brain is the same whatever the team, as
every agent runs it on its own senses. Its scenarios are `scenarios` layouts of `difficulty` from `seed`, drawn by the
generator's held-out stream for the brain's world and team:

- **By default the seed is fresh**: a random one that is neither the training nor the held-out seed of any brain given, so
  the scenarios are new to every brain. The deployment records it, so it can be repeated with `seed`.
- **`seed` may name any seed**, the held-out one included: deploying a run's final brain on its held-out seed repeats the
  run's held-out evaluation, episode for episode.

**The baselines** (random brains and the senses-only controller, `evaluation.baselines`) run on the same scenarios, in the
same world, so every brain's numbers can be read against what an untrained team and a hand-written controller do. Brains
from different worlds get different scenarios (a layout depends on the arena, the box, the agents' size and the team) and
baselines of their own.

The deployment's directory (`deployments/<name>-<timestamp>` by default, `name` the first brain's run's) holds:

- `deploy.toml`: the effective options (the seed among them) and each brain's provenance;
- `results.csv`: one row per brain (or baseline member) per scenario: delivered, steps, progress, `max_pushers`, the box's
  starting distance to the target, the obstacles, the score with the brain's own fitness weights;
- `summary.md`: per brain and baseline, success, mean progress, mean `max_pushers`, the median delivery step, and success by
  the box's starting distance;
- `deployment.mp4`: **a panel per brain on the same scenarios**, the first `video_scenarios` of the set (not picked for how they
  went), each captioned with the brain (its run and generation), the scenario and how its episode ended. Deploying a run's
  early, middle and final brains together shows what training changed.
"""

import csv
import io
import json
import math
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from auxein.backend import Backend
from auxein.environments import ScenarioSet

from auxein_scenarios.box_pushing import evaluation, scenarios
from auxein_scenarios.box_pushing.brain import Brain, BrainFileError, load_brain
from auxein_scenarios.box_pushing.config import FitnessConfig
from auxein_scenarios.box_pushing.environment import BoxPushingWorld
from auxein_scenarios.box_pushing.evaluation import Episode
from auxein_scenarios.common.cli import backend_from
from auxein_scenarios.common.runs import timestamped

__all__ = [
    "DEPLOYMENTS",
    "DIFFICULTIES",
    "MARKDOWN_FILE",
    "OPTIONS_FILE",
    "RESULTS_FILE",
    "VIDEO_FILE",
    "DeployError",
    "Deployed",
    "Deployment",
    "deploy",
    "fresh_seed",
]

DEPLOYMENTS = Path("deployments")
"""Where deployments go by default (ignored by git)."""
OPTIONS_FILE = "deploy.toml"
RESULTS_FILE = "results.csv"
MARKDOWN_FILE = "summary.md"
VIDEO_FILE = "deployment.mp4"

DIFFICULTIES = {"full": scenarios.FULL, "easy": scenarios.EASY}
SEEDS = 2**31
"""Fresh seeds are drawn below this."""

Print = Callable[[str], None]


class DeployError(ValueError):
    """Options or brains that cannot be deployed: no brain, a bad count, a brain file this code cannot run."""


def fresh_seed(brains: Sequence[Brain]) -> int:
    """A random seed that is not the training or held-out seed of any of `brains`, so that their scenarios are new."""
    taken = {seed for b in brains for seed in b.seeds}
    while True:
        seed = secrets.randbelow(SEEDS)
        if seed not in taken:
            return seed


@dataclass(frozen=True)
class Deployed:
    """One brain (or baseline) deployed: who it is, the world and scenarios it ran in, and its episodes (for a baseline of
    several members, theirs one after another)."""

    name: str
    kind: str
    brain: Brain | None
    world: BoxPushingWorld
    scenarios: ScenarioSet
    episodes: tuple[Episode, ...]
    members: int = 1


@dataclass(frozen=True)
class Deployment:
    """What `deploy` did: its directory, its options, the brains and baselines deployed."""

    directory: Path
    options: dict[str, object]
    brains: tuple[Deployed, ...]
    baselines: tuple[Deployed, ...]


def _scorer(brain: Brain | None, world: BoxPushingWorld) -> evaluation.Scorer:
    weights = FitnessConfig(**brain.provenance["fitness"]) if brain is not None else FitnessConfig()
    return evaluation.scorer(weights, world.steps)


def _outcome(m: Mapping[str, float]) -> str:
    d0, d = m["initial_distance"], m["final_distance"]
    if m["delivered"] > 0.5:
        return f"delivered at step {int(m['steps'])}"
    return f"not delivered  ·  progress {(d0 - d) / d0 if d0 > 0 else 0.0:+.2f}"


def deploy(
    paths: Sequence[str | Path],
    *,
    count: int = 64,
    difficulty: str = "full",
    seed: int | None = None,
    agents: int | None = None,
    backend: Backend | None = None,
    video: bool = True,
    video_scenarios: int = 2,
    out_dir: str | Path | None = None,
    out: Print = print,
) -> Deployment:
    """Deploys the brains in `paths` (module docstring); returns what it did, its files written."""
    if not paths:
        raise DeployError("give at least one brain file")
    if count < 1 or (agents is not None and agents < 1) or video_scenarios < 1:
        raise DeployError("--scenarios, --agents and --video-scenarios must be at least 1")
    if difficulty not in DIFFICULTIES:
        raise DeployError(f"--difficulty is one of {list(DIFFICULTIES)}, got {difficulty!r}")
    try:
        brains = [load_brain(p) for p in paths]
    except BrainFileError as error:
        raise DeployError(str(error)) from None
    chosen = backend if backend is not None else backend_from()[0]
    seed = fresh_seed(brains) if seed is None else seed
    name = str(brains[0].provenance.get("name", "deploy"))
    target = Path(out_dir) if out_dir is not None else timestamped(DEPLOYMENTS, name)
    target.mkdir(parents=True, exist_ok=True)
    out(
        f"deploying {len(brains)} brain{'s' if len(brains) != 1 else ''} on {count} {difficulty} scenarios from seed {seed} "
        f"({chosen.name} {chosen.device} {chosen.precision}), into {target}"
    )

    sets: dict[str, ScenarioSet] = {}
    deployed: list[Deployed] = []
    for b in brains:
        team = agents if agents is not None else b.agents
        world = BoxPushingWorld(b.world, b.sensing)
        key = json.dumps([b.world.to_dict(), team], sort_keys=True)
        if key not in sets:
            sets[key] = scenarios.held_out_set(
                team, b.world, seed=seed, count=count, difficulty=DIFFICULTIES[difficulty], prefix="scenario"
            )
        chosen_set = sets[key]
        genome = chosen.asarray(b.genome.reshape(1, -1))
        (measured,) = evaluation.measure(world, b.policy.decode_batch(genome), list(chosen_set), chosen)
        episodes = evaluation.episodes(list(chosen_set), measured, _scorer(b, world))
        deployed.append(Deployed(b.label, "brain", b, world, chosen_set, episodes))
        r = evaluation.summarise(episodes)
        out(f"  {b.label}: {r.delivered} of {r.episodes - r.failed} delivered ({r.success:.0%}), progress {r.progress:+.2f}")

    baselines: list[Deployed] = []
    groups = {json.dumps([d.brain.world.to_dict(), len(d.scenarios[0].params["agents"])], sort_keys=True): d  # type: ignore[union-attr]
              for d in deployed}  # fmt: skip
    for i, first in enumerate(groups.values()):
        assert first.brain is not None
        suffix = f" (world {i + 1})" if len(groups) > 1 else ""
        score = _scorer(None, first.world)
        for kind, members in evaluation.baselines(first.world, first.brain.policy, list(first.scenarios), chosen, score).items():
            flat = tuple(e for member in members for e in member)
            baselines.append(Deployed(kind + suffix, "baseline", None, first.world, first.scenarios, flat, len(members)))

    options: dict[str, object] = {
        "scenarios": count, "difficulty": difficulty, "seed": seed, "agents": agents if agents is not None else "each brain's own",
        "backend": chosen.name, "device": chosen.device, "precision": chosen.precision, "video": video,
        "video_scenarios": video_scenarios,
    }  # fmt: skip
    deployment = Deployment(target, options, tuple(deployed), tuple(baselines))
    (target / OPTIONS_FILE).write_text(_options_toml(deployment))
    (target / RESULTS_FILE).write_text(_results_csv(deployment))
    (target / MARKDOWN_FILE).write_text(_summary(deployment))
    if video:
        out(f"rendering {target / VIDEO_FILE}")
        _video(deployment, min(video_scenarios, count), chosen)
    out(f"done: {', '.join(p.name for p in sorted(target.iterdir()))} in {target}")
    return deployment


# --- the files ---


def _toml(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else ("nan" if math.isnan(value) else ("inf" if value > 0 else "-inf"))
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)  # a TOML basic string: the same escapes as JSON, UTF-8 kept
    if value is None:
        return '""'
    if isinstance(value, Mapping):
        items = cast("Mapping[str, object]", value)
        return "{ " + ", ".join(f"{k} = {_toml(v)}" for k, v in items.items()) + " }"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml(v) for v in cast("Sequence[object]", value)) + "]"
    raise TypeError(f"cannot write {value!r} to TOML")  # pragma: no cover - a programming error


def _options_toml(deployment: Deployment) -> str:
    lines = ["# what `scenarios box-pushing deploy` was asked, and the brains it ran", "", "[options]"]
    lines += [f"{key} = {_toml(value)}" for key, value in deployment.options.items()]
    for d in deployment.brains:
        assert d.brain is not None
        lines += ["", "[[brains]]", f"path = {_toml(str(d.brain.path))}", f"label = {_toml(d.name)}",
                  f"agents = {len(cast('list[object]', d.scenarios[0].params['agents']))}"]  # fmt: skip
        lines += [f"{key} = {_toml(value)}" for key, value in d.brain.provenance.items()]
    return "\n".join(lines) + "\n"


def _number(value: float, digits: int = 6) -> str:
    return "" if math.isnan(value) else f"{value:.{digits}f}"


def _results_csv(deployment: Deployment) -> str:
    rows: list[dict[str, object]] = []
    for d in (*deployment.brains, *deployment.baselines):
        per = len(d.scenarios)
        for i, e in enumerate(d.episodes):
            member = f" {i // per + 1}" if d.members > 1 else ""
            rows.append(
                {
                    "subject": d.name + member,
                    "kind": d.kind,
                    "brain": str(d.brain.path) if d.brain is not None else "",
                    "scenario_id": e.scenario_id,
                    "difficulty": e.difficulty,
                    "distance": _number(e.distance, 3),
                    "obstacles": e.obstacles,
                    "ok": int(e.ok),
                    "delivered": int(e.delivered),
                    "steps": e.steps,
                    "progress": _number(e.progress),
                    "max_pushers": e.max_pushers,
                    "score": _number(e.score),
                }
            )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _summary(deployment: Deployment) -> str:
    o = deployment.options
    team = "its own team size" if isinstance(o["agents"], str) else f"{o['agents']} agents"
    lines = [
        "# Deployment",
        "",
        f"{o['scenarios']} {o['difficulty']} scenarios from seed {o['seed']}, new to every brain unless the seed is one of "
        f"theirs; {o['backend']} {o['device']} {o['precision']}. Each brain runs in the world it was trained in, with "
        f"{team}. The baselines run on the same "
        "scenarios: random brains (what an untrained team does) and the hand-written senses-only controller (what these senses "
        "allow).",
        "",
        "| brain | delivered | progress | `max_pushers` | delivery step (median) | 6–8 units | 8–10 | 10–12 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in (*deployment.brains, *deployment.baselines):
        r = evaluation.summarise(d.episodes)
        ran = r.episodes - r.failed
        who = d.name if d.kind == "brain" else f"*{d.name}*"
        success = "" if math.isnan(r.success) else f" ({r.success:.0%})"
        steps = "" if math.isnan(r.delivery_steps) else f"{r.delivery_steps:.0f}"
        bands = " | ".join(f"{delivered}/{n}" if n else "" for _, delivered, n in evaluation.bands(d.episodes))
        lines.append(f"| {who} | {r.delivered}/{ran}{success} | {r.progress:+.2f} | {r.max_pushers:.1f} | {steps} | {bands} |")
    failed = sum(evaluation.summarise(d.episodes).failed for d in (*deployment.brains, *deployment.baselines))
    lines += [
        "",
        "Progress is how far the box moved toward the target, as a share of its starting distance, averaged over the episodes; "
        "`max_pushers` is the most agents touching the box at once, averaged; the bands are full-difficulty deliveries by the "
        "box's starting distance to the target, which most decides whether a push finishes in time.",
    ]
    if failed:
        lines += ["", f"{failed} episodes failed (a non-finite state) and are left out of the numbers."]
    return "\n".join(lines) + "\n"


# --- the video ---


def _video(deployment: Deployment, k: int, backend: Backend) -> Path:
    from auxein_scenarios.box_pushing.render import Panel, render_panels  # matplotlib, only when rendering

    panels: list[Panel] = []
    for d in deployment.brains:
        assert d.brain is not None
        shown = list(d.scenarios)[:k]
        genome = backend.asarray(np.asarray(d.brain.genome, dtype=np.float64).reshape(1, -1))
        trajectory, rollout = d.world.watch(d.brain.policy.decode_batch(genome), shown)
        measured = {name: np.asarray(backend.to_numpy(v), dtype=np.float64) for name, v in rollout.measurements.items()}
        for j, scenario in enumerate(shown):
            m = {name: float(values[0, j]) for name, values in measured.items()}
            panels.append(
                Panel(
                    trajectory,
                    0,
                    j,
                    scenario.params,
                    d.world.params,
                    d.name,
                    f"{scenario.id} · {scenario.params['difficulty']}",
                    _outcome(m),
                )  # fmt: skip
            )
    o = deployment.options
    n = len(deployment.brains)
    return render_panels(
        panels,
        deployment.directory / VIDEO_FILE,
        kicker=f"Deployment · {n} brain{'s' if n != 1 else ''} · seed {o['seed']}",
        title=f"{n} brain{'s' if n != 1 else ''} on {k} new scenario{'s' if k != 1 else ''}",
        subtitle=f"A row per brain, each in the world it was trained in, on the first {k} of the {o['scenarios']} scenarios "
        "deployed (not chosen for how they went). Bright agents touch the box; the box lights up while it moves.",
        columns=k,
    )
