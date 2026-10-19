"""The box-pushing world as an Auxein environment: sensing, the brain, the physics and the measurements.

`BoxPushingWorld` implements **both** of Auxein's environment interfaces around the physics core (`world.py`) and the sensors
(`sensors.py`):

- **`run_batch(agents, scenarios, rng)`**, the batched path `EpisodeEvaluator` takes when the decoder has `decode_batch`: every
  candidate of the batch on every scenario at once, measurements as `(n, s)` arrays on the run's backend. `agents` is anything
  with `act_batch(observations) -> actions` (Auxein's `BatchPolicy`), not only `MLPPolicy`'s networks.
- **`roles` and `run_episode(agents, scenario, rng)`**, the per-episode path `EpisodeEvaluator` requires of every environment:
  the same code with one candidate and one scenario, the single agent's `act` applied to the team's `(N, size)` observations.

**One step loop** (`BoxPushingWorld.run`) serves both paths and rendering: observe the state at the start of the step → the
policy's actions → one physics step, stopping early once every scenario of every candidate is done. An optional **hook** sees
every state with what the agents sensed in it, so that a video records exactly the episode that would be scored.

- **No randomness.** The world has none of its own: a scenario's layout is entirely in its params, and the brain is
  deterministic. `rng` (the agents' stream) is accepted, as Auxein's interfaces require, and unused.
- **The layout cache.** The padded `Layout` arrays of a scenario sequence are built once and cached by the scenarios'
  fingerprint (ids, seeds and params, as Auxein's `ScenarioSet` computes it) and the backend, so that a run's generations, which
  all use the training set, do not rebuild them.
- **Failures.** A scenario whose state becomes non-finite (a physics blow-up, or a policy returning NaN) is **frozen at its last
  finite state** and reported in `EpisodeBatchResult.failures` with the step it failed at; Auxein then records that episode, and
  so that candidate, as failed, and the rest of the batch goes on. Its measurements are meaningless and ignored, as Auxein's
  contract says.

The measurements, raw, never a score (the score is the fitness's, so weights can change without re-simulating):

| Measurement | Meaning |
|---|---|
| `delivered` | 1 if the box's centre entered the target zone, else 0 |
| `steps` | the step of delivery, or the steps simulated (the step limit) |
| `initial_distance` | the box's centre to the target's, at the start |
| `final_distance` | the same at the end (at delivery, if delivered) |
| `approach` | the mean over agents of `1 − min(1, g / r)`, `g` the agent's closest gap to the box during the episode |
| `max_pushers` | the most agents touching the box at the end of any step (step 0 included) |

The gap `g` is between the agent's disc and the box's surface (0 when touching), so an agent pressed against the box scores 1.
"""

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, cast

import numpy as np
from auxein.backend import Array, Backend, backend_of
from auxein.core import Status
from auxein.environments import EpisodeBatchResult, EpisodeFailure, EpisodeResult, MLPPolicy, Scenario
from auxein.random import RandomStream

from auxein_scenarios.box_pushing import scenarios as scenario_sets
from auxein_scenarios.box_pushing.config import SensingParams, WorldParams
from auxein_scenarios.box_pushing.sensors import Senses, observation_layout, sense
from auxein_scenarios.box_pushing.world import Layout, State, Trajectory, TrajectoryRecorder, initial_state, step
from auxein_scenarios.common.geometry import obb_contact

__all__ = [
    "ACTIONS",
    "HIDDEN",
    "MEASUREMENTS",
    "ROLE",
    "TOUCH_TOLERANCE",
    "BoxPushingWorld",
    "Rollout",
    "StepHook",
    "brain",
]

ROLE = "team"
"""The one evolved role: the team's shared brain."""

ACTIONS = 2
"""The brain's outputs: thrust and turning, in `[-1, 1]`."""

HIDDEN = (32,)
"""The brain's hidden layers by default: one of 32 `tanh` units."""

MEASUREMENTS = ("delivered", "steps", "initial_distance", "final_distance", "approach", "max_pushers")
"""The measurements every episode reports."""

TOUCH_TOLERANCE = 0.02
"""An agent touches the box when the gap between its disc and the box's surface is at most this (contacts overlap slightly, or
not at all for a light touch). Used by `max_pushers` and by the renderer's bright agents."""

StepHook = Callable[[int, State, Senses], None]
"""Called by `BoxPushingWorld.run` with the step number, the state at the start of that step and what the agents sensed in it;
then once more with the final state (and its senses)."""

Act = Callable[[Array], Array]


def brain(sensing: SensingParams | None = None, hidden: Sequence[int] = HIDDEN) -> MLPPolicy:
    """The team's brain: an `MLPPolicy` from the observation (its size from `sensors.observation_layout`) to
    the two actions, `tanh` hidden layers and `tanh` outputs."""
    return MLPPolicy(observation_layout(sensing).size, tuple(hidden), ACTIONS, activation="tanh", output="tanh")


def fingerprint(scenarios: Sequence[Scenario]) -> str:
    """A SHA-256 of the scenarios' ids, seeds and params in order: Auxein's `ScenarioSet` fingerprint, for any sequence."""
    document = [[s.id, s.seed, _json(s.params)] for s in scenarios]
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _json(params: Mapping[str, object]) -> object:
    to_json = getattr(params, "to_json", None)
    return to_json() if callable(to_json) else dict(params)


@dataclass(frozen=True)
class Rollout:
    """What `BoxPushingWorld.run` returns: the final state, the measurements as `(n, s)` arrays on the backend, and the
    scenarios that failed (`failed`, boolean) with the step each failed at (`failed_step`, 0 for the others)."""

    state: State
    measurements: dict[str, Array]
    failed: Array
    failed_step: Array


def _gap(state: State, params: WorldParams) -> Array:
    """`(n, s, N)`: the gap between each agent's disc and the box's surface (0 when they overlap)."""
    xp = state.backend.xp
    _, _, distance, _, _ = obb_contact(
        state.agent_x,
        state.agent_y,
        state.box_x[..., None],
        state.box_y[..., None],
        xp.cos(state.box_angle)[..., None],
        xp.sin(state.box_angle)[..., None],
        params.box.width / 2.0,
        params.box.height / 2.0,
    )
    return xp.clip(distance - params.agent.radius, 0.0, None)


def _finite(state: State) -> Array:
    """`(n, s)`: whether every number of the scenario's state is finite."""
    xp = state.backend.xp
    team = (state.agent_x, state.agent_y, state.agent_vx, state.agent_vy, state.agent_heading, state.agent_turn)
    ok = xp.all(xp.stack([xp.isfinite(a) for a in team], axis=-1), axis=(-2, -1))
    for a in (state.box_x, state.box_y, state.box_angle, state.box_vx, state.box_vy, state.box_spin):
        ok = ok & xp.isfinite(a)
    return ok


def _keep(mask: Array, old: State, new: State) -> State:
    """`new`, except the scenarios where `mask` (`(n, s)`) is true, which keep `old`."""
    xp = old.backend.xp
    values: dict[str, Array] = {}
    for f in fields(old):
        a, b = getattr(old, f.name), getattr(new, f.name)
        values[f.name] = xp.where(mask if b.ndim == mask.ndim else mask[..., None], a, b)
    return State(**values)


class BoxPushingWorld:
    """The world as an Auxein `Environment` and `BatchedEnvironment` (see the module docstring).

    - `params`, `sensing`: the world and what agents sense (defaults: `WorldParams()`, `SensingParams()`).
    - `steps`: the episode's step limit (default `params.simulation.max_steps`).
    - `backend`: where `run_episode` simulates, and `run_batch` when its agents do not say (numpy float64 by default); `run_batch`
      otherwise simulates where the agents' networks are, so that nothing crosses devices.
    """

    roles: tuple[str, ...] = (ROLE,)

    def __init__(
        self,
        params: WorldParams | None = None,
        sensing: SensingParams | None = None,
        *,
        steps: int | None = None,
        backend: Backend | None = None,
        cache_size: int = 4,
    ) -> None:
        self.params = params or WorldParams()
        self.sensing = sensing or SensingParams()
        self.steps = self.params.simulation.max_steps if steps is None else int(steps)
        if self.steps < 1:
            raise ValueError(f"steps must be at least 1, got {self.steps}")
        self.backend = backend
        self._cache: OrderedDict[tuple[str, str, str, str], Layout] = OrderedDict()
        self._cache_size = cache_size
        self.layouts_built = 0
        """How many times a `Layout` was built (not taken from the cache): for tests and the probe."""

    @property
    def observation_size(self) -> int:
        return observation_layout(self.sensing).size

    def describe(self) -> str:
        """A description that is the same in every process (Auxein compares it on resume): the parameters, as JSON."""
        sensing = {**self.sensing.to_dict(), "rays": self.sensing.rays, "sectors": self.sensing.sectors}
        document = {"world": self.params.to_dict(), "sensing": sensing, "steps": self.steps}
        return f"BoxPushingWorld({json.dumps(document, sort_keys=True, separators=(',', ':'))})"

    def __repr__(self) -> str:
        return f"BoxPushingWorld(steps={self.steps}, sensing={self.sensing})"

    # --- layouts ---

    def layout(self, scenarios: Sequence[Scenario], backend: Backend) -> Layout:
        """The padded `Layout` of `scenarios` on `backend`, from the cache when it holds it."""
        key = (fingerprint(scenarios), backend.name, backend.device, backend.precision)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        built = scenario_sets.layout_arrays([s.params for s in scenarios], backend)
        self.layouts_built += 1
        self._cache[key] = built
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return built

    # --- the step loop ---

    def run(self, act: Act, layout: Layout, n: int, *, hook: StepHook | None = None, stop_when_done: bool = True) -> Rollout:
        """The episode loop for `n` candidates on the layout's scenarios: observe, `act(observations) -> actions`, step; at most
        `self.steps` steps, ending early once every scenario of every candidate is done (one host read per step).

        `hook`, if given, is called with each step's starting state and senses, and once more with the final state."""
        params, sensing = self.params, self.sensing
        xp = backend_of(layout.box_x).xp
        state = initial_state(layout, n)
        r = sensing.range
        dx, dy = state.box_x - layout.target_x, state.box_y - layout.target_y
        initial_distance = xp.sqrt(dx * dx + dy * dy)
        gap = _gap(state, params)
        closest = gap
        pushers = xp.sum(xp.astype(gap <= TOUCH_TOLERANCE, xp.int32), axis=-1)
        failed = state.done  # all false
        failed_step = xp.zeros_like(state.steps)
        ran = 0
        for t in range(self.steps):
            ran = t + 1
            senses = sense(state, layout, params, sensing)
            if hook is not None:
                hook(t, state, senses)
            moved = step(state, act(senses.observation), layout, params)
            bad = ~_finite(moved) & ~failed
            moved = _keep(bad, state, moved)
            state = State(**{**{f.name: getattr(moved, f.name) for f in fields(moved)}, "done": moved.done | bad})
            failed = failed | bad
            failed_step = xp.where(bad, t + 1, failed_step)
            gap = _gap(state, params)
            closest = xp.minimum(closest, gap)
            pushers = xp.maximum(pushers, xp.sum(xp.astype(gap <= TOUCH_TOLERANCE, xp.int32), axis=-1))
            if stop_when_done and bool(xp.all(state.done)):
                break
        if hook is not None:
            hook(ran, state, sense(state, layout, params, sensing))
        floats = state.box_x.dtype
        dx, dy = state.box_x - layout.target_x, state.box_y - layout.target_y
        measurements = {
            "delivered": xp.astype(state.done & ~failed, floats),
            "steps": xp.astype(state.steps, floats),
            "initial_distance": initial_distance + 0.0 * dx,
            "final_distance": xp.sqrt(dx * dx + dy * dy),
            "approach": xp.mean(1.0 - xp.clip(closest / r, None, 1.0), axis=-1),
            "max_pushers": xp.astype(pushers, floats),
        }
        return Rollout(state, measurements, failed, failed_step)

    def watch(self, policy: Any, scenarios: Sequence[Scenario], *, stop_when_done: bool = True) -> tuple[Trajectory, Rollout]:
        """Runs `policy` (a batch policy, as `run_batch` takes) on `scenarios` through the same step loop as training, recording
        every step for rendering: the trajectory and the rollout with its measurements. For videos, which show a few
        candidates on a few scenarios (the trajectory is kept on the host, every step)."""
        backend, n = self._batch_of(policy)
        recorder = TrajectoryRecorder()
        rollout = self.run(
            cast("Act", policy.act_batch), self.layout(scenarios, backend), n, hook=lambda t, state, _: recorder(t, state),
            stop_when_done=stop_when_done,
        )  # fmt: skip
        return recorder.trajectory(), rollout

    # --- Auxein's interfaces ---

    def run_batch(self, agents: Any, scenarios: Sequence[Scenario], rng: RandomStream) -> EpisodeBatchResult:
        """Every candidate of `agents` (anything with `act_batch`, and `n` or `weights` to say the batch size) on every
        scenario. `rng` is unused: the world has no randomness (module docstring)."""
        backend, n = self._batch_of(agents)
        layout = self.layout(scenarios, backend)
        rollout = self.run(cast("Act", agents.act_batch), layout, n)
        return EpisodeBatchResult(rollout.measurements, self._failures(rollout, scenarios))

    def run_episode(self, agents: Mapping[str, Any], scenario: Scenario, rng: RandomStream) -> EpisodeResult:
        """One candidate's agent (with `act`, as `MLPPolicy.decode` makes) on one scenario. `rng` is unused."""
        agent = agents[ROLE]
        backend = self.backend or Backend("numpy", "cpu", "float64")
        layout = self.layout([scenario], backend)
        team = layout.agents

        def act(observations: Array) -> Array:
            actions = np.asarray(agent.act(backend.to_numpy(observations)[0, 0]), dtype=np.float64)
            return backend.asarray(actions.reshape(1, 1, team, ACTIONS))

        rollout = self.run(act, layout, 1)
        failures = self._failures(rollout, [scenario])
        if failures:
            return EpisodeResult.failed(failures[(0, 0)].error)
        return EpisodeResult({name: float(backend.to_numpy(value)[0, 0]) for name, value in rollout.measurements.items()})

    def _batch_of(self, agents: Any) -> tuple[Backend, int]:
        """The backend and the number of candidates of a batch of agents: `MLPAgentBatch` holds its weights (on the run's
        backend) and `n`; other policies say `n` and, optionally, `backend`."""
        weights = getattr(agents, "weights", None)
        backend = self.backend
        if backend is None:
            if weights:
                backend = backend_of(weights[0])
            elif isinstance(getattr(agents, "backend", None), Backend):
                backend = cast("Backend", agents.backend)
            else:
                raise TypeError(f"cannot tell the backend of {type(agents).__name__}: give it a `backend` or the world backend=")
        n = getattr(agents, "n", None)
        if n is None:
            raise TypeError(f"{type(agents).__name__} does not say how many candidates it holds (`n`)")
        return backend, int(n)

    @staticmethod
    def _failures(rollout: Rollout, scenarios: Sequence[Scenario]) -> dict[tuple[int, int], EpisodeFailure]:
        backend = rollout.state.backend
        if not bool(backend.xp.any(rollout.failed)):
            return {}
        failed, at = backend.to_numpy(rollout.failed), backend.to_numpy(rollout.failed_step)
        out: dict[tuple[int, int], EpisodeFailure] = {}
        for row, column in zip(*np.nonzero(failed), strict=True):
            row, column = int(row), int(column)
            out[(row, column)] = EpisodeFailure(
                Status.FAILED,
                f"the state became non-finite at step {int(at[row, column])} in scenario {scenarios[column].id!r} "
                "(a physics blow-up, or non-finite actions): the scenario was frozen and its measurements are not meaningful",
            )
        return out
