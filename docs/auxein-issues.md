# Auxein issues found from the scenarios

The scenarios are Auxein's first real user. Every Auxein bug they turn up, and every generic feature they turn out to need, is
logged here, newest first, following the protocol in [`CLAUDE.md`](../CLAUDE.md) ("Working on Auxein from this repository").

Each entry has this format:

```markdown
## YYYY-MM-DD: <short title>

- **Kind:** bug | feature request
- **Symptom:** what went wrong, as seen from the scenario.
- **Minimal reproduction:** a few lines that use Auxein (and its own test fixtures) only, nothing from this repository.
- **Auxein branch / PR:** `fix/<short-description>`, and the PR link once opened (none for a feature request: the owner decides).
- **Status:** open | fixed in PR <link> | released in tag `vX.Y.Z`
- **Workaround:** none, or where the `# AUXEIN-WORKAROUND: <this entry's title>` comments are, and what removing them takes.
```

---

## 2026-10-16: one Ctrl-C does not stop a batched run

- **Kind:** bug
- **Symptom:** `box-pushing train` (a batched `EpisodeEvaluator`, inline) went on training after Ctrl-C, generation after
  generation; only a second Ctrl-C stopped it, in the middle of a generation. `auxein.run` drives its loop with
  `asyncio.run`, which turns the first Ctrl-C into a cancellation of the driver's task, delivered only when the task next
  yields to the event loop; the live loop yielded only while replaying (`_step`: `if recorded: await asyncio.sleep(0)`), and
  an evaluator that computes on the driver's thread never yields, so the cancellation waited for the end of the run. Auxein's
  own interrupt tests raise `KeyboardInterrupt` from an evaluator or a callback, never a real SIGINT.
- **Minimal reproduction** (in a clone of Auxein): a recorded `auxein.run` of the team fixture (`tests/support/team.py`) with a
  large budget, in a subprocess started with SIGINT at its default; send it one SIGINT after 3 s: it is still running 10 s
  later; a second SIGINT stops it (status `interrupted`).
- **Auxein branch / PR:** `fix/ctrl-c-between-batches` (pushed; https://github.com/auxein/auxein/pull/new/fix/ctrl-c-between-batches):
  the driver yields to the event loop after every tell, so one Ctrl-C stops a run after the batch in progress, recorded as
  interrupted with a checkpoint. Regression tests in `tests/driver/run_recording_test.py` (a cancellation during a batch, and
  a real SIGINT to a subprocess); design doc §10.4 and `CHANGELOG.md` (Unreleased) updated.
- **Status:** released in tag `v0.3.3` (`b423e90`), merged as Auxein PR #51.
- **Workaround:** removed on 2026-10-17, when the scenarios moved to `v0.3.3`. It was `train.StopOnCtrlC`, a SIGINT handler
  of the scenario's own and a progress callback that raised `KeyboardInterrupt` after the generation's tell. Auxein now stops
  the run itself after the batch in progress; `tests/box_pushing/train_test.py` sends one real SIGINT to `box-pushing train`
  and checks that it ends with the interrupt's exit code and a resumable, interrupted recording.

## 2026-10-16: a parameterised aggregator source hides its parameters

- **Kind:** bug
- **Symptom:** the box-pushing score is a parameterised source (`fitness.EpisodeScore(weights, max_steps)`), as M2a decided.
  Auxein labels a callable source by its `__qualname__`, which an instance does not have, so the label is the class's name
  whatever the weights: `mean(<EpisodeScore>)`, and the same for a `functools.partial` (`mean(<partial>)`). The evaluator's
  description, which `resume` compares, then does not change with a weight, so **resuming a run with changed fitness weights
  was accepted** and the rest of the run judged with the new ones, against "only the budget may change" (core design §10.4).
- **Minimal reproduction** (in a clone of Auxein, with its team fixture):

  ```python
  class Weighted:
      def __init__(self, weight): self.weight = weight
      def __call__(self, m): return m["final_distance"] + self.weight * m["effort"]
      def __repr__(self): return f"Weighted(weight={self.weight})"

  mean(Weighted(0.02)).label == mean(Weighted(5.0)).label        # True: "mean(<Weighted>)"
  # auxein.run(...) with Weighted(0.02), then auxein.resume(...) with Weighted(5.0) and a larger budget: accepted
  ```

- **Auxein branch / PR:** `fix/parameterised-source-labels` (pushed;
  https://github.com/auxein/auxein/pull/new/fix/parameterised-source-labels). A source is labelled by its `describe()`, its
  class's own `__repr__`, or, for a partial, its function and bound arguments; functions and plain objects keep their labels.
  Regression tests in `tests/aggregators/aggregator_test.py` and `tests/driver/episode_run_test.py` (a changed weight is
  refused on resume); Auxein's design doc §6.5 and `CHANGELOG.md` (Unreleased) updated. The full suite passes (3,747 tests).
- **Status:** released in tag `v0.3.3` (`b423e90`), merged as Auxein PR #50 (with a follow-up for Python 3.14, where a
  `functools.partial` counts as a routine for `inspect.isroutine`, so it is now recognised first).
- **Workaround:** removed on 2026-10-17, when the scenarios moved to `v0.3.3`. `fitness.EpisodeScore` and
  `fitness.EpisodeProgress` had set an instance `__qualname__` to their description; Auxein now uses `describe()`, which
  gives the same label, so runs recorded with the workaround stay resumable. The training command also compares the whole
  config on resume (`train.py`), weights included.

## 2026-10-16: a batched environment cannot learn the batch's size or backend from the protocol

- **Kind:** feature request
- **Symptom:** `BatchedEnvironment.run_batch(agents, scenarios, rng)` must build its state for `n` candidates on the run's
  backend and device, but nothing in the interface says what `n` or the backend is. `agents` is opaque (`AgentBatch = object`),
  and `BatchPolicy`, the protocol a batched environment is meant to be written against (core design §6.2), has only
  `act_batch`. The driver happens to build the batch stream `rng` on the run's backend, but that is not documented. So a
  batched world reads `MLPAgentBatch`'s own attributes: Auxein's team fixture takes the backend from `agents.weights[0]`
  (`tests/support/team.py`), and the scenarios do the same, plus `agents.n`. A policy of the world's own (a scripted
  controller written as a `BatchPolicy`) has to carry `n` and `backend` attributes by convention.
- **Minimal reproduction:** write `run_batch` for `tests/support/team.py`'s world against `BatchPolicy` only (no `weights`):
  the initial state `(n, s, agents)` cannot be built before the first `act_batch`, which needs the observations of that state.
- **What would help** (the owner decides): `n` and `backend` on `BatchPolicy` (`MLPAgentBatch` has `n`, and its backend
  privately), or a documented guarantee that `run_batch`'s `rng` is on the run's backend (it is, today: `episode-batch` streams
  use the run's backend).
- **Auxein branch / PR:** none (a feature request).
- **Status:** open.
- **Workaround:** not a workaround of a bug, so not marked: `BoxPushingWorld._batch_of` reads `agents.weights` and `agents.n`
  (`MLPAgentBatch`), or `agents.backend` and `agents.n` (the scenarios' own policies), or the world's `backend=`.

## 2026-10-14: v0.3.1 reports version 0.3.0

- **Kind:** bug (release packaging)
- **Symptom:** installed from the tag `v0.3.1`, `auxein.__version__` is `"0.3.0"`, so the scaffold's smoke test cannot check
  for `0.3.1`. The tag (a lightweight tag on `5636796`, the head of `master`) carries `version = "0.3.0"` in `pyproject.toml`, and
  its `CHANGELOG.md` still lists the release's changes under "Unreleased". A run's `metadata.json` records the versions it ran
  with, so recordings made with `v0.3.1` will say `0.3.0` too.
- **Minimal reproduction:**

  ```
  uv venv /tmp/v && uv pip install --python /tmp/v "auxein @ git+https://github.com/auxein/auxein@v0.3.1"
  /tmp/v/bin/python -c "import auxein; print(auxein.__version__)"    # prints 0.3.0
  ```

- **Auxein branch / PR:** none. The fix is a version bump (and the changelog heading), which only the owner makes; a new tag is
  needed for it to reach this repository. A CI check in Auxein that a tag matches `pyproject.toml` would prevent a repeat (a
  feature request for the owner to decide).
- **Status:** released in tag `v0.3.2`. The tag was first created on the `v0.3.1` commit (`5636796`), so it still reported
  `0.3.0`; on 2026-10-14 it was moved to the release commit `d63e9c7` ("Version 0.3.2"), which reports `0.3.2`.
- **Workaround:** removed on 2026-10-14, when the scenarios moved to `v0.3.2` at `d63e9c7`; `tests/smoke_test.py` now checks
  the version (`0.3.2`) as well as the tag and the commit.
