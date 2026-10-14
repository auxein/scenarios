"""The scaffold's smoke test: the pinned Auxein is installed, and its `MLPPolicy` acts for a team with the design doc's sizes.

The sizes are those of `docs/design/box-pushing.md` §3.6. They are written out here rather than imported because M0 has no
sensor code yet; once `sensors.py` defines the observation layout (M1), this test should take them from there.
"""

import json
from importlib import metadata

import auxein
import numpy as np
import pytest
from auxein.backend import Backend
from auxein.environments import MLPPolicy
from auxein.random import RandomStream

AUXEIN_TAG = "v0.3.2"
AUXEIN_COMMIT = "d63e9c77942a912b9243dddfeeff29411da48a93"
AUXEIN_VERSION = "0.3.2"

OBSERVATION_SIZE = 43  # target 3 + rays 12 × 2 + box cue 6 + teammate sectors 8 + proprioception 2
HIDDEN = (32,)
OUTPUTS = 2  # thrust and turning, in [-1, 1]
GENOME_LENGTH = 1_474  # 43 × 32 + 32, then 32 × 2 + 2

CANDIDATES, SCENARIOS, AGENTS = 4, 3, 20


def test_auxein_is_the_pinned_tag() -> None:
    """The installed Auxein comes from the tag in `[tool.uv.sources]`, not from PyPI or a local clone (PEP 610 metadata)."""
    direct_url = json.loads(metadata.distribution("auxein").read_text("direct_url.json") or "{}")
    vcs = direct_url.get("vcs_info", {})
    assert direct_url.get("url") == "https://github.com/auxein/auxein"
    assert vcs.get("requested_revision") == AUXEIN_TAG
    assert vcs.get("commit_id") == AUXEIN_COMMIT


def test_auxein_version() -> None:
    assert auxein.__version__ == AUXEIN_VERSION


def test_auxein_has_what_the_design_relies_on() -> None:
    """The features the precondition names: progress callbacks and the reader's per-generation queries."""
    from auxein.callbacks import print_progress, stop_when
    from auxein.recording.reader import RunReader

    assert callable(print_progress) and callable(stop_when)
    for query in ("best_up_to", "champions", "generation_history"):
        assert callable(getattr(RunReader, query))


def _act(backend: Backend) -> None:
    policy = MLPPolicy(OBSERVATION_SIZE, HIDDEN, OUTPUTS, activation="tanh", output="tanh")
    assert policy.dim == GENOME_LENGTH
    rng = RandomStream(np.random.SeedSequence(7), backend)
    # sampled from the policy's own space, as the GA's first generation is
    genomes = policy.space().sample_genomes(CANDIDATES, rng, backend)
    observations = rng.normal((CANDIDATES, SCENARIOS, AGENTS, OBSERVATION_SIZE))

    actions = policy.decode_batch(genomes).act_batch(observations)

    assert tuple(actions.shape) == (CANDIDATES, SCENARIOS, AGENTS, OUTPUTS)
    values = backend.to_numpy(actions)
    assert np.all(np.abs(values) <= 1.0)
    # parameter sharing: one candidate's network applied to one agent's observation is what the batch gave that agent
    single = policy.decode(backend.to_numpy(genomes)[1]).act(backend.to_numpy(observations)[1, 2])
    np.testing.assert_allclose(np.asarray(single), values[1, 2], rtol=1e-5, atol=1e-6)


def test_mlp_policy_acts_for_a_team_on_numpy() -> None:
    _act(Backend("numpy", "cpu", "float64"))


def test_mlp_policy_acts_for_a_team_on_torch() -> None:
    pytest.importorskip("torch")
    _act(Backend("torch", "cpu", "float32"))
