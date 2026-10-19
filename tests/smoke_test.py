"""The smoke test: the pinned Auxein is installed, and its `MLPPolicy` acts for a team with the scenario's sizes.

The sizes come from where they are defined: the observation from `sensors.observation_layout`, the brain from
`environment.brain`. Nothing is written out here, so a change of sensing changes this test with it.
"""

import json
from importlib import metadata

import auxein
import numpy as np
import pytest
from auxein.backend import Backend
from auxein.random import RandomStream

from auxein_scenarios.box_pushing.environment import ACTIONS, HIDDEN, brain
from auxein_scenarios.box_pushing.sensors import observation_layout

AUXEIN_TAG = "v0.3.3"
AUXEIN_COMMIT = "b423e90324d10b273b6847f9f07487f308e9319a"
AUXEIN_VERSION = "0.3.3"

OBSERVATION_SIZE = observation_layout().size
OUTPUTS = ACTIONS  # thrust and turning, in [-1, 1]

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
    policy = brain()
    # one hidden layer: its weights and biases, then the output layer's (MLPPolicy's genome layout)
    (hidden,) = HIDDEN
    assert policy.observation_size == OBSERVATION_SIZE
    assert policy.dim == OBSERVATION_SIZE * hidden + hidden + hidden * OUTPUTS + OUTPUTS
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
