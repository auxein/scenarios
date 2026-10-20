"""Brain files: the genome back bit for bit, the metadata a brain carries, refusals of brains this code cannot run, and plain
tensors any safetensors reader understands."""

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from auxein_scenarios.box_pushing.brain import FORMAT, Brain, BrainFileError, layout_of, load_brain, save_brain
from auxein_scenarios.box_pushing.config import TrainingConfig, parse_config
from auxein_scenarios.box_pushing.environment import brain
from auxein_scenarios.box_pushing.sensors import observation_layout
from auxein_scenarios.common.brains import load_mlp, save_mlp


def genome(config: TrainingConfig, seed: int = 0) -> np.ndarray:
    """A genome of the config's brain, in the run's precision (a float32 run's genomes are float32 values)."""
    values = np.random.default_rng(seed).uniform(-1.0, 1.0, brain(config.sensing, config.brain.hidden).dim)
    return values.astype(np.float32).astype(np.float64) if config.run.precision == "float32" else values


def save(path: Path, config: TrainingConfig, values: np.ndarray) -> Path:
    return save_brain(
        path, values, config, run_dir=Path("runs/try-20261020-120000"), generation=7, candidate_id=42, score=1.5, success=0.25
    )


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_the_genome_round_trips_bit_for_bit(tmp_path: Path, precision: str) -> None:
    config = parse_config(f"[run]\nprecision = '{precision}'\n[brain]\nhidden = [16, 8]")
    values = genome(config)
    loaded = load_brain(save(tmp_path / "b.safetensors", config, values))
    assert loaded.genome.dtype == np.float64 and np.array_equal(loaded.genome, values)
    assert loaded.precision == precision and loaded.policy.hidden == (16, 8)
    # and the network it decodes to acts as the original genome's
    observation = np.random.default_rng(1).normal(size=(5, loaded.policy.observation_size))
    original = brain(config.sensing, config.brain.hidden).decode(values).act(observation)
    np.testing.assert_array_equal(np.asarray(loaded.policy.decode(loaded.genome).act(observation)), np.asarray(original))


def test_a_genome_the_precision_cannot_hold_is_refused(tmp_path: Path) -> None:
    config = TrainingConfig()
    with pytest.raises(ValueError, match="not exactly representable in float32"):
        save(tmp_path / "b.safetensors", config, np.random.default_rng(0).uniform(-1, 1, brain().dim))


def test_the_metadata_says_what_the_brain_is_and_where_it_came_from(tmp_path: Path) -> None:
    config = parse_config("name = 'try'\n[team]\nagents = 12\n[sensing]\nrange = 6.0\n[fitness]\ntime = 2.0\n[world.friction]\nk = 3")
    loaded: Brain = load_brain(save(tmp_path / "b.safetensors", config, genome(config)))
    raw = load_mlp(tmp_path / "b.safetensors", FORMAT).metadata
    assert raw["format"] == FORMAT and raw["scenario"] == "box-pushing"
    size = observation_layout().size
    assert raw["brain"] == {"observation_size": size, "hidden": [32], "outputs": 2, "activation": "tanh", "output": "tanh"}
    assert raw["observation_layout"] == layout_of(config.sensing) and sum(n for _, n in raw["observation_layout"]) == size  # type: ignore[misc]
    assert raw["sensing"] == {"range": 6.0, "rays": 12, "sectors": 8} and raw["team"] == {"agents": 12}
    assert loaded.world == config.world and loaded.world.friction.k == 3 and loaded.sensing == config.sensing and loaded.agents == 12
    p = loaded.provenance
    assert (p["name"], p["run"], p["generation"], p["candidate_id"], p["final"]) == ("try", "try-20261020-120000", 7, 42, False)
    assert (p["training_score"], p["training_success"]) == (1.5, 0.25)
    assert p["fitness"] == {"progress": 1.0, "delivery": 2.0, "time": 2.0, "approach": 0.1}
    assert p["scenarios"] == {"easy": 4, "full": 28, "seed": 1, "held_out": 64, "held_out_seed": 2} and loaded.seeds == (1, 2)
    assert {"auxein_version", "scenarios_version", "scenarios_commit", "created", "run_seed", "run_dir"} <= set(p)
    assert loaded.label == "try-20261020-120000 · generation 7"


def test_the_file_is_plain_tensors_of_a_few_kilobytes(tmp_path: Path) -> None:
    config = TrainingConfig()
    path = save(tmp_path / "b.safetensors", config, genome(config))
    tensors = load_file(str(path))
    assert {name: (t.shape, str(t.dtype)) for name, t in tensors.items()} == {
        "layers.0.weight": ((32, observation_layout().size), "float32"),
        "layers.0.bias": ((32,), "float32"),
        "layers.1.weight": ((2, 32), "float32"),
        "layers.1.bias": ((2,), "float32"),
    }
    assert path.stat().st_size < 16_000, "a few kilobytes: the float32 weights and the metadata"
    # the layers are the genome's: per layer the weights (out, in) row by row, then the biases
    values = genome(config)
    np.testing.assert_array_equal(
        load_file(str(save(tmp_path / "c.safetensors", config, values)))["layers.0.weight"].reshape(-1),
        values[: 32 * observation_layout().size],
    )


def test_another_format_is_refused(tmp_path: Path) -> None:
    path = save_mlp(tmp_path / "other.safetensors", np.zeros(brain().dim), brain().layers, "float64", {"format": "someone-else/brain/1"})
    with pytest.raises(BrainFileError, match="format 'someone-else/brain/1'"):
        load_brain(path)
    (tmp_path / "junk.safetensors").write_bytes(b"not a brain")
    with pytest.raises(BrainFileError, match="not a brain file"):
        load_brain(tmp_path / "junk.safetensors")


def test_another_observation_layout_is_refused(tmp_path: Path) -> None:
    config = TrainingConfig()
    path = save(tmp_path / "b.safetensors", config, genome(config))
    loaded = load_mlp(path, FORMAT)
    metadata = dict(loaded.metadata)
    metadata["observation_layout"] = [["target_distance", 1], ["ray_distance", 8]]
    edited = save_mlp(tmp_path / "edited.safetensors", loaded.genome, loaded.layers, "float32", metadata)
    with pytest.raises(BrainFileError, match="another observation layout"):
        load_brain(edited)
    assert json.dumps(metadata["observation_layout"]) in str(pytest.raises(BrainFileError, load_brain, edited).value)


def test_the_default_brains_size_follows_the_layout() -> None:
    """The genome's length comes from the observation's size: per layer, the weights and the biases."""
    (hidden,) = TrainingConfig().brain.hidden
    size = observation_layout().size
    assert brain().dim == size * hidden + hidden + hidden * 2 + 2
    assert brain().layers == ((size, hidden), (hidden, 2))


def test_a_brain_of_the_layout_before_the_box_course_is_refused(tmp_path: Path) -> None:
    """A brain trained without `box_course` (its file says so) reads its inputs at other positions: it is refused."""
    config = TrainingConfig()
    current = load_mlp(save(tmp_path / "now.safetensors", config, genome(config)), FORMAT).metadata
    old_layout = [[name, size] for name, size in layout_of(config.sensing) if name != "box_course"]
    old_size = sum(int(size) for _, size in old_layout)  # type: ignore[call-overload]
    metadata = {**current, "observation_layout": old_layout, "brain": {**current["brain"], "observation_size": old_size}}  # type: ignore[dict-item]
    layers = ((old_size, 32), (32, 2))
    path = save_mlp(tmp_path / "old.safetensors", np.zeros(old_size * 32 + 32 + 32 * 2 + 2), layers, "float32", metadata)
    with pytest.raises(BrainFileError, match="another observation layout"):
        load_brain(path)
