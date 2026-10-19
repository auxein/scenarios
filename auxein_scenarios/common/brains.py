"""Brains as `.safetensors` files: a feed-forward network's layers as named tensors, and what it is as metadata.

A trained brain is a genome of Auxein's `MLPPolicy`: per layer, from the input side, the weights `(out, in)` row by row, then the
biases. A file stores it **as tensors a reader understands without Auxein**: `layers.<i>.weight` `(out, in)` and
`layers.<i>.bias` `(out,)`, in the run's precision, so any tool that reads safetensors sees a plain network. Loading lays them
out again in the genome's order, **bit for bit**: float32 genomes are float32 to begin with (a float32 run's genomes are), so
widening them to float64 for the genome loses nothing.

**Metadata** is safetensors' string-to-string header: each value here is JSON, so a scenario can store structured facts (the
world it was trained in, its provenance). Every file carries a **format tag** naming the scenario and the version of its
layout; `load_mlp` refuses a file whose tag is not the one the caller expects, before reading anything else, so a brain is
never run in a world it was not made for.
"""

import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

__all__ = ["FORMAT_KEY", "BrainFileError", "MLPFile", "load_mlp", "save_mlp"]

FORMAT_KEY = "format"
"""The metadata key of the format tag."""

Genome = npt.NDArray[np.float64]

# safetensors' numpy API, with the signatures it documents: imported by name, as its stubs leave the dict types unknown
_safetensors: Any = importlib.import_module("safetensors.numpy")
load_file = cast("Callable[[str], dict[str, npt.NDArray[Any]]]", _safetensors.load_file)
save_file = cast("Callable[..., None]", _safetensors.save_file)


class BrainFileError(ValueError):
    """A file that is not a brain this code can run: another format, another observation layout, or not a brain at all."""


@dataclass(frozen=True)
class MLPFile:
    """A loaded brain: its genome (float64, the genome's order), its layers' `(in, out)` sizes, its precision, its metadata."""

    genome: Genome
    layers: tuple[tuple[int, int], ...]
    precision: str
    metadata: dict[str, object]


def _tensors(genome: Genome, layers: Sequence[tuple[int, int]], dtype: type[np.float32] | type[np.float64]) -> dict[str, npt.NDArray[Any]]:
    tensors: dict[str, npt.NDArray[Any]] = {}
    start = 0
    for i, (size, out) in enumerate(layers):
        tensors[f"layers.{i}.weight"] = genome[start : start + out * size].reshape(out, size).astype(dtype)
        start += out * size
        tensors[f"layers.{i}.bias"] = genome[start : start + out].astype(dtype)
        start += out
    if start != genome.shape[0]:
        raise ValueError(f"the layers {list(layers)} hold {start} values, the genome {genome.shape[0]}")
    return tensors


def _header(path: Path) -> dict[str, str]:
    """The file's metadata: a safetensors file starts with the length of its JSON header (8 bytes, little-endian), and the
    header's `__metadata__` holds the string-to-string metadata."""
    with open(path, "rb") as file:
        size = int.from_bytes(file.read(8), "little")
        if not 0 < size < 100_000_000:
            raise ValueError("no safetensors header")
        header = json.loads(file.read(size))
    if not isinstance(header, dict):
        raise ValueError("a safetensors header is a JSON object")
    metadata = cast("dict[str, object]", header).get("__metadata__", {})
    return {str(k): str(v) for k, v in cast("dict[str, object]", metadata).items()}


def save_mlp(path: str | Path, genome: object, layers: Sequence[tuple[int, int]], precision: str, metadata: Mapping[str, object]) -> Path:
    """Writes `genome` (an `MLPPolicy` genome with these `(in, out)` layers) to `path` in `precision` ("float32" or "float64"),
    with `metadata` (each value written as JSON; it must include `FORMAT_KEY`). Refuses a genome the precision cannot hold
    exactly. The file is written to a temporary name and renamed, so a reader never sees half a brain."""
    values = np.asarray(genome, dtype=np.float64).reshape(-1)
    dtype = {"float32": np.float32, "float64": np.float64}[precision]
    if not bool((values.astype(dtype).astype(np.float64) == values).all()):
        raise ValueError(f"the genome is not exactly representable in {precision}: save it in float64")
    if FORMAT_KEY not in metadata:
        raise ValueError(f"a brain's metadata needs a {FORMAT_KEY!r} tag")
    header = {key: json.dumps(value, sort_keys=True) for key, value in metadata.items()}
    header["precision"] = json.dumps(precision)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    save_file(_tensors(values, layers, dtype), str(temporary), metadata=header)
    temporary.replace(target)
    return target


def load_mlp(path: str | Path, format_tag: str) -> MLPFile:
    """Reads a brain written by `save_mlp`, refusing one whose format tag is not `format_tag`."""
    source = Path(path)
    try:
        header = _header(source)
        arrays = load_file(str(source))
    except (OSError, ValueError, RuntimeError) as error:  # a missing or malformed file
        raise BrainFileError(f"{source} is not a brain file: {error}") from None
    names = list(arrays)
    found = json.loads(header.get(FORMAT_KEY, "null"))
    if found != format_tag:
        raise BrainFileError(f"{source} is a brain of format {found!r}; this code runs {format_tag!r} brains")
    metadata: dict[str, object] = {key: json.loads(value) for key, value in header.items()}
    count = sum(1 for name in names if name.endswith(".weight"))
    layers: list[tuple[int, int]] = []
    parts: list[npt.NDArray[np.float64]] = []
    for i in range(count):
        weight, bias = arrays[f"layers.{i}.weight"], arrays[f"layers.{i}.bias"]
        out, size = (int(d) for d in weight.shape)
        layers.append((size, out))
        parts += [weight.astype(np.float64).reshape(-1), bias.astype(np.float64).reshape(-1)]
    precision = str(metadata.pop("precision", "float64"))
    genome: Genome = np.concatenate(parts) if parts else np.zeros(0)  # pyright: ignore[reportUnknownMemberType]
    return MLPFile(genome, tuple(layers), precision, metadata)
