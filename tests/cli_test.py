"""The `scenarios` command: its scenarios and their verbs, clear errors for unknown ones, and nothing left of the commands it
replaced."""

import importlib.util
import tomllib
from pathlib import Path

import pytest

from auxein_scenarios.common import cli

ROOT = Path(__file__).resolve().parents[1]


def test_scenarios_help_lists_box_pushing(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit:
        cli.main(["--help"])
    assert exit.value.code == 0 and "box-pushing" in capsys.readouterr().out


def test_box_pushing_help_lists_train_and_deploy(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit:
        cli.main(["box-pushing", "--help"])
    shown = capsys.readouterr().out
    assert exit.value.code == 0 and "train" in shown and "deploy" in shown
    with pytest.raises(SystemExit):
        cli.main(["box-pushing", "deploy", "--help"])
    shown = capsys.readouterr().out
    for option in ("--scenarios", "--difficulty", "--seed", "--agents", "--backend", "--device", "--precision", "--no-video",
                   "--video-scenarios", "--out"):  # fmt: skip
        assert option in shown, option


@pytest.mark.parametrize(
    ("argv", "message"), [(["sokoban"], "invalid choice: 'sokoban'"), (["box-pushing", "watch"], "invalid choice: 'watch'")]
)
def test_unknown_scenarios_and_verbs_are_clear_errors(argv: list[str], message: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit:
        cli.main(argv)
    error = capsys.readouterr().err
    assert exit.value.code == 2 and message in error and "choose from" in error


def test_train_needs_a_config_or_a_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["box-pushing", "train"]) == 2
    assert "give a CONFIG" in capsys.readouterr().err
    assert cli.main(["box-pushing", "train", str(ROOT / "configs" / "smoke.toml"), "--generations", "5"]) == 2
    assert cli.main(["box-pushing", "train", str(ROOT / "configs" / "nothing.toml")]) == 2


def test_deploy_refuses_what_is_not_a_brain(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"junk")
    assert cli.main(["box-pushing", "deploy", str(junk), "--backend", "numpy", "--out", str(tmp_path / "d")]) == 2
    assert "not a brain file" in capsys.readouterr().err


def test_the_old_commands_are_gone() -> None:
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts == {"scenarios": "auxein_scenarios.common.cli:main"}
    for module in ("demos", "compare", "showcase", "report"):
        assert importlib.util.find_spec(f"auxein_scenarios.box_pushing.{module}") is None, module
    assert not (ROOT / "benchmarks" / "physics_timing.py").exists()
