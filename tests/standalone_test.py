"""The code, tests, benchmarks and configs stand on their own: no pointer to the design document, to the instructions for
working on the repository, to the history of how the code was built, or to the decisions behind it. A docstring states its
reason in place; the design document is where the history and the decisions live."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKED = ("auxein_scenarios", "tests", "benchmarks", "configs", ".github", "pyproject.toml")
FORBIDDEN = re.compile(r"§|\bM[0-9]|(?i:design doc|CLAUDE|milestone|decision [0-9]|docs/design|\bas built\b|\badded at\b)")
"""Section signs, milestone names (M0, M1a, ... as words: not `arm64`), and in any case the design document and its path, the
repository's instructions file, the word milestone, numbered decisions, and history phrases."""


def files() -> list[Path]:
    out: list[Path] = []
    for name in CHECKED:
        path = ROOT / name
        candidates = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
        out += [p for p in candidates if p.suffix in (".py", ".toml", ".yml", ".yaml", ".md", ".json") and p != Path(__file__)]
    return out


def test_nothing_points_outside_the_code() -> None:
    found = [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for path in files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if FORBIDDEN.search(line)
    ]
    assert not found, "pointers to the design doc, the instructions or the history:\n" + "\n".join(found)
