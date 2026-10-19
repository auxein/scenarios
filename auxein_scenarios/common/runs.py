"""Output directories: a run's (`runs/<name>-<timestamp>`) and a deployment's (`deployments/<name>-<timestamp>`), both ignored by
git. The timestamp is local time to the second, so directories sort by when they were made and a person can tell them apart."""

from datetime import datetime
from pathlib import Path

__all__ = ["timestamped"]


def timestamped(root: Path, name: str) -> Path:
    """`root/<name>-<YYYYmmdd-HHMMSS>`, not created."""
    return root / f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
