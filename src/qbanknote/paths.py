"""Project path discovery and notebook import bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root (directory containing ``cross_validation/``)."""
    cwd = (start or Path.cwd()).resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "cross_validation").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate project root containing cross_validation/. "
        f"Started search from {cwd}."
    )


def ensure_importable(start: Path | None = None) -> Path:
    """Add ``src/`` to ``sys.path`` so ``import qbanknote`` works from notebooks."""
    root = find_project_root(start)
    src = root / "src"
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return root
