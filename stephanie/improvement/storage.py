"""Filesystem boundaries for isolated Mission improvement operations."""
from __future__ import annotations

import os
from pathlib import Path


def memory_root(experiment_root: Path) -> Path:
    """Use a shared isolated memory store only for Mission's per-run roots.

    The configured memory directory must be a sibling of ``investigations``
    under the same SC4 run root. Standalone CLI runs remain root-local.
    """
    configured = os.environ.get("MISSION_CONTROL_STEPHANIE_MEMORY_ROOT")
    if not configured:
        return experiment_root
    candidate = Path(configured).resolve()
    expected_parent = experiment_root.resolve().parent.parent
    if candidate.parent != expected_parent or experiment_root.resolve().parent.name != "investigations":
        raise ValueError("Mission Stephanie memory root is outside the configured isolated run root")
    return candidate


def memory_db_path(experiment_root: Path, *, local_name: str) -> Path:
    """Return the isolated shared DB for Mission runs or a local DB otherwise."""
    if os.environ.get("MISSION_CONTROL_STEPHANIE_MEMORY_ROOT"):
        return memory_root(experiment_root) / "stephanie-sc3-isolated.sqlite3"
    return experiment_root / local_name
