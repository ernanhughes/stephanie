# stephanie/components/jitter/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class JasConfig:
    # identity
    name: str = "jitter_orchestrator"
    description: str = "Jitter Autopoietic System Orchestrator"

    # timing / lifecycle
    tick_interval: float = 1.0
    graceful_shutdown_timeout: float = 30.0
    enable_reproduction: bool = True
    reproduction_interval: int = 1000
    reproduction_energy_threshold: float = 80.0
    reproduction_health_threshold: float = 0.7

    # legacy / persistence
    legacy_min_value: float = 0.7
    legacy_max_count: int = 50
    base_dir: Path = Path("data/jitter")

    # telemetry
    telemetry: Dict[str, Any] = field(default_factory=lambda: {
        "subject": "arena.jitter.telemetry",
        "interval": 1.0
    })

    # zero-model + vpm
    zero_model: Dict[str, Any] = field(default_factory=lambda: {
        "fps": 8,
        "output_dir": "data/vpms",
        "pipeline": [
            {"stage": "normalize", "params": {}},
            {"stage": "feature_engineering", "params": {}},
            {"stage": "organization", "params": {"strategy": "spatial"}},
        ],
    })
    timeline_scale_mode: str = "robust01"

    # components config (forwarded as-is)
    core: Dict[str, Any] = field(default_factory=dict)
    triune: Dict[str, Any] = field(default_factory=dict)
    homeostasis: Dict[str, Any] = field(default_factory=dict)
    apoptosis: Dict[str, Any] = field(default_factory=dict)

    # plugin config (shared factory format)
    plugins: Dict[str, Any] = field(default_factory=dict)
