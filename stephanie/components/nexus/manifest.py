# stephanie/components/nexus/manifest.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

@dataclass
class ManifestItem:
    item_id: str
    scorable_id: str
    scorable_type: str
    turn_index: Optional[int] = None
    chat_id: Optional[str] = None

    # annotations from prior passes
    domains: List[str] = field(default_factory=list)
    ner: List[str] = field(default_factory=list)
    near_identity: Dict[str, Any] = field(default_factory=dict)

    # metrics/timeline
    metrics_columns: List[str] = field(default_factory=list)
    metrics_values: List[float] = field(default_factory=list)
    metrics_vector: Dict[str, float] = field(default_factory=dict)

    # embeddings (namespaced for future multi-embed)
    embeddings: Dict[str, List[float]] = field(default_factory=dict)

    # VPM artifacts
    vpm_png: Optional[str] = None          # composite image path
    vpm_channels: List[str] = field(default_factory=list)
    rollout: Dict[str, Any] = field(default_factory=dict)  # steps, delta, gif, frames


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)  

@dataclass
class NexusRunManifest:
    run_id: str
    created_utc: float
    n_items: int = 0
    items: List[ManifestItem] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    def append(self, item: ManifestItem) -> None:
        self.items.append(item)
        self.n_items = len(self.items)

    def save(self, run_dir: Path) -> str:
        run_dir.mkdir(parents=True, exist_ok=True)
        out = run_dir / "manifest.json"
        data = asdict(self)
        # keep floats compact but readable
        import json
        with out.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return str(out)
