# stephanie/components/nexus/app/manifest.py

from dataclasses import asdict, dataclass, field
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
    vpm_png: Optional[str] = None
    vpm_channels: List[str] = field(default_factory=list)

    # Rollout pointers (NEW — file paths, counts, simple progress snapshot)
    rollout: Dict[str, Any] = field(default_factory=dict)  # {
        # "events_jsonl": "…/blossom_<id>.jsonl",
        # "frames_json": "…/frames.json",    # optional
        # "gif": "…/filmstrip.gif",          # optional
        # "frames": 0,                       # optional
        # "progress": {"made":0,"accepted":0,"rejected":0,"total":0}
    # }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class NexusRunManifest:
    run_id: str
    created_utc: float
    n_items: int = 0
    items: List[ManifestItem] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)  # can include {"version":"1.1","blossom_id":...}

    def append(self, item: ManifestItem) -> None:
        self.items.append(item)
        self.n_items = len(self.items)

    def attach_rollout(self, item_id: str, rollout: Dict[str, Any]) -> None:  # NEW helper
        for it in self.items:
            if it.item_id == item_id:
                it.rollout.update(rollout)
                return

    def save(self, run_dir: Path) -> str:
        run_dir.mkdir(parents=True, exist_ok=True)
        out = run_dir / "manifest.json"
        import json
        with out.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        return str(out)
