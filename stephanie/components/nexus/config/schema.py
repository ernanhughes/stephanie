# stephanie/components/nexus/config/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class NexusConfig:
    model: Dict[str, Any] = field(default_factory=lambda: {
        "visual_encoder": "timesformer_s",
        "token_merge": True,
    })
    index: Dict[str, Any] = field(default_factory=lambda: {
        "coarse": {"type": "faiss_ivfpq", "nlist": 4096, "m": 32, "nprobe": 16},
        "rerank": {"late_interaction": True, "topk_coarse": 200, "topk_final": 20},
    })
    graph: Dict[str, Any] = field(default_factory=lambda: {
        "knn_k": 32,
        "keep_top_m_lateint": 12,
    })
    path: Dict[str, Any] = field(default_factory=lambda: {
        "steps_max": 12,
        "weights": {
            "alpha_text": 1.0,
            "beta_goal": 0.7,
            "gamma_stability": 0.6,
            "zeta_agreement": 0.4,
            "delta_domain": 0.0,     # added for pathfinder
            "epsilon_entity": 0.0,   # added for pathfinder
            "eta_novelty": 0.3,
            "kappa_switch": 0.2,
        }
    })
    tables: Dict[str, str] = field(default_factory=lambda: {
        "nodes": "nexus_nodes",
        "edges": "nexus_edges",
        "paths": "nexus_paths",
    })
    bus: Dict[str, Any] = field(default_factory=lambda: {"topic_prefix": "nexus.events"})

def coerce(cfg: dict) -> dict:
    # minimal coerce/validate; could expand to pydantic later
    return cfg or NexusConfig().__dict__
