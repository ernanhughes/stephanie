from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

# Minimal defaults that match your SSP usage
DEFAULTS = {
    "self_play": {
        "enabled": True,
        "batch_size": 8,
        "max_episode_length": 10,
        "temperature": 0.7,
        "verification_threshold": 0.85,
        # tree
        "M": 2,
        "N": 2,
        "L": 1,
        "scorer_name": "sicql",
        "dimensions": ["alignment"],
        "use_zscore_intra": False,
        "use_zscore_inter": True,
        "value_alpha": 0.0,
        "prefer_non_buggy": True,
        "db": {
            "name": "co",
            "user": "co",
            "password": "co",
            "host": "localhost",
            "port": 5432
        },
        "qmax": {
            "initial_difficulty": 0.3,
            "difficulty_step": 0.05,
            "max_difficulty": 0.95,
            "competence_window": 100,
        },
        "curriculum": {
            "min_success_rate": 0.6,
            "max_failure_rate": 0.3,
            "difficulty_adjustment": 0.1,
        },
        "snapshots": {"keep": 5, "update_interval": 50},
        "jitter": {
            "tick_interval": 2.0,  # seconds
            "sensory_channels": ["vpm", "scm", "epistemic"],
            "metabolic_rate": 0.8,
        },
        "proposer": {
            "max_query_length": 512,
            "verification_depth": 3,
            "novelty_threshold": 0.7,
        },
        "solver": {
            "search_depth": 5,
            "max_retries": 3,
            "reasoning_budget": 2048,
        },
        "verifier": {
            "hrms": [
                {"name": "coherence", "weight": 0.3},
                {"name": "novelty", "weight": 0.2},
                {"name": "causality", "weight": 0.25},
                {"name": "consistency", "weight": 0.25},
            ],
            "min_evidence_count": 2,
        },
    }
}


def ensure_cfg(cfg_like=None) -> DictConfig:
    """
    - Convert dict → DictConfig
    - Accept `ssp:` as alias for `self_play:`
    - Merge with DEFAULTS so required keys exist
    """
    if isinstance(cfg_like, DictConfig):
        cfg = cfg_like
    else:
        cfg = OmegaConf.create(cfg_like or {})

    # alias support: if user provided `ssp:` but not `self_play:`
    if "self_play" not in cfg:
        if "ssp" in cfg:
            cfg.self_play = cfg["ssp"]  # copy alias
        else:
            cfg.self_play = {}  # will be filled by defaults

    # merge with defaults (user values override defaults)
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULTS), cfg)
    return cfg
