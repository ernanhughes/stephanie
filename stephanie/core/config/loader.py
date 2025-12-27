# stephanie/config/loader.py
from __future__ import annotations

import hashlib
import json
import os
from typing import List

from omegaconf import OmegaConf

from stephanie.config.schema import AppConfig
from stephanie.utils.hash_utils import hash_text


def load_config(yaml_paths: List[str] | None = None,
                cli_overrides: List[str] | None = None,
                env_prefix: str = "STEPHANIE__") -> AppConfig:
    base = OmegaConf.create(AppConfig().model_dump())
    yaml_cfg = OmegaConf.create()
    for p in (yaml_paths or []):
        yaml_cfg = OmegaConf.merge(yaml_cfg, OmegaConf.load(p))

    # env like STEPHANIE__MCTS__MAX_DEPTH=6
    env_items = []
    for k,v in os.environ.items():
        if not k.startswith(env_prefix): continue
        key = k.removeprefix(env_prefix).lower().replace("__",".")
        env_items.append(f"{key}={v}")
    env_cfg = OmegaConf.from_cli(env_items)

    cli_cfg = OmegaConf.from_cli(cli_overrides or [])
    merged = OmegaConf.merge(base, yaml_cfg, env_cfg, cli_cfg)
    # validate
    return AppConfig(**OmegaConf.to_container(merged, resolve=True))

def snapshot_and_fingerprint(cfg: AppConfig) -> tuple[dict, str]:
    blob = cfg.model_dump()
    s = json.dumps(blob, sort_keys=True, separators=(",",":"))
    fp = hash_text(s)
    return blob, fp
