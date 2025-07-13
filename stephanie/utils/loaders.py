# stephanie/utils/loaders.py

import os

from omegaconf import OmegaConf

from stephanie.logs import JSONLogger
from stephanie.memory import MemoryTool


def get_memory(config_name: str = "db/postgres") -> MemoryTool:
    cfg = get_config(config_name)
    print(f"Loading memory with config: {cfg}")
    """Initialize and return a default Memory instance."""
    return MemoryTool(cfg, JSONLogger(log_path="memory_log.jsonl"))


def get_logger(file_path: str = "log.jsonl") -> object:
    """Return a logger instance with the specified name."""
    return JSONLogger(file_path)


def get_config(config_name: str = "config.yaml"):
    """
    Load a Hydra-style YAML config from the configs/agents directory.
    Example: get_config("lats") -> loads configs/agents/lats.yaml
    """
    path = os.path.join("config", f"{config_name}.yaml")
    return OmegaConf.to_container(
        OmegaConf.load(path), resolve=True, throw_on_missing=True
    )
