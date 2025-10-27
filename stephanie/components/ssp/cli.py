# stephanie/components/ssp/cli.py
from __future__ import annotations

import click
import yaml
from typing import Optional

from stephanie.components.ssp.substrate import SspComponent
from stephanie.components.ssp.config import ensure_cfg
from stephanie.logging.json_logger import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from stephanie.services.registry_loader import load_services_profile
from stephanie.services.service_container import ServiceContainer
from stephanie.utils.json_sanitize import dumps_safe
import asyncio
from stephanie.components.ssp.tree_bridge import SspTreeBridge



def _load_cfg(path: Optional[str]):
    """Load YAML (if provided) and normalize with ensure_cfg()."""
    if not path:
        return ensure_cfg({})
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ensure_cfg(raw)


def _print_json(obj) -> None:
    """Always emit JSON-safe output (handles numpy, datetimes, etc.)."""
    click.echo(dumps_safe(obj, indent=2))


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def ssp():
    """Search Self-Play (SSP) component CLI.

    Examples:
      python -m stephanie.components.ssp.cli start -c ssp.local.yaml --steps 5
      python -m stephanie.components.ssp.cli tick -c ssp.local.yaml
      python -m stephanie.components.ssp.cli status -c ssp.local.yaml
    """


@ssp.command()
@click.option("--config", "-c", type=click.Path(exists=True), default=None, help="YAML config")
@click.option("--steps", type=int, default=None, help="Max training steps (blocks; None = continuous)")
def start(config, steps):
    """Run the self-play loop. If --steps is set, block until done (clean shutdown)."""
    cfg = _load_cfg(config)
    comp = SspComponent(cfg)
    # Block to avoid background thread shutdown races
    comp.start(max_steps=steps, background=(steps is None))
    if steps is not None:
        comp.stop()
    _print_json({"ok": True, "event": "started", "steps": steps})


@ssp.command()
@click.option("--config", "-c", type=click.Path(exists=True), default=None, help="YAML config")
def tick(config):
    """Perform a single jitter substrate tick and print the payload."""
    cfg = _load_cfg(config)
    comp = SspComponent(cfg)
    out = comp.tick()
    _print_json(out)


@ssp.command()
@click.option("--config", "-c", type=click.Path(exists=True), default=None, help="YAML config")
def status(config):
    """Print current SSP status as JSON."""
    cfg = _load_cfg(config)
    comp = SspComponent(cfg)
    _print_json(comp.status())


@ssp.command("train-step")
@click.option("--config", "-c", type=click.Path(exists=True), default=None, help="YAML config")
def train_step_cmd(config):
    """Execute a single training step and print metrics (debug/ops)."""
    cfg = _load_cfg(config)
    comp = SspComponent(cfg)
    # Be defensive if substrate exposes trainer differently
    trainer = getattr(comp, "trainer", None)
    if trainer and hasattr(trainer, "train_step"):
        metrics = trainer.train_step()
        _print_json({"ok": True, "metrics": metrics})
    else:
        _print_json({"ok": False, "error": "trainer.train_step not available"})

@ssp.command("tree-grpo")
@click.option("--config", "-c", type=click.Path(exists=True), default=None)
@click.option("--goal", "-g", type=str, required=True, help="Goal/Task text")
@click.option("--value", type=float, default=0.0, help="Optional [0,1] value weight")
def tree_grpo_cmd(config, goal, value):
    """Run a single Tree-GRPO rollout and print a compact report."""
    cfg = _load_cfg(config)
    logger = JSONLogger("logs/sis.jsonl")
    memory = MemoryTool(cfg=cfg.self_play, logger=logger)
    container = ServiceContainer(cfg=cfg.self_play, logger=logger)
    load_services_profile(
        container,
        cfg=cfg,
        memory=memory,
        logger=logger,
        profile_path="./config/services/all.yaml",
    )
    bridge = SspTreeBridge(cfg, memory, container=container, logger=logger)
    out = asyncio.run(bridge.rollout(goal_text=goal, value=value))
    click.echo(dumps_safe(out["report"], indent=2))


if __name__ == "__main__":
    ssp()
