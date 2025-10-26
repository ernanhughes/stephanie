# stephanie/components/ssp/cli.py
from __future__ import annotations

import os
import click
from omegaconf import OmegaConf, DictConfig
from stephanie.components.ssp.substrate import SspComponent
from stephanie.utils.json_sanitize import dumps_safe

def _load_cfg(path: str) -> DictConfig:
    if path and os.path.exists(path):
        return OmegaConf.load(path)
    here = os.path.dirname(__file__)
    return OmegaConf.load(os.path.join(here, "config.yaml"))

@click.group()
def ssp():
    """SSP component CLI"""

@ssp.command()
@click.option("--config", default=None, help="Path to config.yaml")
@click.option("--steps", default=0, type=int, help="0 = continuous; >0 = run N steps")
def start(config, steps):
    """
    Foreground run.
      steps == 0  → run continuously until Ctrl+C
      steps > 0   → run exactly N steps and exit
    """
    cfg = _load_cfg(config)
    comp = SspComponent(cfg)
    try:
        comp.start(max_steps=None if steps == 0 else steps, background=False)  # ← foreground
        # If steps>0, we return here after finishing; if continuous, this call blocks until Ctrl+C
    finally:
        # ensure clean shutdown if we ever break/interrupt
        comp.stop()
    # print final status
    click.echo(dumps_safe(comp.status(), indent=2))

@ssp.command()
@click.option("--config", default=None, help="Path to config.yaml")
def tick(config):
    """
    Execute one training step in foreground, then emit one substrate tick and exit.
    """
    cfg = _load_cfg(config)
    comp = SspComponent(cfg)
    try:
        comp.start(max_steps=1, background=False)  # run exactly one step
        out = comp.tick()                          # safe to tick in-process
        click.echo(dumps_safe(out, indent=2))
    finally:
        comp.stop()

@ssp.command()
@click.option("--config", default=None, help="Path to config.yaml")
def status(config):
    cfg = _load_cfg(config)
    comp = SspComponent(cfg)  # no background thread started
    click.echo(dumps_safe(comp.status(), indent=2))


if __name__ == "__main__":
    ssp()
