"""
CLI to run the JAS lifecycle agent from YAML config.
"""
from __future__ import annotations
import argparse, asyncio, signal, sys, yaml, logging
from pathlib import Path
import torch

from stephanie.services.bus.nats_client import get_js
from stephanie.components.jitter.jas_lifecycle_agent import JitterLifecycleAgent
from stephanie.components.jitter.adapters import EBTAdapter, VPMAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("stephanie.jas.cli")

class _Models: pass
class _Managers: pass

async def _amain(cfg_path: str):
    # load config
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    # init bus (env vars may override)
    _ = await get_js(cfg.get("bus"))

    # wire models/managers (replace with your DI container if you have one)
    models, managers = _Models(), _Managers()
    models.ebt = EBTAdapter(dim=1024)
    managers.vpm = VPMAdapter()

    agent = JitterLifecycleAgent(
        {
            "energy": {"initial_cognitive": 50.0, "initial_metabolic": 50.0, "initial_reserve": 20.0},
            "membrane": {"initial_thickness": 0.8},
            "homeostasis": cfg.get("homeostasis", {}),
            "tick_interval": cfg["jitter"]["tick_interval"],
        },
        models,
        managers,
    )

    # start
    await agent.start()
    log.info("JAS lifecycle started (tick %.2fs)", cfg["jitter"]["tick_interval"])

    # wait on signal
    stop = asyncio.Future()
    for s in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(s, lambda: stop.cancel())
    try:
        await stop
    except asyncio.CancelledError:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="conf/jitter/jas.local.yaml")
    args = ap.parse_args()
    try:
        asyncio.run(_amain(args.config))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
