# stephanie/agents/maintenance/risk_trainer.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional

from stephanie.agents.base_agent import BaseAgent

# Trainer (XGBoost + Isotonic) we created earlier
from stephanie.scoring.training.risk_trainer import RiskTrainer, RiskTrainerConfig


def _maybe_build_dataset(
    run_dir: Optional[str],
    out_parquet: str,
    energy_thresh: float,
    logger,
) -> None:
    """
    If a GAP run directory is provided, (re)build the risk dataset parquet.
    Falls back gracefully if import path differs (e.g., local scripts layout).
    """
    if not run_dir:
        return

    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Preferred: call the builder function directly (no subprocess)
        from scripts.build_risk_dataset import build_from_run
        logger.log("RiskDatasetBuildStart", {
            "run_dir": str(run_dir),
            "out": str(out_parquet),
            "energy_thresh": energy_thresh,
        })
        build_from_run(Path(run_dir), out_path, label_energy_thresh=energy_thresh)
        logger.log("RiskDatasetBuildDone", {"rows_path": str(out_parquet)})
    except Exception as e:
        # Fallback no-op: leave existing dataset as-is; agent won’t crash
        logger.log("RiskDatasetBuildFailed", {"error": str(e), "run_dir": str(run_dir)})


class RiskTrainerAgent(BaseAgent):
    """
    Maintenance agent that trains the Risk Predictor (hallucination risk)
    from a parquet dataset (optionally built from a GAP run directory).

    Config keys (see YAML below):
      - data_path: path to risk_dataset.parquet
      - out_dir: where to write models/risk/bundle.joblib
      - run_dir: (optional) GAP run to build the dataset from
      - energy_thresh: (optional) label threshold for max_energy → y
      - trainer: (dict) overrides for RiskTrainerConfig (hp & switches)
      - output_key: context key to place results into (default: "risk_training")
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Agent I/O config
        self.output_key: str = cfg.get("output_key", "risk_training")

        # Dataset/build inputs
        self.data_path: str = cfg.get("data_path", "reports/risk_dataset.parquet")
        self.run_dir: Optional[str] = cfg.get("run_dir")  # optional
        self.energy_thresh: float = float(cfg.get("energy_thresh", 0.55))

        # Trainer config block (falls back to sensible defaults)
        trainer_cfg = cfg.get("trainer", {}) or {}
        # If top-level data_path/out_dir provided, mirror them into trainer cfg
        trainer_cfg.setdefault("data_path", self.data_path)
        trainer_cfg.setdefault("out_dir", cfg.get("out_dir", "models/risk"))

        self.trainer_cfg = trainer_cfg
        self.trainer = RiskTrainer(self.trainer_cfg, memory, container, logger)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Optionally (re)build dataset from a GAP run
        _maybe_build_dataset(
            run_dir=self.run_dir,
            out_parquet=self.data_path,
            energy_thresh=self.energy_thresh,
            logger=self.logger,
        )

        # 2) Train the model (XGB + Isotonic + per-domain gates)
        self.logger.log("RiskTrainerInvoked", {
            "data_path": self.data_path,
            "out_dir": self.trainer_cfg.get("out_dir", "models/risk"),
        })
        meta = self.trainer.train(samples=None, dimension="risk")

        # 3) Return stats in context (mirrors your other agents)
        context[self.output_key] = {
            "training_stats": meta,
            "bundle_path": str(Path(self.trainer_cfg.get("out_dir", "models/risk")) / "bundle.joblib"),
            "metrics_path": str(Path(self.trainer_cfg.get("out_dir", "models/risk")) / "metrics.json"),
            "data_path": self.data_path,
        }
        self.logger.log("RiskTrainerCompleted", context[self.output_key])
        return context
