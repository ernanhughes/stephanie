# stephanie/components/critic/agents/critic_dataset.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.critic.model.dataset import (
    CORE_FEATURE_COUNT, canonicalize_metric_names, collect_visicalc_samples,
    save_dataset)

log = logging.getLogger(__name__)


class CriticDatasetAgent(BaseAgent):
    """
    Agent wrapper around the Tiny Critic dataset builder (tiny_critic_dataset.py).

    This agent:
      - Traverses the entire VisiCalc runs directory
      - Loads CSV/JSON metric matrices
      - Loads targeted/baseline reports
      - Builds core + dynamic feature vectors
      - Canonicalizes metric names
      - Saves 'critic.npz'
      - Logs dataset stats in context

    This is EXACTLY the same logic as running the CLI dataset builder,
    but used as a pipeline agent inside Stephanie.
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Configurable paths
        self.visicalc_root = Path(cfg.get("visicalc_root", "runs/visicalc"))
        self.output_path = Path(cfg.get("output_path", "data/critic.npz"))
        self.meta_path = Path(cfg.get("meta_path", "data/critic_metadata.json"))

        # Ensure directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log.info("🚀 CriticDatasetBuilderAgent: Starting dataset generation…")
        log.info(f"📂 VisiCalc root: {self.visicalc_root.resolve()}")
        log.info(f"📦 Output dataset: {self.output_path.resolve()}")

        # --------------------------------------------------------------
        # 1. Collect dataset (same as CLI)
        # --------------------------------------------------------------
        X, y, metric_names, groups = collect_visicalc_samples(self.visicalc_root)

        # Canonicalize feature names
        metric_names = canonicalize_metric_names(metric_names)

        # --------------------------------------------------------------
        # 2. Sanity checks (same as CLI)
        # --------------------------------------------------------------
        core_dim = CORE_FEATURE_COUNT
        core_block = X[:, :core_dim]

        if core_block.std(axis=0).min() <= 1e-12:
            log.warning("⚠️ Some core features have near-zero variance. Check VisiCalc runs.")

        # --------------------------------------------------------------
        # 3. Save dataset
        # --------------------------------------------------------------
        save_dataset(X, y, metric_names, groups, self.output_path)

        # --------------------------------------------------------------
        # 4. Write metadata file
        # --------------------------------------------------------------
        meta = {
            "rows": int(X.shape[0]),
            "dim": int(X.shape[1]),
            "targeted": int((y == 1).sum()),
            "baseline": int((y == 0).sum()),
            "metric_names": metric_names,
            "groups": list(groups),
            "output_path": str(self.output_path),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2))

        # Add results to context
        context["critic_dataset"] = meta

        log.info("🎉 CriticDatasetBuilderAgent: Dataset generation complete!")
        return context
