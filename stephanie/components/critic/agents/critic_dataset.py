# stephanie/components/critic/agents/critic_dataset.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.critic.model.dataset import (
    canonicalize_metric_names, collect_frontier_lens_samples, save_dataset)

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
        self.visicalc_root = Path(cfg.get("visicalc_root", "runs/critic"))
        self.output_path = Path(cfg.get("output_path", "data/critic.npz"))
        self.meta_path = Path(cfg.get("meta_path", "data/critic_metadata.json"))

        # Ensure directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log.info("🚀 CriticDatasetBuilderAgent: Starting dataset generation…")
        log.info(f"📂 VisiCalc root: {self.visicalc_root.resolve()}")
        log.info(f"📦 Output dataset: {self.output_path.resolve()}")

        # 1) Collect full dataset
        kept_columns = self.memory.metrics.get_kept_columns(self.run_id)
        kept_columns = canonicalize_metric_names(kept_columns) 
        X, y, metric_names, groups = collect_frontier_lens_samples(kept_columns, self.visicalc_root)

        # 2) Project to DB-locked kept columns (order preserved)
        run_id = context.get("pipeline_run_id") or self.run_id
        if not kept_columns:
            raise RuntimeError(
                f"[CriticDataset] No kept columns for run_id={run_id}. "
                "Run ScorableProcessor with MetricFilterGroupFeature first."
            )

        idxs = _align_kept_columns(metric_names, kept_columns, casefold=True)
        X = X[:, idxs] 
        metric_names = [metric_names[i] for i in idxs]

        log.info(
            "[CriticDataset] projected dataset to DB-locked features: kept=%d of collected=%d",
            len(metric_names), len(kept_columns)
        )

        # 3) Save dataset (uses filtered X + metric_names)
        save_dataset(X, y, metric_names, groups, self.output_path)

        # 4) Write metadata file
        meta = {
            "rows": int(X.shape[0]),
            "dim": int(X.shape[1]),
            "targeted": int((y == 1).sum()),
            "baseline": int((y == 0).sum()),
            "metric_names": metric_names,
            "groups": list(groups),
            "output_path": str(self.output_path),
            "run_id": str(run_id),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2))

        context["critic_dataset"] = meta
        log.info("🎉 CriticDatasetBuilderAgent: Dataset generation complete!")
        return context

def _align_kept_columns(
    metric_names: list[str],
    kept_columns: list[str],
    *,
    casefold: bool = True,
) -> list[int]:
    """
    Return indices into `metric_names` for the kept_columns, in kept order.
    Robust to case and common aliasing via canonicalize_metric_names (already imported).
    """
    # Canonicalize both sides
    canon_metric = canonicalize_metric_names(metric_names)
    canon_kept   = canonicalize_metric_names(kept_columns)

    if casefold:
        canon_metric = [m.casefold() for m in canon_metric]
        canon_kept   = [k.casefold() for k in canon_kept]

    # Map canonical metric name -> first index in metric_names
    first_idx: dict[str, int] = {}
    for i, nm in enumerate(canon_metric):
        # only the first occurrence matters for projection
        first_idx.setdefault(nm, i)

    idxs: list[int] = []
    missing: list[str] = []
    for k in canon_kept:
        j = first_idx.get(k)
        if j is None:
            missing.append(k)
        else:
            idxs.append(j)

    if missing:
        log.warning(
            "[CriticDataset] %d kept columns were not found in collected metric_names (showing first 10): %s",
            len(missing), missing[:10]
        )

    if not idxs:
        raise RuntimeError("[CriticDataset] after alignment, no kept columns matched the dataset metrics")

    return idxs
