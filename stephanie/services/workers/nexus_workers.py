# stephanie/services/workers/nexus_workers.py
from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from stephanie.scoring.metrics.scorable_processor import ScorableProcessor
from stephanie.scoring.scorable import Scorable
from stephanie.services.zeromodel_service import ZeroModelService
from stephanie.utils.json_sanitize import dumps_safe
from stephanie.utils.vpm_utils import vpm_image_details

log = logging.getLogger(__name__)


# ===========================
# Nexus VPM Worker (Inline)
# ===========================
@dataclass
class NexusVPMWorkerInline:
    """
    New version: delegates everything to ScorableProcessor.
      - SP computes embeddings, domains, NER, scores, metrics vector
      - SP renders VPM from metrics and (optionally) persists vpm_gray.png
      - This worker only (optionally) does rollout frames + timeline append + artifacts
    """

    sp: ScorableProcessor
    zm: ZeroModelService
    logger: Any = None

    def __post_init__(self):
        self.logger = self.logger or log

    def start_run(
        self, run_id: str, *, metrics: List[str], out_dir: Optional[str] = None
    ):
        odir = out_dir or self.sp.vpm_out_root
        Path(odir).mkdir(parents=True, exist_ok=True)
        self.zm.timeline_open(run_id, metrics=list(metrics), out_dir=str(odir))

    async def run_item(
        self,
        run_id: str,
        item: Scorable,
        row: Dict[str, Any],
        *,
        out_dir: Optional[str] = None,
        dims_for_score: List[str] = (
            "clarity",
            "coherence",
            "complexity",
            "alignment",
            "coverage",
        ),
        rollout_steps: int = 0,
        rollout_strategy: str = "none",  # or "zoom_max"
        save_channels: bool = False,  # kept for compatibility; SP has its own cfg.save_vpm_channels
        name_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        t0 = time.time()
        out_root = Path(out_dir or self.sp.vpm_out_root) / f"nexus_{run_id}"
        out_root.mkdir(parents=True, exist_ok=True)
        # Compose per-item dir; SP may already have saved a gray PNG
        item_id = (
            name_hint or f"{getattr(item, 'id', '') or 'item'}"
        ).replace("/", "_")
        item_dir = out_root / item_id
        item_dir.mkdir(parents=True, exist_ok=True)

        metrics_values = row.get("metrics_values", []) or []
        metrics_columns = row.get("metrics_columns", []) or []
        vpm_png, adapter_meta = await self.zm.vpm_from_scorable(
            item,
            metrics_values=metrics_values,
            metrics_columns=metrics_columns,
        )

        # Recreate a preview from VPM array, if ZeroModel can re-render quickly
        # Prefer deterministic reconstruction from metrics (no feature drift).
        details = vpm_image_details(
            vpm_png,
            out_path=item_dir / "vpm_gray.png",
            save_per_channel=save_channels,
        )
        vpm_png = details["gray_path"]

        # Rollout (frames + per-step summaries) — ZeroModel owns the policy
        frames, summaries = self.zm.vpm_rollout(
            # If SP didn’t stash raw arrays, we can re-render deterministically from metrics:
            None,  # vpm array optional; ZeroModel should accept None + metrics to recompose internally
            steps=int(rollout_steps),
            dims=list(dims_for_score),
            strategy=rollout_strategy,
            metrics_columns=metrics_columns,
            metrics_values=metrics_values,
        )

        self.zm.timeline_append_row(
            run_id,
            metrics_columns=metrics_columns,
            metrics_values=metrics_values,
        )

        # Persist filmstrip + metrics.json
        try:
            from imageio.v2 import mimsave
            gif_path = item_dir / "filmstrip.gif"
            mimsave(gif_path, frames, fps=1, loop=0)
        except Exception as e:
            self.logger.warning("Failed to save GIF for %s: %s", item_id, e)
            gif_path = None

        step_json = {
            "scorable_id": getattr(item, "id", ""),
            "target_type": getattr(item, "target_type", "custom"),
            "vpm_png": vpm_png,
            "dims": list(dims_for_score),
            "rollout_steps": int(rollout_steps),
            "strategy": rollout_strategy,
            "summaries": summaries,
            "gif": (str(gif_path) if gif_path else None),
            "metrics_columns": row.get("metrics_columns") or [],
            "metrics_values": row.get("metrics_values") or [],
        }
        (item_dir / "metrics.json").write_text(
            dumps_safe(step_json, indent=2), encoding="utf-8"
        )

        return {
            "item_dir": str(item_dir),
            "gif": (str(gif_path) if gif_path else None),
            "metrics_json": str((item_dir / "metrics.json")),
            "vpm_png": vpm_png,
            "latency_ms": (time.time() - t0) * 1000.0,
        }

    async def finalize(
        self, run_id: str, *, out_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        return await self.zm.timeline_finalize(
            run_id, out_path=str(out_dir or self.sp.vpm_out_root)
        )
