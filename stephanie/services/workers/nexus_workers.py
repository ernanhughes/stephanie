# stephanie/services/workers/nexus_workers.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio
import logging
import time

import numpy as np
from PIL import Image

from stephanie.scoring.scorable import Scorable
from stephanie.services.scoring_service import ScoringService

log = logging.getLogger(__name__)


# ===========================
# Nexus VPM Worker (Inline)
# ===========================
@dataclass
class NexusVPMWorkerInline:
    """
    Orchestrates VPM generation + (optional) visual rollout for a Scorable.
    Defers intelligence to ZeroModelService via:
      - zm.vpm_from_scorable()
      - zm.vpm_rollout()
      - zm.score_vpm_image()
      - zm.timeline_*()
    """
    zm: Any                              # ZeroModelService
    logger: Any = None

    def __post_init__(self):
        self.logger = self.logger or log

    def start_run(self, run_id: str, *, metrics: List[str], out_dir: Optional[str] = None):
        odir = out_dir or self.zm._out_dir
        Path(odir).mkdir(parents=True, exist_ok=True)
        self.zm.timeline_open(run_id, metrics=list(metrics), out_dir=odir)

    async def run_item(self,
        run_id: str,
        item: Scorable,
        *,
        out_dir: Optional[str] = None,
        dims_for_score: List[str] = ("clarity","coherence","complexity","alignment","coverage"),
        rollout_steps: int = 0,
        rollout_strategy: str = "none",      # or "zoom_max"
        save_channels: bool = False,
        name_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Produces VPM + (optional) rollout, appends per-step rows to timeline,
        writes filmstrip.gif + per-item metrics.json.
        """
        t0 = time.time()
        out_dir = Path(out_dir or self.zm._out_dir) / f"nexus_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Scorable → VPM (uint8 [3,H,W]) + adapter meta
        vpm_u8, meta = self.zm.vpm_from_scorable(item, img_size=256)

        # 2) Optional: persist static VPM composite
        item_id = (name_hint or f"{getattr(item, 'id', '') or 'item'}").replace("/", "_")
        item_dir = out_dir / item_id
        item_dir.mkdir(parents=True, exist_ok=True)

        # Composite preview
        comp = vpm_u8.mean(axis=0)
        comp_img = (np.clip(comp, 0, 255)).astype(np.uint8)
        Image.fromarray(comp_img).save(item_dir / "vpm_gray.png")

        # Optional per-channel saves
        if save_channels:
            C, H, W = vpm_u8.shape
            for c in range(C):
                Image.fromarray(vpm_u8[c]).save(item_dir / f"vpm_ch{c}.png")

        # 3) Ask ZeroModel for rollout (frames + step summaries)
        frames, summaries = self.zm.vpm_rollout(
            vpm_u8,
            steps=int(rollout_steps),
            dims=list(dims_for_score),
            strategy=rollout_strategy,
        )

        # 4) Append per-step rows to the ZeroModel timeline
        for s in summaries:
            cols = list(dims_for_score)
            vals = [float(s["scores"].get(d, 0.0)) for d in cols]
            self.zm.timeline_append_row(run_id, metrics_columns=cols, metrics_values=vals)

        # 5) Save filmstrip.gif and step JSON
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
            "adapter_meta": meta,
            "dims": list(dims_for_score),
            "rollout_steps": int(rollout_steps),
            "strategy": rollout_strategy,
            "summaries": summaries,
            "gif": (str(gif_path) if gif_path else None),
        }
        (item_dir / "metrics.json").write_text(
            json_dumps(step_json, indent=2),
            encoding="utf-8"
        )

        return {
            "item_dir": str(item_dir),
            "gif": (str(gif_path) if gif_path else None),
            "metrics_json": str((item_dir / "metrics.json")),
            "latency_ms": (time.time() - t0) * 1000.0,
        }

    async def finalize(self, run_id: str, *, out_dir: Optional[str] = None) -> Dict[str, Any]:
        """Finalize the global timeline → timeline GIF + summary PNG + meta JSON."""
        res = await self.zm.timeline_finalize(run_id, out_path=(out_dir or self.zm._out_dir))
        return res


# ==============================
# Nexus Metrics Worker (Inline)
# ==============================
class NexusMetricsWorkerInline:
    """
    Fat scorable→metrics row builder for Nexus:
      - uses configured scorers via ScoringService (like MetricsWorkerInline)
      - augments with cheap text features (length, tokens, punctuation density, etc.)
      - appends a SINGLE row to ZeroModel timeline with all columns.
    """
    def __init__(self,
        scoring: ScoringService,
        scorers: list[str],
        dimensions: list[str],
        *,
        persist: bool = False,
    ):
        self.scoring = scoring
        self.scorers = list(scorers)
        self.dimensions = list(dimensions)
        self.persist = bool(persist)

    def _text_feats(self, text: str) -> Dict[str, float]:
        text = text or ""
        n = len(text)
        words = text.split()
        nw = len(words)
        caps = sum(1 for c in text if c.isupper())
        punc = sum(1 for c in text if c in "!?;:,.()[]{}\"'`")
        lines = text.count("\n") + 1
        avgw = (sum(len(w) for w in words) / max(1, nw)) if nw else 0.0
        return {
            "text.len": float(n),
            "text.words": float(nw),
            "text.avgw": float(avgw),
            "text.caps_ratio": float(caps / max(1, n)),
            "text.punc_ratio": float(punc / max(1, n)),
            "text.lines": float(lines),
        }

    async def score_and_append(
        self,
        zm: Any,                      # ZeroModelService for timeline append
        scorable: Scorable,
        context: Dict[str, Any],
        run_id: str,
    ) -> Dict[str, Any]:
        """
        Builds a dense metrics vector (scorers + text features),
        appends it as one timeline row, and returns columns/values/vector.
        """
        vector: Dict[str, float] = {}
        summary: Dict[str, Any] = {}

        # 1) Scorers (exactly like MetricsWorkerInline)
        for name in self.scorers:
            bundle = (
                self.scoring.score_and_persist(
                    scorer_name=name, scorable=scorable, context=context, dimensions=self.dimensions
                ) if self.persist else
                self.scoring.score(
                    scorer_name=name, scorable=scorable, context=context, dimensions=self.dimensions
                )
            )
            agg = float(bundle.aggregate())
            per = {d: float(sr.score) for d, sr in bundle.results.items()}
            summary[name] = {"aggregate": agg, "per_dimension": per}

            flat = bundle.flatten(include_scores=True, include_attributes=True, numeric_only=True)
            for k, v in flat.items():
                vector[f"{name}.{k}"] = float(v)
            vector[f"{name}.aggregate"] = agg
            await asyncio.sleep(0)  # yield

        # 2) Cheap text features
        tfeats = self._text_feats(getattr(scorable, "text", ""))
        vector.update(tfeats)

        # 3) Deterministic row
        columns = sorted(vector.keys())
        values = [vector[c] for c in columns]

        # 4) Append to ZeroModel timeline
        zm.timeline_append_row(run_id, metrics_columns=columns, metrics_values=values)

        return {"columns": columns, "values": values, "vector": vector, "scores": summary}


def json_dumps(obj, indent=2):
    import json
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False)
    except Exception:
        return json.dumps(obj, indent=indent)
