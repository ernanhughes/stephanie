# zeromodel/gap_gauge.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from zeromodel.pipeline.organizer.top_left import TopLeft

from stephanie.scoring.metrics.visicalc import (VisiCalcEpisode,
                                                episode_features)

# If you’ve moved VisiCalcReport into Stephanie core, you can also
# import the report-level gauge from there. For now, assume a pure
# numpy/episode-based path inside ZeroModel.


@dataclass
class GapGaugeResult:
    """
    Unified comparison between a baseline and a targeted run.

    Combines:
      - VisiCalc-style numeric comparison (episode features)
      - TopLeft-style visual comparison (canonicalized VPMs)
    """
    episode_id: str

    # VisiCalc episode features
    baseline_feats: np.ndarray
    target_feats: np.ndarray
    diff_feats: np.ndarray
    feat_names: List[str]

    # Simple scalar gauges
    baseline_quality: float
    target_quality: float
    diff_quality: float

    # Visual comparison from TopLeft
    topleft_improvement_ratio: float
    topleft_gain: float
    topleft_loss: float

    # Optional debug artifacts
    meta: Dict[str, Any]


def _episode_from_scores(
    episode_id: str,
    scores: np.ndarray,
    metric_names: List[str],
    item_ids: List[str],
) -> Tuple[np.ndarray, List[str], float]:
    """
    Helper: turn a score matrix into:
      - episode feature vector
      - feature names
      - a simple scalar 'quality' (mean frontier-ish)
    """
    episode = VisiCalcEpisode(
        episode_id=episode_id,
        scores=scores,
        metric_names=list(metric_names),
        item_ids=list(item_ids),
        meta={},
    )
    feats, feat_names = episode_features(episode)

    # Very simple scalar: mean over all normalized metrics
    # (You can later replace this with a learned critic.)
    quality = float(feats.mean())

    return feats.astype(np.float32), feat_names, quality


def _compare_vpms_topleft(
    vpm_base: np.ndarray,
    vpm_tgt: np.ndarray,
    *,
    metric_mode: str = "luminance",
    iterations: int = 5,
    push_corner: str = "tl",
    monotone_push: bool = True,
    stretch: bool = True,
) -> Dict[str, Any]:
    """
    Canonicalize baseline & target VPMs with TopLeft, then compute a simple
    'is target visually better than baseline?' metric.
    """
    stage = TopLeft(
        metric_mode=metric_mode,
        iterations=iterations,
        push_corner=push_corner,
        monotone_push=monotone_push,
        stretch=stretch,
    )

    tl_base, meta_base = stage.process(vpm_base)
    tl_tgt, meta_tgt = stage.process(vpm_tgt)

    tl_base = tl_base.astype(np.float32)
    tl_tgt = tl_tgt.astype(np.float32)
    assert tl_base.shape == tl_tgt.shape, "Base/Target shapes must match after TopLeft"

    diff = tl_tgt - tl_base

    gain = float(np.sum(np.clip(diff, 0.0, None)))
    loss = float(np.sum(np.clip(-diff, 0.0, None)))
    total = gain + loss + 1e-8

    improvement_ratio = gain / total

    return {
        "topleft_base": tl_base,
        "topleft_tgt": tl_tgt,
        "topleft_diff": diff,
        "gain": gain,
        "loss": loss,
        "improvement_ratio": improvement_ratio,
        "meta_base": meta_base,
        "meta_tgt": meta_tgt,
    }


def compute_gap_gauge(
    *,
    episode_id: str,
    scores_baseline: np.ndarray,   # shape (N, M)
    scores_target: np.ndarray,     # shape (N, M)
    metric_names: List[str],
    item_ids: List[str],
    vpm_baseline: Optional[np.ndarray] = None,  # for TopLeft; can be None
    vpm_target: Optional[np.ndarray] = None,
    topleft_cfg: Optional[Dict[str, Any]] = None,
) -> GapGaugeResult:
    """
    Core entry point: given numeric scores + optional VPMs, produce a GapGaugeResult.
    """
    scores_baseline = np.asarray(scores_baseline, dtype=np.float32)
    scores_target = np.asarray(scores_target, dtype=np.float32)
    assert scores_baseline.shape == scores_target.shape, "Baseline/Target scores must match shape"

    # 1) Numeric side: VisiCalc episodes
    base_feats, feat_names, base_q = _episode_from_scores(
        f"{episode_id}:baseline",
        scores_baseline,
        metric_names,
        item_ids,
    )
    tgt_feats, feat_names2, tgt_q = _episode_from_scores(
        f"{episode_id}:target",
        scores_target,
        metric_names,
        item_ids,
    )
    assert feat_names == feat_names2, "Feature name mismatch between baseline and target"

    diff_feats = tgt_feats - base_feats
    diff_q = float(diff_feats.mean())

    # 2) Visual side: TopLeft canonicalization (optional)
    topleft_improvement_ratio = 0.5
    topleft_gain = 0.0
    topleft_loss = 0.0
    tl_meta: Dict[str, Any] = {}

    if vpm_baseline is not None and vpm_target is not None:
        cfg = dict(
            metric_mode="luminance",
            iterations=5,
            push_corner="tl",
            monotone_push=True,
            stretch=True,
        )
        if topleft_cfg:
            cfg.update(topleft_cfg)

        tl = _compare_vpms_topleft(
            vpm_base=vpm_baseline,
            vpm_tgt=vpm_target,
            **cfg,
        )
        topleft_improvement_ratio = tl["improvement_ratio"]
        topleft_gain = tl["gain"]
        topleft_loss = tl["loss"]
        tl_meta = {
            "meta_base": tl["meta_base"],
            "meta_tgt": tl["meta_tgt"],
            "shape": tl["topleft_base"].shape,
        }

    meta: Dict[str, Any] = {
        "episode_id": episode_id,
        "metric_names": list(metric_names),
        "num_items": len(item_ids),
        "topleft": tl_meta,
    }

    return GapGaugeResult(
        episode_id=episode_id,
        baseline_feats=base_feats,
        target_feats=tgt_feats,
        diff_feats=diff_feats,
        feat_names=feat_names,
        baseline_quality=base_q,
        target_quality=tgt_q,
        diff_quality=diff_q,
        topleft_improvement_ratio=topleft_improvement_ratio,
        topleft_gain=topleft_gain,
        topleft_loss=topleft_loss,
        meta=meta,
    )
