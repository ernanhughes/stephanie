# stephanie/components/critic/agents/self_improver.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from stephanie.agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


# ---------- Lightweight model registry (file-based) ----------

@dataclass
class ModelRecord:
    model_path: str
    meta_path: str
    auroc: Optional[float] = None
    ece: Optional[float] = None
    pr_auc: Optional[float] = None
    accuracy: Optional[float] = None


class ModelRegistry:
    """
    Very small, local registry backed by a json file.
    Tracks the 'current' (promoted) critic model and its eval metrics.
    """
    def __init__(self, path: str | Path = "models/critic_registry.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_current(self) -> Optional[ModelRecord]:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text())
            cur = data.get("current")
            if not cur:
                return None
            return ModelRecord(**cur)
        except Exception as e:
            log.warning("[ModelRegistry] failed to load: %s", e)
            return None

    def promote(self, rec: ModelRecord) -> None:
        payload = {"current": rec.__dict__}
        self.path.write_text(json.dumps(payload, indent=2))
        log.info("🏆 ModelRegistry: promoted model=%s (AUROC=%.4f, ECE=%s, PR-AUC=%s, ACC=%s)",
                 rec.model_path, rec.auroc or -1, rec.ece, rec.pr_auc, rec.accuracy)


# ---------- Promotion policy (simple & explicit) ----------

@dataclass
class PromotionPolicy:
    """
    Gate new model vs. current. Keep this dumb & auditable.
    """
    min_auroc: float = 0.70           # absolute floor for any promotion
    max_ece: float = 0.20             # absolute ceiling
    require_improvement: bool = True  # must beat current by margin?
    auroc_margin: float = 0.01        # >1pt AUROC better than current
    ece_margin: float = -0.02         # <=2pt better (negative means lower is better)

    def ok(self, new: ModelRecord, cur: Optional[ModelRecord]) -> tuple[bool, str]:
        if new.auroc is None:
            return False, "new model missing AUROC"
        if new.auroc < self.min_auroc:
            return False, f"AUROC {new.auroc:.4f} below min {self.min_auroc:.4f}"

        if new.ece is not None and new.ece > self.max_ece:
            return False, f"ECE {new.ece:.4f} above max {self.max_ece:.4f}"

        if not self.require_improvement or cur is None or cur.auroc is None:
            return True, "no current or improvement not required"

        # Compare AUROC
        if (new.auroc - cur.auroc) < self.auroc_margin:
            return False, f"AUROC gain {new.auroc - cur.auroc:.4f} < margin {self.auroc_margin:.4f}"

        # Compare ECE if both exist (lower is better). Allow missing ECE gracefully.
        if new.ece is not None and cur.ece is not None:
            if (new.ece - cur.ece) > self.ece_margin:
                return False, f"ECE delta {new.ece - cur.ece:+.4f} > margin {self.ece_margin:+.4f}"

        return True, "meets improvement margins"


# ---------- The Agent ----------

class CriticSelfImproverAgent(BaseAgent):
    """
    End-of-pipeline guard that:
      1) Reads the freshly trained critic model & meta
      2) Reads the currently promoted model (if any)
      3) Compares via PromotionPolicy
      4) Promotes by writing to a small registry if better
    Assumes your CriticTrainerAgent wrote 'models/critic.joblib' and 'models/critic.meta.json'.
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.model_path = Path(cfg.get("model_path", "models/critic.joblib"))
        self.meta_path  = Path(cfg.get("meta_path", "models/critic.meta.json"))
        self.registry_path = Path(cfg.get("registry_path", "models/critic_registry.json"))
        self.policy_cfg = cfg.get("policy", {}) or {}
        self.policy = PromotionPolicy(**self.policy_cfg)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # 0) sanity: the trainer should have just written these files
        if not self.model_path.exists() or not self.meta_path.exists():
            log.warning("🤷 SelfImprover: missing model/meta; skip promotion. model=%s meta=%s",
                        self.model_path, self.meta_path)
            return context

        # 1) parse meta to get eval metrics
        try:
            meta = json.loads(self.meta_path.read_text())
        except Exception as e:
            log.warning("SelfImprover: failed reading meta=%s: %s", self.meta_path, e)
            return context

        # Try a few common places (keep flexible w.r.t. your trainer)
        auroc   = pick_auroc(meta, default=None)
        pr_auc  = deep_get(meta, ["holdout", "pr_auc"], ["eval", "pr_auc"])
        ece     = deep_get(meta, ["holdout", "ece"], ["calibration", "ece"], ["eval", "ece"])
        acc     = deep_get(meta, ["holdout", "accuracy"], ["eval", "accuracy"])

        new_rec = ModelRecord(
            model_path=str(self.model_path),
            meta_path=str(self.meta_path),
            auroc=auroc,
            ece=ece,
            pr_auc=pr_auc,
            accuracy=acc,
        )

        # 2) compare with current
        reg = ModelRegistry(self.registry_path)
        cur = reg.load_current()
        ok, reason = self.policy.ok(new_rec, cur)

        # 3) decide
        log.info("🔎 Promotion check: new(AUROC=%.4f, ECE=%s) vs cur(%s) → %s",
                 new_rec.auroc or -1, f"{new_rec.ece:.4f}" if new_rec.ece is not None else "NA",
                 f"AUROC={cur.auroc:.4f}, ECE={cur.ece:.4f}" if cur and cur.auroc is not None else "none",
                 "PROMOTE" if ok else f"REJECT ({reason})")

        context.setdefault("critic_self_improver", {})
        context["critic_self_improver"].update({
            "new": new_rec.__dict__,
            "current": cur.__dict__ if cur else None,
            "decision": "promote" if ok else f"reject: {reason}",
            "policy": self.policy.__dict__,
        })

        if ok:
            reg.promote(new_rec)

        return context


def deep_get(obj, *paths, default=None):
    """
    Safely traverse nested dicts. 
    Example: deep_get(meta, ["holdout_summary", "roc_auc"], ["cv", "auroc"], default=None)
    Returns first found value or default.
    """
    for path in paths:
        cur = obj
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            return cur
    return default

def pick_auroc(meta, default=None):
    """
    Look for AUROC under multiple spellings and containers.
    Supports: holdout / holdout_summary / cv / cv_summary / eval
    and metric names: auroc / roc_auc
    """
    containers = ["holdout_summary", "holdout", "cv_summary", "cv", "eval", "evaluation"]
    metric_keys = ["auroc", "roc_auc", "roc-auc", "rocAuc"]  # generous spellings
    paths = []
    for c in containers:
        for m in metric_keys:
            paths.append([c, m])
    # also allow top-level (rare)
    for m in metric_keys:
        paths.append([m])
    val = deep_get(meta, *paths, default=default)
    # best-effort cast
    try:
        return float(val) if val is not None else default
    except Exception:
        return default
