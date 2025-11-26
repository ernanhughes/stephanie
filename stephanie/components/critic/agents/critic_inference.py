# stephanie/components/critic/agents/critic_inference.py
from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from stephanie.agents.base_agent import BaseAgent
import numpy as np
from sklearn.pipeline import Pipeline
from stephanie.components.critic.critic_self_eval import self_evaluate, update_competence_ema
from stephanie.components.critic.critic_teachpack import export_teachpack, teachpack_meta

log = logging.getLogger(__name__)

# -----------------------------
# Shadow pack I/O
# -----------------------------

@dataclass
class ShadowPack:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    groups: Optional[np.ndarray]
    meta: Dict[str, Any]

def _npz_get_dict(npz) -> Dict[str, Any]:
    """Robustly convert np.load(...) to a dict of arrays/objects."""
    out = {}
    for k in npz.files:
        out[k] = npz[k]
    return out

def _load_shadow(shadow_path: Path) -> ShadowPack:
    p = Path(shadow_path)
    if not p.exists():
        raise FileNotFoundError(f"critic_shadow not found at: {p}")
    with np.load(p, allow_pickle=True) as d:
        dd = _npz_get_dict(d)

    # Robust keys
    X = dd.get("X")
    if X is None:
        X = dd.get("features")
    y = dd.get("y")
    if y is None:
        y = dd.get("labels")
    groups = dd.get("groups")
    # feature names may be as array of str or an object pickled list
    feat = dd.get("feature_names")
    if feat is None:
        feat = dd.get("names")  # fallback
    if feat is None:
        # last-ditch: synthesize generic names by width
        n = X.shape[1] if X is not None else 0
        feat = np.array([f"col_{i}" for i in range(n)], dtype=object)
    feature_names = [str(x) for x in list(feat.tolist())]

    meta = dd.get("meta")
    if isinstance(meta, np.ndarray) and meta.dtype == object:
        meta = meta.item()
    if meta is None:
        # sometimes saved as JSON text
        meta_text = dd.get("meta_json")
        if meta_text is not None:
            try:
                meta = json.loads(str(meta_text))
            except Exception:
                meta = {}
    if meta is None:
        meta = {}

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64) if y is not None else None
    if groups is not None:
        groups = np.asarray(groups)

    log.info("[critic_inference] shadow: X=%s, y=%s, groups=%s, names=%d",
             X.shape if X is not None else None,
             y.shape if y is not None else None,
             groups.shape if isinstance(groups, np.ndarray) else None,
             len(feature_names))
    return ShadowPack(X=X, y=y, feature_names=feature_names, groups=groups, meta=meta)

# -----------------------------
# Model/Sidecar utilities
# -----------------------------

def _load_model(path: Path | str):
    path = Path(path)
    if not path.exists():
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception as e:
        log.warning("Failed to load model at %s: %s", path, e)
        return None

def _model_fingerprint_from_file(path: str) -> str:
    try:
        b = Path(path).read_bytes()
        return hashlib.sha1(b).hexdigest()
    except Exception:
        return "NA"

def _write_model_sidecar_features(model_path: Path | str,
                                  feature_names: List[str],
                                  meta: Optional[Dict[str, Any]] = None) -> None:
    """Write `<model>.features.txt` and `<model>.meta.json` best-effort."""
    mp = Path(model_path)
    feat_path = mp.with_suffix(mp.suffix + ".features.txt")
    meta_path = mp.with_suffix(mp.suffix + ".meta.json")
    try:
        feat_path.write_text("\n".join(feature_names), encoding="utf-8")
    except Exception as e:
        log.warning("Failed writing sidecar features: %s", e)
    if meta is not None:
        try:
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning("Failed writing sidecar meta: %s", e)

def _read_sidecar_feature_names(model_path: Path | str) -> Optional[List[str]]:
    mp = Path(model_path)
    feat_path = mp.with_suffix(mp.suffix + ".features.txt")
    if not feat_path.exists():
        return None
    try:
        names = [ln.strip() for ln in feat_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return names if names else None
    except Exception:
        return None

def _load_feature_names_for_model(model_path: str, shadow_path: Optional[str] = None) -> List[str]:
    """
    Priority:
      1) <model>.features.txt
      2) shadow.feature_names
      3) fall back to []
    """
    names = _read_sidecar_feature_names(model_path)
    if names:
        return names

    if shadow_path:
        try:
            sh = _load_shadow(Path(shadow_path))
            if sh.feature_names:
                return list(sh.feature_names)
        except Exception:
            pass
    return []  # will be reconciled to fitted width

def _model_n_features(model) -> Optional[int]:
    """Resolve the true fitted width from any step in a Pipeline or the model."""
    try:
        if isinstance(model, Pipeline):
            # Prefer steps' n_features_in_
            for _, step in model.named_steps.items():
                nfi = getattr(step, "n_features_in_", None)
                if nfi is not None:
                    return int(nfi)
            # Imputer width (statistics_)
            imp = model.named_steps.get("imputer")
            if imp is not None and hasattr(imp, "statistics_"):
                return int(imp.statistics_.shape[0])
        nfi = getattr(model, "n_features_in_", None)
        return int(nfi) if nfi is not None else None
    except Exception:
        return None

# -----------------------------
# Name reconcile & projection
# -----------------------------

def _reconcile_need_names(need: List[str], nfit: Optional[int], tag: str) -> List[str]:
    """
    Make the requested feature-name list consistent with the fitted pipeline width:
      - If longer than width → trim
      - If shorter → pad synthetic placeholders (projection zero-fills)
    """
    need = list(need or [])
    if nfit is None:
        return need
    if len(need) > nfit:
        log.warning("[%s] meta requested %d > pipeline width %d; trimming.", tag, len(need), nfit)
        return need[:nfit]
    if len(need) < nfit:
        pad_ct = nfit - len(need)
        log.warning("[%s] meta requested %d < pipeline width %d; padding %d synthetic cols.",
                    tag, len(need), nfit, pad_ct)
        need = need + [f"__pad_{i}__" for i in range(pad_ct)]
    return need

def _project_to_names(X: np.ndarray, have_names: List[str], need_names: List[str]) -> np.ndarray:
    """Reorder/expand X to match 'need_names'. Missing columns are zero-filled."""
    pos = {n: i for i, n in enumerate(have_names)}
    n = X.shape[0]
    Xp = np.zeros((n, len(need_names)), dtype=X.dtype)
    missing = []
    for j, name in enumerate(need_names):
        i = pos.get(name)
        if i is None:
            missing.append(name)
        else:
            Xp[:, j] = X[:, i]
    if missing:
        log.warning("[critic_inference] %d required features missing; padded 0.0 (e.g. %s)",
                    len(missing), ", ".join(missing[:10]) + ("..." if len(missing) > 10 else ""))
    return Xp

def _predict_proba1(model, X: np.ndarray, have_names: List[str], need_names: List[str], tag: str) -> np.ndarray:
    """Project then predict_proba with meta order; warn if pipeline width disagrees."""
    Xp = _project_to_names(X, have_names, need_names)
    try:
        n_fit = _model_n_features(model)
        if n_fit is not None and n_fit != Xp.shape[1]:
            log.warning("[%s] pipeline n_features_in_=%s but meta need_names=%s; continuing with meta order.",
                        tag, n_fit, Xp.shape[1])
    except Exception as e:
        log.warning("[%s] failed to verify model n_features_in_: %s", tag, e)
    return model.predict_proba(Xp)[:, 1]

# -----------------------------
# Agent
# -----------------------------

class CriticInferenceAgent(BaseAgent):
    """
    Compares current vs candidate TinyCritic models on a fixed shadow set.
    - Loads `models/critic_shadow.npz`
    - Loads current (`models/critic.joblib`) and candidate (`models/critic_candidate.joblib`)
    - Reconciles feature name lists with true pipeline widths (trim/pad)
    - Projects shadow X to each model's feature order
    - Computes AUROC/accuracy; optional promotion if candidate better
    - Writes a JSON report to `runs/critic_inference_report.json`
    """

    def __init__(self, cfg: Dict[str, Any], memory=None, container=None, logger=None):
        super().__init__(cfg, memory, container, logger)
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger

        # Paths
        self.shadow_path = Path(self.cfg.get("shadow_path", "models/critic_shadow.npz"))
        self.model_path = Path(self.cfg.get("model_path", "models/critic.joblib"))
        self.candidate_path = Path(self.cfg.get("candidate_path", "models/critic_candidate.joblib"))
        self.report_path = Path(self.cfg.get("report_path", "runs/critic_inference_report.json"))

        # Behavior
        self.metric_to_compare = str(self.cfg.get("metric_to_compare", "auroc")).lower()
        self.promote_when_better = bool(self.cfg.get("promote_when_better", True))

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # 0) Load shadow pack
        shadow = _load_shadow(self.shadow_path)
        X, y, have_names = shadow.X, shadow.y, shadow.feature_names
        groups = shadow.groups
        shadow_meta = shadow.meta or {}

        # 1) Load models
        cur = _load_model(self.model_path)
        cand = _load_model(self.candidate_path)
        if cur is None and cand is None:
            raise FileNotFoundError(f"Missing model(s): {self.model_path} and {self.candidate_path}")
        missing = []
        if cur is None:
            missing.append(str(self.model_path))
        if cand is None:
            missing.append(str(self.candidate_path))
        if missing:
            raise FileNotFoundError(f"Missing model(s): {missing}")

        # 2) Load feature-name lists
        need_current   = _load_feature_names_for_model(str(self.model_path),    shadow_path=str(self.shadow_path))
        need_candidate = _load_feature_names_for_model(str(self.candidate_path), shadow_path=str(self.shadow_path))

        # 3) Reconcile with fitted widths (key stabilization)
        nfit_cur  = _model_n_features(cur)
        nfit_cand = _model_n_features(cand)
        need_current   = _reconcile_need_names(need_current,   nfit_cur,  tag="current")
        need_candidate = _reconcile_need_names(need_candidate, nfit_cand, tag="candidate")

        log.info("[critic_inference] widths: shadow=%d | current_need=%d | candidate_need=%d",
                 X.shape[1], len(need_current), len(need_candidate))

        # 4) Project THEN predict on shadow (keep projected matrices for flips/teachpack)
        X_curr_proj = _project_to_names(X, have_names, need_current)
        X_cand_proj = _project_to_names(X, have_names, need_candidate)
        try:
            p_cur  = cur.predict_proba(X_curr_proj)[:, 1]
        except Exception as e:
            log.error("[current] predict_proba failed: %s", e)
            raise
        try:
            p_cand = cand.predict_proba(X_cand_proj)[:, 1]
        except Exception as e:
            log.error("[candidate] predict_proba failed: %s", e)
            raise

        # Self-eval (uses candidate's need list for fingerprinting)
        ser = self_evaluate(y, p_cur, p_cand, need_candidate)
        ema = update_competence_ema(Path(f"runs/critic/{self.run_id}/competence_state.json"), ser.competence)

        # 5) Metrics
        from sklearn.metrics import roc_auc_score, accuracy_score
        def _scores(y_true, y_prob):
            out = {}
            try:
                out["auroc"] = float(roc_auc_score(y_true, y_prob))
            except Exception:
                out["auroc"] = float("nan")
            try:
                y_hat = (y_prob >= 0.5).astype(int)
                out["accuracy"] = float(accuracy_score(y_true, y_hat))
            except Exception:
                out["accuracy"] = float("nan")
            return out

        metrics = {
            "current":   _scores(y, p_cur),
            "candidate": _scores(y, p_cand),
        }

        # 6) Promotion logic
        compare_key = self.metric_to_compare if self.metric_to_compare in ("auroc", "accuracy") else "auroc"
        new_val = metrics["candidate"].get(compare_key, float("nan"))
        cur_val = metrics["current"].get(compare_key, float("nan"))
        ece_new = "NA"; ece_cur = "NA"

        fp_cur  = _model_fingerprint_from_file(str(self.model_path))
        fp_cand = _model_fingerprint_from_file(str(self.candidate_path))

        promote = False
        if np.isnan(cur_val) and not np.isnan(new_val):
            promote = True
            log.info("🔎 Promotion check: new(%s=%.4f, ECE=%s) vs cur(none) → PROMOTE", compare_key, new_val, ece_new)
        elif not np.isnan(new_val) and not np.isnan(cur_val) and new_val > cur_val:
            promote = True
            log.info("🔎 Promotion check: new(%s=%.4f) > cur(%.4f) → PROMOTE", compare_key, new_val, cur_val)
        else:
            log.info("🔎 Promotion check: new(%s=%.4f, ECE=%s) vs cur(%.4f) → REJECT",
                     compare_key, new_val, ece_new, cur_val)

        # 6b) Error flips (unambiguous boolean grouping)
        bad_to_good = ((p_cur < 0.5) & (y == 1) & (p_cand >= 0.5))
        good_to_bad = ((p_cur >= 0.5) & (y == 0) & (p_cand < 0.5))
        flip_idx = np.where(bad_to_good | good_to_bad)[0]
        order = flip_idx[np.argsort(-np.abs(p_cand[flip_idx] - p_cur[flip_idx]))][:12]
        flips = order.tolist()

        # 6c) Teachpack export
        teach_dir = Path("runs/critic/teachpacks")
        teach_dir.mkdir(parents=True, exist_ok=True)
        teachpack_file = teach_dir / f"teachpack_{ser.feature_fingerprint}.npz"
        meta = teachpack_meta(fp_cand, need_candidate, calib=None)
        export_teachpack(teachpack_file, X_cand_proj, need_candidate, y, p_cand, meta)

        # 7) Report
        report = {
            **asdict(ser),
            "competence_ema": ema,
            "shadow_path": str(self.shadow_path),
            "n_samples": int(X.shape[0]),
            "n_features_shadow": int(X.shape[1]),
            "have_names_sample": have_names[:5],
            "current": {
                "model_path": str(self.model_path),
                "fingerprint": fp_cur,
                "expects": int(nfit_cur) if nfit_cur is not None else None,
                "need_names": len(need_current),
                "metrics": metrics["current"],
                "feature_hash": _digest_names(need_current),
            },
            "candidate": {
                "model_path": str(self.candidate_path),
                "fingerprint": fp_cand,
                "expects": int(nfit_cand) if nfit_cand is not None else None,
                "need_names": len(need_candidate),
                "metrics": metrics["candidate"],
                "feature_hash": _digest_names(need_candidate),
            },
            "compare_key": compare_key,
            "promote": bool(promote and self.promote_when_better),
            "teachpack": str(teachpack_file),
            "self_eval": {
                "error_flips_idx": flips
            }
        }
        try:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            self.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning("Failed to write inference report: %s", e)

        # 8) Promotion (file swap + sidecars)
        if promote and self.promote_when_better:
            try:
                Path(self.model_path).write_bytes(Path(self.candidate_path).read_bytes())
                _write_model_sidecar_features(self.model_path, need_candidate, meta={
                    "promoted_from": str(self.candidate_path),
                    "compare_key": compare_key,
                    "metrics": metrics["candidate"],
                })
                log.info("🚀 Promoted candidate → %s", self.model_path)
            except Exception as e:
                log.warning("Promotion failed: %s", e)

        # 9) Ledger append (use self.run_id)
        try:
            ledger = Path(f"runs/critic/{self.run_id}/promotion_ledger.jsonl")
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with ledger.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "shadow": str(self.shadow_path),
                    "current_fp": fp_cur,
                    "candidate_fp": fp_cand,
                    "current_feature_hash": _digest_names(need_current),
                    "candidate_feature_hash": _digest_names(need_candidate),
                    "metrics": metrics["candidate"],
                    "compare_key": compare_key,
                    "promote": bool(promote and self.promote_when_better),
                }) + "\n")
        except Exception as e:
            log.warning("promotion ledger write failed: %s", e)

        # 10) Attach to context
        context["critic_inference"] = report
        return context


def _digest_names(names: List[str]) -> str:
    import hashlib
    return hashlib.sha256("|".join(names).encode("utf-8")).hexdigest()[:12]

def top_error_flips(X_cur, X_cand, y, p_cur, p_cand, k=10):
    idx = np.where((p_cur >= 0.5) & (y == 0) & (p_cand < 0.5) | (p_cur < 0.5) & (y == 1) & (p_cand >= 0.5))[0]
    # prioritize biggest |p_cand - p_cur|
    order = idx[np.argsort(-np.abs(p_cand[idx] - p_cur[idx]))][:k]
    return order.tolist()

