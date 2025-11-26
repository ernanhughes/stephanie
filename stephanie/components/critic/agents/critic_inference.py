from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Any, Dict

import numpy as np
import joblib

from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve


from stephanie.agents.base_agent import BaseAgent

def _index_map(names: list[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(names)}

def _project_to_names(X, have: list[str], need: list[str]):
    """
    Reorder/align X to exactly the columns in `need`.
    Any missing column is filled with 0.0; extras are dropped.
    Returns (X_proj, ok, missing, extra).
    """
    import numpy as np

    have_idx = _index_map(have)
    d = X.shape[0]
    Z = np.zeros((d, len(need)), dtype=X.dtype)

    missing, used = [], set()
    for j, name in enumerate(need):
        i = have_idx.get(name)
        if i is None:
            missing.append(name)
        else:
            Z[:, j] = X[:, i]
            used.add(i)

    extra = [have[i] for i in range(len(have)) if i not in used]
    ok = (len(missing) == 0 and len(extra) == 0)
    return Z, ok, missing, extra

def _print_feature_drift(tag: str, have: list[str], need: list[str]):
    hs, ns = set(have), set(need)
    missing = sorted(ns - hs)
    extra   = sorted(hs - ns)
    if missing or extra:
        import logging
        log = logging.getLogger(__name__)
        log.warning("[%s] Feature drift detected", tag)
        if missing:
            log.warning("  missing (%d): %s", len(missing), missing[:12] + (["…"] if len(missing) > 12 else []))
        if extra:
            log.warning("  extra   (%d): %s", len(extra),   extra[:12]   + (["…"] if len(extra) > 12 else []))

def _sha1_bytes(x: bytes) -> str:
    return hashlib.sha1(x).hexdigest()[:12]

def _safe(obj):  # json-safe floats
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj

def _metrics(y_true, p1) -> Dict[str, float]:
    # p1 = prob for class 1
    # Robust to edge cases; return NaNs (as floats) if metric is undefined
    def _try(fn, default=np.nan):
        try:
            return float(fn())
        except Exception:
            return float(default)
    return {
        "auroc": _try(lambda: roc_auc_score(y_true, p1)),
        "accuracy@0.5": _try(lambda: accuracy_score(y_true, (p1 >= 0.5).astype(int))),
        "brier": _try(lambda: brier_score_loss(y_true, p1)),
        "logloss": _try(lambda: log_loss(y_true, np.vstack([1 - p1, p1]).T, labels=[0,1])),
        # Simple ECE proxy (10-bin expected calibration error)
        "ece10": _try(lambda: _ece_bin(y_true, p1, bins=10)),
    }

def _ece_bin(y_true, p1, bins=10) -> float:
    frac_pos, mean_pred = calibration_curve(y_true, p1, n_bins=bins, strategy="uniform")
    # |acc - conf| weighted by bin mass
    # Compute bin masses approximately from histogram
    hist, edges = np.histogram(p1, bins=bins, range=(0,1))
    mass = hist / max(1, hist.sum())
    # align lengths
    m = min(len(frac_pos), len(mean_pred), len(mass))
    return float(np.sum(np.abs(frac_pos[:m] - mean_pred[:m]) * mass[:m]))


def _load_models_safe(current_path: str, candidate_path: str | None = None) -> dict:
    """
    Load available models without crashing if candidate is missing.
    Returns dict like {"current": model, "candidate": model?}.
    """
    out = {}
    cur_p = Path(current_path)
    if cur_p.exists():
        out["current"] = joblib.load(cur_p)
    else:
        raise FileNotFoundError(f"Missing model: {current_path}")

    if candidate_path:
        cand_p = Path(candidate_path)
        if cand_p.exists():
            out["candidate"] = joblib.load(cand_p)
    return out

def _load_shadow(path: str):
    """
    Load a 'critic_shadow.npz' pack in a tolerant way.
    Accepts multiple historical key layouts.
    Returns: X, y, kept_feature_names, groups, meta_dict
    """
    with np.load(path, allow_pickle=True) as d:
        keys = set(d.files)

        # --- X ---
        if "X" in keys:
            X = d["X"]
        elif "features" in keys:
            X = d["features"]
        else:
            raise KeyError(f"shadow pack missing feature matrix; keys={sorted(keys)}")

        # --- y ---
        if "y" in keys:
            y = d["y"]
        elif "labels" in keys:
            y = d["labels"]
        else:
            # y can be absent for pure-inference packs; synthesize zeros if needed
            y = np.zeros((X.shape[0],), dtype=np.int64)

        # --- feature names / kept columns ---
        kept = None
        for k in ("feature_names", "kept", "names", "metric_names"):
            if k in keys:
                arr = d[k]
                # np.savez can wrap lists as 0-d object arrays; normalize:
                try:
                    kept = arr.tolist()
                except Exception:
                    kept = list(arr)
                # ensure list[str]
                kept = [str(x) for x in kept]
                break
        if kept is None:
            raise KeyError(f"'feature_names' (or kept/names/metric_names) not found; keys={sorted(keys)}")

        # --- groups ---
        if "groups" in keys:
            groups = d["groups"]
        else:
            groups = np.array(["unknown"] * X.shape[0])

        # --- meta ---
        meta = {}
        if "meta" in keys:
            m = d["meta"]
            # could be json string, dict-like object, or 0-d object array
            if isinstance(m, np.ndarray) and m.dtype == object:
                m = m.item()
            if isinstance(m, (bytes, str)):
                try:
                    m = json.loads(m)
                except Exception:
                    pass
            if isinstance(m, dict):
                meta = m

    return X, y, kept, groups, meta


def _required_feature_names_for(model_tag: str, *, current_meta: dict | None, candidate_meta: dict | None, shadow_meta: dict | None) -> list[str]:
    """
    Decide which feature-name ordering to require for a given model.
    - current: prefer current_meta["feature_names"], else shadow
    - candidate: prefer candidate_meta["feature_names"], else shadow
    """
    if model_tag == "current":
        names = (current_meta or {}).get("feature_names") or []
        return list(names or (shadow_meta or {}).get("feature_names") or [])
    else:
        names = (candidate_meta or {}).get("feature_names") or []
        return list(names or (shadow_meta or {}).get("feature_names") or [])

def _predict_proba1(model, X, have_names: list[str], need_names: list[str], tag: str):
    import numpy as np
    # Replace NaNs defensively
    if np.isnan(X).any():
        from numpy import nan_to_num
        X = nan_to_num(X, copy=False)

    Xp, ok, missing, extra = _project_to_names(X, have_names, need_names)
    if not ok:
        _print_feature_drift(tag, have_names, need_names)

    # ---- Hard sanity against fitted n_features_in_ if available ----
    # Pipeline → look at final estimator or first step with n_features_in_
    n_fit = None
    try:
        if hasattr(model, "n_features_in_"):
            n_fit = int(model.n_features_in_)
        elif hasattr(model, "steps") and model.steps:
            for _, step in model.steps[::-1]:
                if hasattr(step, "n_features_in_"):
                    n_fit = int(step.n_features_in_)
                    break
    except Exception:
        n_fit = None

    if n_fit is not None and Xp.shape[1] != n_fit:
        raise ValueError(f"[{tag}] Feature count mismatch after projection: X has {Xp.shape[1]}, model expects {n_fit}. "
                         f"(need_names={len(need_names)}, have_names={len(have_names)})")

    return model.predict_proba(Xp)[:, 1]

def _model_fingerprint(model) -> str:
    try:
        from joblib import hash as joblib_hash
        return joblib_hash(model)[:12]
    except Exception:
        import hashlib
        return hashlib.sha1(repr(model).encode("utf-8")).hexdigest()[:12]

class CriticInferenceAgent(BaseAgent):
    """
    Compares CURRENT vs CANDIDATE critic models on a fixed shadow dataset.
    Emits a markdown report + context fields for the improver/promotion step.
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Files
        self.shadow_path   = Path(cfg.get("shadow_path", "models/critic_shadow.npz"))
        self.current_model = Path(cfg.get("current_model_path", "models/critic.joblib"))
        self.current_meta  = Path(cfg.get("current_meta_path",   "models/critic.meta.json"))
        self.candidate_model = Path(cfg.get("candidate_model_path", "models/critic_candidate.joblib"))
        self.candidate_meta  = Path(cfg.get("candidate_meta_path",   "models/critic_candidate.meta.json"))
        self.report_dir    = Path(cfg.get("report_dir", "runs/critic/reports"))
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Gates
        g = cfg.get("promotion_gates", {}) or {}
        self.min_auroc_gain = float(g.get("min_auroc_gain", 0.00))
        self.max_ece_increase = float(g.get("max_ece_increase", 0.02))
        self.min_auroc_absolute = float(g.get("min_auroc_absolute", 0.50))
        self.require_shadow_version_match = bool(g.get("require_shadow_version_match", False))

        # Optional plotting
        viz = cfg.get("viz", {}) or {}
        self.enable_plots = bool(viz.get("enabled", True))
        self.viz_dir = self.report_dir / "viz"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log = self.logger
        log.info("CriticInference: loading shadow dataset → %s", self.shadow_path)
        X, y, kept, groups, shadow_meta = _load_shadow(self.shadow_path)

        # Optional: check version alignment
        if self.require_shadow_version_match:
            run_id = str(context.get("pipeline_run_id", ""))
            if shadow_meta.get("locked_run_id") and str(shadow_meta["locked_run_id"]) != run_id:
                log.warning("Shadow locked_run_id %s != current run_id %s",
                            shadow_meta["locked_run_id"], run_id)

        have_names = list(shadow_meta.get("feature_names") or kept or [])
        # Load current meta.json (the one saved with the *current* model)
        current_meta = {}
        # Load current + candidate meta
        with open(self.current_meta, "r", encoding="utf-8") as f:
            current_meta = json.load(f)
        candidate_meta = {}
        if self.candidate_meta.exists():
            with open(self.candidate_meta, "r", encoding="utf-8") as f:
                candidate_meta = json.load(f)

        have_names = list(shadow_meta.get("feature_names") or kept or [])

        # shadow_meta already loaded by _load_shadow(...)
        need_current   = _pick_used_names("current",   candidate_meta=None,          shadow_meta=shadow_meta, current_meta=current_meta)
        need_candidate = _pick_used_names("candidate", candidate_meta=candidate_meta, shadow_meta=shadow_meta, current_meta=current_meta)
        # Fallbacks if empty
        if not need_current:
            need_current = have_names
        if not need_candidate:
            need_candidate = have_names


        # Load models
        log.info("CriticInference: loading models")
        cur = joblib.load(self.current_model) if self.current_model.exists() else None
        cand = joblib.load(self.candidate_model) if self.candidate_model.exists() else None
        if cur is None or cand is None:
            missing = []
            if cur is None: missing.append(str(self.current_model))
            if cand is None: missing.append(str(self.candidate_model))
            raise FileNotFoundError(f"Missing model(s): {missing}")

        p_cur  = _predict_proba1(cur,  X, have_names, need_current,   tag="current")
        p_cand = _predict_proba1(cand, X, have_names, need_candidate, tag="candidate")


        m_cur  = _metrics(y, p_cur)
        m_cand = _metrics(y, p_cand)

        decision = self._decide(m_cur, m_cand)

        # Try plots, but never fail the run
        artifacts = {}
        if self.enable_plots:
            try:
                artifacts.update(self._maybe_plots(y, p_cur, p_cand))
            except Exception as e:
                log.warning("CriticInference: plotting skipped (%s)", e)

        # Write markdown report
        report_path = self._write_report(
            X, y, kept, groups, shadow_meta,
            m_cur=m_cur, m_cand=m_cand, decision=decision, artifacts=artifacts
        )

        # Fill context for improver/promotion agent
        context["critic_compare"] = {
            "shadow_path": str(self.shadow_path),
            "kept_feature_count": len(kept),
            "n_samples": int(len(y)),
            "metrics_current": m_cur,
            "metrics_candidate": m_cand,
            "decision": decision,
            "report_path": str(report_path),
            "current_fp": _model_fingerprint(cur),
            "candidate_fp": _model_fingerprint(cand),
        }
        self.logger.info("CriticInference: compare complete → %s", report_path)
        return context

    # ------------------------------------------------------------------
    def _decide(self, cur: Dict[str,float], cand: Dict[str,float]) -> Dict[str, Any]:
        au_gain = _safe(cand["auroc"] - cur["auroc"])
        ece_delta = _safe(cand["ece10"] - cur["ece10"])
        pass_au_abs  = bool(cand["auroc"] >= self.min_auroc_absolute)
        pass_au_gain = bool(au_gain >= self.min_auroc_gain)
        pass_ece     = bool(ece_delta <= self.max_ece_increase)

        promote = bool(pass_au_abs and pass_au_gain and pass_ece)
        return {
            "promote": promote,
            "reasons": {
                "auroc_gain": au_gain,
                "ece10_delta": ece_delta,
                "min_auroc_absolute": self.min_auroc_absolute,
                "min_auroc_gain": self.min_auroc_gain,
                "max_ece_increase": self.max_ece_increase,
                "gates": {
                    "pass_auroc_abs": pass_au_abs,
                    "pass_auroc_gain": pass_au_gain,
                    "pass_ece_delta": pass_ece,
                }
            }
        }

    def _maybe_plots(self, y, p_cur, p_cand) -> Dict[str, str]:
        # Import locally to avoid hard deps during training-only runs
        import matplotlib.pyplot as plt

        paths = {}

        # ROC curves
        try:
            from sklearn.metrics import RocCurveDisplay
            fig = plt.figure()
            RocCurveDisplay.from_predictions(y, p_cur, name="current")
            RocCurveDisplay.from_predictions(y, p_cand, name="candidate")
            out = self.viz_dir / "roc_compare.png"
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            paths["roc"] = str(out)
        except Exception:
            pass

        # Reliability diagram
        try:
            fig = plt.figure()
            for p, name in [(p_cur, "current"), (p_cand, "candidate")]:
                frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
                plt.plot(mean_pred, frac_pos, marker="o", label=name)
            plt.plot([0,1], [0,1], "k--", linewidth=1)
            plt.xlabel("Mean predicted value"); plt.ylabel("Fraction of positives")
            plt.legend()
            out = self.viz_dir / "reliability_compare.png"
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            paths["reliability"] = str(out)
        except Exception:
            pass

        return paths

    def _write_report(self, X, y, kept, groups, shadow_meta, *, m_cur, m_cand, decision, artifacts) -> Path:
        report = []
        report.append("# Critic Model A/B Comparison (shadow)\n")
        report.append(f"- Samples: **{len(y)}**")
        report.append(f"- Features (kept): **{len(kept)}**")
        if "shadow_version" in shadow_meta:
            report.append(f"- Shadow version: `{shadow_meta['shadow_version']}`")
        report.append("")
        report.append("## Metrics\n")
        def row(label, m):
            return f"| {label} | {m['auroc']:.4f} | {m['accuracy@0.5']:.4f} | {m['brier']:.4f} | {m['logloss']:.4f} | {m['ece10']:.4f} |"
        report.append("| model | AUROC | Acc@0.5 | Brier | LogLoss | ECE@10 |")
        report.append("|---|---:|---:|---:|---:|---:|")
        report.append(row("current", m_cur))
        report.append(row("candidate", m_cand))
        report.append("")
        report.append("## Decision\n")
        report.append(f"- **Promote**: `{decision['promote']}`")
        reasons = decision["reasons"]
        report.append(f"- AUROC gain: `{reasons['auroc_gain']:.4f}` (gate ≥ {self.min_auroc_gain})")
        report.append(f"- ECE Δ: `{reasons['ece10_delta']:.4f}` (gate ≤ {self.max_ece_increase})")
        report.append(f"- AUROC absolute gate: ≥ {self.min_auroc_absolute}")
        report.append("")
        if artifacts:
            report.append("## Artifacts")
            for k, v in artifacts.items():
                report.append(f"- {k}: {v}")
            report.append("")
        report.append("## Shadow meta (excerpt)\n")
        snippet = {k: shadow_meta[k] for k in sorted(shadow_meta.keys()) if k in ("locked_run_id","kept_hash","shadow_version","built_at")}
        report.append("```json")
        report.append(json.dumps(snippet, indent=2))
        report.append("```")

        out = self.report_dir / f"critic_compare_{_sha1_bytes(np.random.bytes(8))}.md"
        out.write_text("\n".join(report), encoding="utf-8")
        return out

def _pick_used_names(tag: str, *, candidate_meta, shadow_meta, current_meta) -> list[str]:
    # Strict priority: candidate → shadow → current
    for src, meta in (("candidate", candidate_meta),
                      ("shadow", shadow_meta),
                      ("current", current_meta)):
        if meta and "feature_names_used" in meta and meta["feature_names_used"]:
            return list(meta["feature_names_used"])
        if meta and "feature_names" in meta and meta["feature_names"]:
            # fallback if older trainer only wrote "feature_names"
            return list(meta["feature_names"])
    raise RuntimeError(f"[{tag}] no feature name list available (candidate/shadow/current)")
