# stephanie/components/gap/processors/significance.py
from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from stephanie.utils.json_sanitize import dumps_safe
from typing import Any, Dict, List, Iterable

def _save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps_safe(obj, indent=2))

def _ensure_dict(d: Any) -> Dict[str, Any]:
    """Return a dict or {} if it's not a mapping."""
    return d if isinstance(d, dict) else {}

def _to_float_list(x: Any) -> List[float]:
    """
    Accepts:
    - list/tuple/ndarray of floats
    - list of dicts with 'top_H1_persistence' (or 'top'/'persistence')
    - a single float / numpy scalar
    Returns a clean List[float].
    """
    if x is None:
        return []
    # numpy array
    if isinstance(x, np.ndarray):
        try:
            x = x.tolist()
        except Exception:
            return []
    # numpy scalar
    if isinstance(x, (np.generic,)):
        try:
            return [float(x)]
        except Exception:
            return []
    # already a number
    if isinstance(x, (int, float)):
        return [float(x)]
    # iterable
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes, bytearray)):
        out: List[float] = []
        for item in x:
            if isinstance(item, dict):
                v = (
                    item.get("top_H1_persistence",
                        item.get("top",
                            item.get("persistence", 0.0)))
                )
                try:
                    out.append(float(v))
                except Exception:
                    out.append(0.0)
            elif isinstance(item, (int, float, np.generic)):
                try:
                    out.append(float(item))
                except Exception:
                    out.append(0.0)
            elif isinstance(item, np.ndarray):
                try:
                    out.extend([float(y) for y in item.ravel().tolist()])
                except Exception:
                    pass
        return out
    # fallback
    try:
        return [float(x)]
    except Exception:
        return []

@dataclass
class SignificanceConfig:
    n_nulls: int = 50
    n_bootstrap: int = 50
    random_seed: int = 42
    max_betti_dim: int = 1  # keep in sync with topology

class SignificanceProcessor:
    def __init__(self, cfg: SignificanceConfig, logger):
        self.cfg = cfg
        self.logger = logger

    async def run(self, run_id: str, base_dir: Path) -> dict:
        base_dir = Path(base_dir)
        met = base_dir / run_id / "metrics"
        ali = base_dir / run_id / "aligned"

        delta_path = ali / "delta_core5.npy"
        betti_path = met / "betti.json"
        if not delta_path.exists() or not betti_path.exists():
            self.logger.log("SignificanceMissingInputs", {"delta": str(delta_path), "betti": str(betti_path)})
            return {"status": "missing_inputs"}

        Delta = np.load(delta_path)                 # (N,5) exactly what PH used
        betti = json.loads(betti_path.read_text())
        observed = float(betti.get("top_H1_persistence", 0.0))
        np.random.seed(self.cfg.random_seed)

        # --- assumption checks
        assumptions = self._validate_tda_assumptions(Delta)
        _save_json(met / "tda_assumptions.json", assumptions)

        # --- nulls (strong)
        nulls = self._null_controls_stronger(Delta)
        _save_json(met / "nulls.json", nulls)

        # --- bootstrap CI (resample rows)
        boots = self._bootstrap_persistence(Delta)
        _save_json(met / "stability.json", {"bootstrap": boots})

        # --- significance from nulls (+ effect size)
        sig = self._statistical_significance(observed, nulls, boots)
        _save_json(met / "statistical_significance.json", sig)

        # --- sensitivity (quick)
        sens = self._parameter_sensitivity(Delta, observed)
        _save_json(met / "parameter_sensitivity.json", sens)

        return {"status": "ok", "observed_top_persistence": observed, "significance": sig, "assumptions": assumptions}

    # ---------- pieces ----------
    def _validate_tda_assumptions(self, D: np.ndarray) -> dict:
        n, d = D.shape
        uniq = len({tuple(x) for x in D})
        dup_ratio = float((n - uniq) / max(1, n))
        q75, q25 = np.percentile(D, [75, 25], axis=0); iqr = q75 - q25 + 1e-9
        outlier_ratio = float(np.mean((D < (q25 - 1.5*iqr)) | (D > (q75 + 1.5*iqr))))
        try: cond = float(np.linalg.cond(D.T @ D + 1e-8*np.eye(d)))
        except Exception: cond = float("inf")
        return {"n_samples": n, "n_features": d, "duplicate_ratio": dup_ratio,
                "outlier_ratio": outlier_ratio, "condition_number": cond,
                "numerical_stability": "good" if cond < 1e10 else "poor",
                "sample_size_adequate": n >= max(20, 2*d)}

    def _bootstrap_persistence(self, D: np.ndarray) -> list[dict]:
        from ripser import ripser
        out = []
        n = D.shape[0]; m = max(2, int(0.8*n))
        for _ in range(self.cfg.n_bootstrap):
            idx = np.random.choice(n, size=m, replace=False)
            H1 = ripser(D[idx], maxdim=self.cfg.max_betti_dim)["dgms"][1] if self.cfg.max_betti_dim>=1 else np.zeros((0,2))
            top = 0.0 if H1.size==0 else float(np.max(H1[:,1]-H1[:,0]))
            out.append({"top_H1_persistence": top})
        return out

    def _null_controls_stronger(self, Delta: np.ndarray, n_samples: int | None = None) -> np.ndarray:
        """
        Draw null controls with the empirical covariance of Delta.
        Robust to high-d; guarantees scalar 'size'; adds PSD jitter; has diagonal fallback.
        """
        Delta = np.asarray(Delta, dtype=np.float64)
        if Delta.ndim != 2 or Delta.size == 0:
            return np.zeros((0, 0), dtype=np.float64)

        n, d = Delta.shape

        # sensible default count; ensure PY-INT (not np.int32/np.ndarray)
        if n_samples is None:
            n_samples = max(512, min(4096, 8 * d))
        n_samples = int(n_samples)

        # empirical covariance (columns are features)
        Sigma = np.cov(Delta, rowvar=False)
        Sigma = 0.5 * (Sigma + Sigma.T)  # symmetrize

        # add tiny jitter so it's PSD even if Delta is skinny
        tr = np.trace(Sigma)
        eps = 1e-8 * (tr / max(d, 1) if tr > 0 else 1.0)
        Sigma = Sigma + np.eye(d) * eps

        mean = np.zeros(d, dtype=np.float64)

        try:
            g = np.random.multivariate_normal(mean=mean, cov=Sigma, size=n_samples)
        except Exception:
            # Fallback: diagonal-only sampling (keeps per-d variances)
            diag = np.clip(np.diag(Sigma), 1e-12, None)
            g = np.random.normal(loc=0.0, scale=np.sqrt(diag), size=(n_samples, d))

        return g




    def _statistical_significance(self, observed_top_persistence: float,
                                nulls_any: Any,
                                bootstrap_any: Any) -> Dict[str, Any]:
        """
        Robust version: accepts messy inputs and extracts numbers safely.
        """
        nulls = _ensure_dict(nulls_any)

        # pooled nulls from all sources
        sign_flip_vals = _to_float_list(nulls.get("sign_flip"))
        gauss_vals     = _to_float_list(nulls.get("gaussian_cov"))
        shuffled_vals  = _to_float_list(nulls.get("shuffled_pairing"))
        self_vals      = _to_float_list(
            [] if not isinstance(nulls.get("self_diff"), dict) else [
                nulls["self_diff"].get("H_minus_H_top", 0.0),
                nulls["self_diff"].get("T_minus_T_top", 0.0),
            ]
        )

        all_nulls = [v for v in (sign_flip_vals + gauss_vals + shuffled_vals + self_vals) if np.isfinite(v)]

        if not all_nulls:
            return {"p_value": 1.0, "significance": "insufficient_nulls"}

        # one-sided p
        obs = float(observed_top_persistence)
        n = len(all_nulls)
        p_value = float(sum(1 for v in all_nulls if v >= obs) / n)
        null_mean = float(np.mean(all_nulls))
        null_std  = float(np.std(all_nulls) + 1e-8)
        cohend    = float((obs - null_mean) / null_std)

        # bootstrap CI (robust)
        bs_vals = _to_float_list(bootstrap_any)
        if bs_vals:
            ci_low  = float(np.percentile(bs_vals, 2.5))
            ci_high = float(np.percentile(bs_vals, 97.5))
        else:
            ci_low = ci_high = obs

        return {
            "p_value": p_value,
            "effect_size_cohens_d": cohend,
            "null_mean": null_mean,
            "null_std": null_std,
            "bootstrap_ci_95": [ci_low, ci_high],
            "significance_level":
                "high" if p_value < 0.01 else
                "medium" if p_value < 0.05 else
                "low" if p_value < 0.10 else
                "not_significant",
        }
    
    def _parameter_sensitivity(self, D: np.ndarray, base: float) -> dict:
        from ripser import ripser
        out = {"maxdim_1": base}
        try:
            H1 = ripser(D, maxdim=2)["dgms"][1]
            out["maxdim_2"] = 0.0 if H1.size==0 else float(np.max(H1[:,1]-H1[:,0]))
        except Exception:
            out["maxdim_2"] = None
        return out
