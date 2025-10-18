# stephanie/scoring/score_bundle.py
import json
from dataclasses import dataclass, field
from statistics import pvariance, stdev
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stephanie.scoring.calculations.weighted_average import \
    WeightedAverageCalculator


@dataclass
class ScoreBundle:
    """Represents all scores for a single Scorable across dimensions and scorers

    Key features:
    - Contains ScoreResults with flexible attributes dictionary
    - Supports tensor operations for analysis
    - Works with MARS calculator for agreement analysis
    - Maintains compatibility with ORM layer
    """
    from stephanie.data.score_result import ScoreResult

    results: Dict[str, ScoreResult] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self, results: Dict[str, ScoreResult], meta: Dict[str, Any] = None
    ):
        self.results = results
        self.meta = meta or {}
        self.calculator = WeightedAverageCalculator()

    def aggregate(self) -> float:
        """Calculate weighted average score across dimensions"""
        return self.calculator.calculate(self)

    def get(self, dimension: str) -> Optional[ScoreResult]:
        return self.results.get(dimension)

    def to_dict(self, include_attributes: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for serialization and storage"""
        bundle_dict = {}
        for dim, result in self.results.items():
            result_dict = result.to_dict()
            if not include_attributes:
                # Remove attributes to keep the dictionary lean
                result_dict.pop("attributes", None)
            bundle_dict[dim] = result_dict

        # Add meta information if present
        if self.meta:
            bundle_dict["_meta"] = self.meta

        return bundle_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoreBundle":
        """Reconstruct from dictionary"""
        from stephanie.data.score_result import ScoreResult

        # Extract meta if present
        meta = data.pop("_meta", None)

        results = {
            dim: ScoreResult.from_dict(score_data)
            for dim, score_data in data.items()
            if isinstance(score_data, dict)
        }
        return cls(results=results, meta=meta)

    def merge(self, other: "ScoreBundle") -> "ScoreBundle":
        """
        Merge two bundles, preferring `self` values but including all from both.
        If a dimension exists in both, the value from `self` is kept.
        """
        merged = dict(self.results)
        for dim, result in other.results.items():
            if dim not in merged:
                merged[dim] = result
        return ScoreBundle(merged, meta={**self.meta, **other.meta})

    def to_json(
        self, stage: str, include_attributes: bool = False
    ) -> Dict[str, Any]:
        """Convert to JSON structure for reporting"""
        final_score = self.aggregate()
        return {
            "stage": stage,
            "dimensions": self.to_dict(include_attributes=include_attributes),
            "final_score": final_score,
            "meta": self.meta,
        }

    def to_orm(self, evaluation_id: int) -> List[Dict[str, Any]]:
        """Convert to ORM-compatible dictionaries for database storage"""
        orm_dicts = []
        for result in self.results.values():
            # Core score data
            orm_dict = {
                "evaluation_id": evaluation_id,
                "dimension": result.dimension,
                "score": result.score,
                "weight": result.weight,
                "rationale": result.rationale,
                "source": result.source,
                "target_type": result.target_type,
                "prompt_hash": result.prompt_hash,
            }

            # Include attribute references
            if result.attributes:
                orm_dict["attributes"] = result.attributes

            orm_dicts.append(orm_dict)
        return orm_dicts

    def to_tensor(
        self, dimensions: List[str], scorers: List[str], metrics: List[str]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Convert to 4D tensor: [1 × dimensions × scorers × metrics]
        For a single ScoreBundle (1 scorable)

        Returns:
            tensor: numpy array of shape (1, n_dimensions, n_scorers, n_metrics)
            metric_metadata: dictionary with metric information
        """
        import numpy as np

        tensor = np.zeros((1, len(dimensions), len(scorers), len(metrics)))
        metric_metadata = {
            "dimensions": dimensions,
            "scorers": scorers,
            "metrics": metrics,
        }

        for dim_idx, dimension in enumerate(dimensions):
            if dimension in self.results:
                result = self.results[dimension]
                scorer_idx = scorers.index(result.source)

                # Fill in metric values
                for metric_idx, metric in enumerate(metrics):
                    if metric in result.attributes:
                        try:
                            tensor[0, dim_idx, scorer_idx, metric_idx] = float(
                                result.attributes[metric]
                            )
                        except (TypeError, ValueError):
                            tensor[0, dim_idx, scorer_idx, metric_idx] = 0.0
                    else:
                        tensor[0, dim_idx, scorer_idx, metric_idx] = 0.0

        return tensor, metric_metadata

    def get_metric_values(self, metric: str) -> Dict[str, float]:
        """Get values for a specific metric across dimensions"""
        return {
            dim: result.attributes.get(metric, None)
            for dim, result in self.results.items()
        }

    def __repr__(self):
        summary = ", ".join(
            f"{dim}: {res.score:.2f}" for dim, res in self.results.items()
        )
        return f"<ScoreBundle({summary})>"

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)

    def to_report(self, title: str = "Score Report") -> str:
        """Generate a comprehensive report including tensor analysis capabilities"""
        lines = [f"## {title}", ""]

        # Add dimension scores
        for dim, result in self.results.items():
            lines.append(f"### Dimension: `{dim}`")
            lines.append(f"- **Score**: `{result.score:.4f}`")
            lines.append(f"- **Weight**: `{result.weight:.2f}`")
            lines.append(f"- **Source**: `{result.source}`")
            if result.rationale:
                lines.append(f"- **Rationale**: {result.rationale}")

            # Add attributes section if present
            if result.attributes:
                lines.append("\n**Extended Metrics:**")
                for key, value in result.attributes.items():
                    # Format value based on type
                    if isinstance(value, (int, float)):
                        formatted = f"{value:.4f}"
                    elif isinstance(value, (list, tuple)):
                        if len(value) > 5:
                            formatted = f"[{', '.join([f'{x:.4f}' for x in value[:5]])}, ...]"
                        else:
                            formatted = (
                                f"[{', '.join([f'{x:.4f}' for x in value])}]"
                            )
                    else:
                        formatted = str(value)
                    lines.append(f"- `{key}`: `{formatted}`")
            lines.append("")  # Empty line between dimensions

        # Add aggregate score
        lines.append(f"**Aggregate Score:** `{self.aggregate():.4f}`")

        return "\n".join(lines)

    def analyze_agreement(self, dimension: str = None) -> Dict[str, Any]:
        """Analyze agreement patterns across scorers for this bundle"""
        if dimension:
            # Analyze specific dimension
            if dimension not in self.results:
                return {"error": "dimension_not_found", "dimension": dimension}

            # For single dimension, there's only one scorer in this bundle
            # So agreement analysis doesn't apply at bundle level
            return {
                "dimension": dimension,
                "agreement_score": 1.0,  # Only one scorer for this dimension
                "explanation": "Single-scorer bundle - no agreement analysis needed",
            }

        # For cross-dimension analysis (not typically meaningful)
        scores = [r.score for r in self.results.values()]
        if len(scores) < 2:
            return {
                "agreement_score": 1.0,
                "explanation": "Only one dimension scored - no agreement analysis needed",
            }

        std_dev = stdev(scores)
        agreement_score = 1.0 - min(std_dev, 1.0)

        return {
            "agreement_score": round(agreement_score, 3),
            "std_dev": round(std_dev, 3),
            "dimension_count": len(scores),
            "explanation": f"Cross-dimension agreement score: {agreement_score:.3f}",
        }


    # ---------- rewards vector helpers ----------
    @staticmethod
    def _safe(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
        try:
            x = float(v)
            if x != x:  # NaN
                return 0.0
            return max(lo, min(hi, x))
        except Exception:
            return 0.0

    @staticmethod
    def _norm01(v: float) -> float:
        try:
            x = float(v)
        except Exception:
            return 0.0
        if x != x:
            return 0.0
        return 0.0 if x < 0 else (1.0 if x > 1 else x)

    @staticmethod
    def _var(values: List[float]) -> float:
        if not values:
            return 0.0
        try:
            return float(pvariance(values))
        except Exception:
            return 0.0

    def to_rewards_vector(
        self,
        *,
        prefix: str = "sicql_",                 # prefix for per-dimension keys
        q_key: str = "q_value",                 # metric names in ScoreResult.attributes
        v_key: str = "state_value",
        energy_key: str = "energy",
        include_overall: bool = True,
        advantage_sigmoid_scale: float = 10.0,  # scale for adv -> [0,1] squash
    ) -> Dict[str, float]:
        """
        Build a normalized rewards vector directly from this bundle.

        Outputs include:
          - per-dimension scores in 0..1 (e.g., sicql_alignment, sicql_clarity, ...)
          - aggregates: sicql_mean, sicql_var, sicql_min, sicql_max
          - cross-dim metrics: sicql_q_mean, sicql_v_mean, sicql_adv, sicql_adv_norm
          - energy: sicql_energy_mean, ebt_energy_inv (0..1 where higher is better)
          - overall_norm (aggregate()/100) if include_overall is True
        """
        rewards: Dict[str, float] = {}
        per_dim_norms: List[float] = []
        q_vals: List[float] = []
        v_vals: List[float] = []
        energies: List[float] = []

        # per-dimension normalized scores & metric harvest
        for dim, res in self.results.items():
            # score 0..100 -> 0..1
            try:
                dim_norm = max(0.0, min(1.0, float(res.score or 0.0) / 100.0))
            except Exception:
                dim_norm = 0.0
            key = f"{prefix}{str(dim).lower()}"
            rewards[key] = dim_norm
            per_dim_norms.append(dim_norm)

            # collect metrics from attributes (and optional .metrics if you use it)
            attrs = {}
            if hasattr(res, "attributes") and isinstance(res.attributes, dict):
                attrs.update(res.attributes)
            if hasattr(res, "metrics") and isinstance(res.metrics, dict):
                # don't overwrite if both exist; attributes win
                for k, v in res.metrics.items():
                    attrs.setdefault(k, v)

            if q_key in attrs:
                try: q_vals.append(float(attrs[q_key]))
                except Exception: pass
            if v_key in attrs:
                try: v_vals.append(float(attrs[v_key]))
                except Exception: pass
            if energy_key in attrs:
                try: energies.append(float(attrs[energy_key]))
                except Exception: pass

        # aggregates of per-dimension normalized scores
        if per_dim_norms:
            rewards[f"{prefix}mean"] = float(sum(per_dim_norms) / len(per_dim_norms))
            rewards[f"{prefix}var"]  = self._var(per_dim_norms)
            rewards[f"{prefix}min"]  = float(min(per_dim_norms))
            rewards[f"{prefix}max"]  = float(max(per_dim_norms))

        # cross-dimension metric aggregates
        if q_vals:
            rewards[f"{prefix}q_mean"] = float(sum(q_vals) / len(q_vals))
        if v_vals:
            rewards[f"{prefix}v_mean"] = float(sum(v_vals) / len(v_vals))
        if q_vals and v_vals:
            adv = rewards.get(f"{prefix}q_mean", 0.0) - rewards.get(f"{prefix}v_mean", 0.0)
            rewards[f"{prefix}adv"] = float(adv)
            # squash to [0,1]
            try:
                import math
                rewards[f"{prefix}adv_norm"] = 1.0 / (1.0 + math.exp(-adv / max(1e-6, advantage_sigmoid_scale)))
            except Exception:
                rewards[f"{prefix}adv_norm"] = 0.5

        # energy → mean and inverse goodness in [0,1]
        if energies:
            mean_energy = float(sum(energies) / len(energies))
            rewards[f"{prefix}energy_mean"] = mean_energy
            # map "lower is better" to [0,1]; scale by magnitude for stability
            scale = max(1.0, abs(mean_energy))
            inv = 1.0 / (1.0 + max(0.0, mean_energy) / scale)
            rewards["ebt_energy_inv"] = max(0.0, min(1.0, inv))

        if include_overall:
            try:
                overall = float(self.aggregate() or 0.0) / 100.0
            except Exception:
                overall = 0.0
            rewards["overall_norm"] = max(0.0, min(1.0, overall))

        return rewards

    def flatten(
        self,
        *,
        include_scores: bool = True,
        include_weights: bool = False,
        include_sources: bool = False,
        include_rationales: bool = False,
        include_attributes: bool = True,
        include_meta: bool = False,
        numeric_only: bool = True,
        sep: str = ".",
        attr_prefix: str = "attr",
        meta_prefix: str = "meta",
        list_policy: str = "index",   # "index" | "sum" | "mean" | "len" | "first"
        max_list_len: int = 16,
        dict_policy: str = "flatten", # "flatten" | "drop" | "string"
    ) -> Dict[str, float]:
        """
        Safer flattener that supports tensors/ndarrays, lists, and nested dicts.

        - Tensors/ndarrays: scalars -> float; vectors -> policy (index/mean/etc)
        - Lists/tuples: same policy
        - Dicts: "flatten" (default) flattens with dotted keys; "drop" skips; "string" keeps str()
        """
        out: Dict[str, Any] = {}

        # Lazy imports (optional if torch/numpy not installed in some envs)
        try:
            import numpy as _np
        except Exception:
            _np = None
        try:
            import torch as _torch
        except Exception:
            _torch = None

        def _is_scalar_number(v) -> bool:
            try:
                float(v)
                return True
            except Exception:
                return False

        def _emit_scalar(key: str, val: Any):
            # Only place we actually write into out
            try:
                x = float(val)
                if x == x:  # not NaN
                    out[key] = x
            except Exception:
                if not numeric_only:
                    out[key] = str(val)

        def _emit_list(key: str, values: list | tuple):
            # Convert elements to floats where possible
            nums: List[float] = []
            for v in values:
                cv = _coerce_scalar(v)
                if cv is not None:
                    nums.append(cv)

            # NEW: no numeric elements → skip in numeric_only mode
            if not nums:
                # Optional alternative if you prefer to keep a signal:
                # out[f"{key}.len"] = float(len(values))
                return

            if list_policy == "index":
                for i, vv in enumerate(nums[:max_list_len]):
                    out[f"{key}[{i}]"] = float(vv)
            elif list_policy == "sum":
                out[key] = float(sum(nums))
            elif list_policy == "mean":
                out[key] = float(sum(nums) / max(1, len(nums)))
            elif list_policy == "len":
                out[key] = float(len(nums))
            elif list_policy == "first":
                out[key] = float(nums[0])

        def _emit_dict(prefix: str, d: Dict[str, Any]):
            if dict_policy == "drop":
                return
            if dict_policy == "string" and not numeric_only:
                out[prefix] = str(d)
                return
            # default: flatten
            for k, v in d.items():
                subkey = f"{prefix}{sep}{k}" if prefix else str(k)
                _emit_any(subkey, v)

        def _coerce_scalar(v: Any) -> Optional[float]:
            """Try to coerce v to a single float; return None if not possible."""
            if isinstance(v, str):
                return None

            # torch.Tensor
            if _torch is not None and isinstance(v, _torch.Tensor):
                if v.numel() == 0:
                    return None
                try:
                    return float(v.detach().cpu().reshape(-1)[0].item())
                except Exception:
                    try:
                        return float(v.detach().cpu().mean().item())
                    except Exception:
                        return None

            # numpy scalar/array
            if _np is not None:
                if isinstance(v, _np.generic):
                    try:
                        return float(v.item())
                    except Exception:
                        return None
                if isinstance(v, _np.ndarray):
                    if v.size == 0:
                        return None
                    try:
                        return float(_np.asarray(v).reshape(-1)[0])
                    except Exception:
                        try:
                            return float(_np.asarray(v, dtype=float).mean())
                        except Exception:
                            return None

            # plain number
            if isinstance(v, (int, float)) and v == v:
                return float(v)

            # generic float()
            try:
                fv = float(v)
                return fv if fv == fv else None
            except Exception:
                return None

        def _emit_any(key: str, val: Any):
            # Dicts
            if isinstance(val, dict):
                if numeric_only and dict_policy == "string":
                    # still flatten to avoid strings in numeric mode
                    _emit_dict(key, val)
                else:
                    _emit_dict(key, val)
                return

            # Tensors & NumPy arrays -> use list policy if non-scalar
            if (_torch is not None and isinstance(val, _torch.Tensor)) or (_np is not None and isinstance(val, _np.ndarray)):
                # scalar?
                scalar = _coerce_scalar(val)
                if scalar is not None and (
                    (getattr(val, "numel", lambda: 1)() == 1) if _torch is not None and isinstance(val, _torch.Tensor)
                    else (getattr(val, "size", lambda: 1) == 1) if _np is not None and isinstance(val, _np.ndarray)
                    else True
                ):
                    _emit_scalar(key, scalar)
                    return
                # treat as list
                try:
                    seq = val.detach().cpu().tolist() if (_torch is not None and isinstance(val, _torch.Tensor)) else val.tolist()
                except Exception:
                    # fallback to mean
                    scalar = _coerce_scalar(val)
                    if scalar is not None:
                        _emit_scalar(key, scalar)
                    return
                _emit_list(key, seq)
                return

            # Lists/tuples
            if isinstance(val, (list, tuple)):
                _emit_list(key, val)
                return

            # Scalars or other
            scalar = _coerce_scalar(val)
            if scalar is not None:
                _emit_scalar(key, scalar)
            else:
                if not numeric_only:
                    out[key] = str(val)
                # else drop silently

        # ---- Walk the bundle ----
        for dim, res in self.results.items():
            base = f"{dim}"
            if include_scores:
                _emit_any(f"{base}{sep}score", res.score)
            if include_weights:
                _emit_any(f"{base}{sep}weight", res.weight)
            if include_sources:
                _emit_any(f"{base}{sep}source", getattr(res, "source", None))
            if include_rationales:
                _emit_any(f"{base}{sep}rationale", getattr(res, "rationale", None))

            if include_attributes and getattr(res, "attributes", None):
                for k, v in res.attributes.items():
                    _emit_any(f"{base}{sep}{attr_prefix}{sep}{k}", v)

            if include_attributes and getattr(res, "metrics", None):
                for k, v in res.metrics.items():
                    kpath = f"{base}{sep}{attr_prefix}{sep}{k}"
                    if kpath not in out:
                        _emit_any(kpath, v)

        if include_meta and self.meta:
            for k, v in self.meta.items():
                _emit_any(f"{meta_prefix}{sep}{k}", v)

        return out
