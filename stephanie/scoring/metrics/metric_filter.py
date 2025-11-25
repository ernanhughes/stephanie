from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

# Optional: fast MI/AUC if sklearn available; otherwise fall back to variance
try:
    from sklearn.metrics import roc_auc_score
    from sklearn.feature_selection import mutual_info_classif

    _SK = True
except Exception:
    _SK = False

_ALIAS_STRIP_RE = re.compile(
    r"\.(raw|debug|z|znorm|norm|standard|std)$", re.IGNORECASE
)


@dataclass
class MetricFilterReport:
    kept: List[str]
    dropped_lowvar: List[str]
    dropped_nonfinite: List[str]
    dropped_duplicates: List[Tuple[str, str, float]]  # (kept, dropped, sim)
    normalized: bool
    topk: int
    stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kept": self.kept,
            "dropped_lowvar": self.dropped_lowvar,
            "dropped_nonfinite": self.dropped_nonfinite,
            "dropped_duplicates": [
                {"kept": k, "dropped": d, "similarity": float(s)}
                for k, d, s in self.dropped_duplicates
            ],
            "normalized": self.normalized,
            "topk": self.topk,
            "stats": self.stats,
        }


class MetricFilter:
    """
    Input:
      - rows: list of ScorableProcessor rows (each has metrics_columns + metrics_values)
      - labels: optional list of 0/1 correctness (len = rows), improves ranking
    Output:
      - filtered matrix X_filtered (N x K), metric_names_filtered, report

    Guarantees:
      - All columns finite, in [0,1]
      - No duplicate/alias columns (post-filter similarity <= dup_threshold)
      - Deterministic ordering

    example config: add to processor:
    ---------------------------------   
      metrics_filter:
        enabled: true
        top_k: 100           # 50 / 100 / 150
        normalize: true
        dup_threshold: 0.995
        min_variance: 1e-8
        include:             # optional, supports * wildcards
            - "HRM.*"
            - "sicql.*"
            - "tiny.*"
        exclude:
            - "*.raw"
            - "*.debug"
            - "*.stdev"        # if you decide to exclude SD-only channels

    """

    def __init__(
        self,
        *,
        k: int = 100,
        dup_threshold: float = 0.995,  # cosine similarity to drop as dup
        min_variance: float = 1e-8,  # drop nearly-constant columns
        normalize: bool = True,
        include_patterns: Optional[List[str]] = None,  # glob-like or regex
        exclude_patterns: Optional[List[str]] = None,
        alias_strip: bool = True,
    ):
        self.k = int(k)
        self.dup_threshold = float(dup_threshold)
        self.min_variance = float(min_variance)
        self.normalize = bool(normalize)
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
        self.alias_strip = bool(alias_strip)

    # ---------------- Core API ----------------
    def run(
        self,
        rows: List[Dict[str, Any]],
        *,
        labels: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, List[str], MetricFilterReport]:
        if not rows:
            raise ValueError("MetricFilter: empty rows")

        names_union = self._union_metric_names(rows)
        names = self._apply_include_exclude(names_union)
        if not names:
            raise ValueError(
                "MetricFilter: no metric names after include/exclude"
            )

        # matrix build
        X = self._build_matrix(rows, names)  # (N, D)
        nonfinite = self._nonfinite_cols(X, names)

        # drop non-finite cols
        keep_mask = np.ones(X.shape[1], dtype=bool)
        for j in nonfinite:
            keep_mask[j] = False
        X, names = (
            X[:, keep_mask],
            [n for j, n in enumerate(names) if keep_mask[j]],
        )

        # variance filter
        lowvar = self._low_variance_cols(X, self.min_variance)
        keep_mask = np.ones(X.shape[1], dtype=bool)
        for j in lowvar:
            keep_mask[j] = False
        X, names = (
            X[:, keep_mask],
            [n for j, n in enumerate(names) if keep_mask[j]],
        )

        # alias collapse (string-level)
        if self.alias_strip:
            names, alias_map = self._alias_collapse(names)
            # merge exact alias duplicates (post-collapse name collisions)
            X, names = self._merge_exact_dups(X, names)

        # normalize to [0,1]
        normalized = False
        if self.normalize:
            X = self._safe_minmax(X)
            normalized = True

        # near-duplicate drop by cosine similarity
        kept_idx, dup_pairs = self._drop_near_duplicates(
            X, names, self.dup_threshold
        )
        X, names = X[:, kept_idx], [names[i] for i in kept_idx]

        # rank / select top-K
        rank_scores, rank_name = self._rank_columns(X, names, labels)
        order = np.argsort(-rank_scores)  # desc
        order = order[: min(self.k, X.shape[1])]
        Xf, names_f = X[:, order], [names[i] for i in order]

        report = MetricFilterReport(
            kept=names_f,
            dropped_lowvar=[
                names_union[i] for i in []
            ],  # keep simple; lowvar listed below
            dropped_nonfinite=[
                names_union[i] for i in []
            ],  # simple; detailed below
            dropped_duplicates=dup_pairs,
            normalized=normalized,
            topk=len(names_f),
            stats={
                "total_raw_names": len(names_union),
                "post_include_exclude": int(len(names_union)),
                "post_nonfinite": int(X.shape[1] + len(nonfinite)),
                "nonfinite_count": int(len(nonfinite)),
                "lowvar_count": int(len(lowvar)),
                "dup_pairs": len(dup_pairs),
                "rank_method": rank_name,
            },
        )
        return Xf, names_f, report

    # ---------------- Helpers ----------------
    def _union_metric_names(self, rows: List[Dict[str, Any]]) -> List[str]:
        seen = set()
        for r in rows:
            for n in r.get("metrics_columns") or []:
                seen.add(str(n))
        return sorted(seen)

    def _apply_include_exclude(self, names: List[str]) -> List[str]:
        def ok(n: str) -> bool:
            if self.include_patterns:
                if not any(
                    re.fullmatch(p.replace("*", ".*"), n)
                    for p in self.include_patterns
                ):
                    return False
            if self.exclude_patterns:
                if any(
                    re.fullmatch(p.replace("*", ".*"), n)
                    for p in self.exclude_patterns
                ):
                    return False
            return True

        return [n for n in names if ok(n)]

    def _build_matrix(self, rows, names) -> np.ndarray:
        name_to_pos = {n: i for i, n in enumerate(names)}
        X = np.zeros((len(rows), len(names)), dtype=np.float32)
        for i, r in enumerate(rows):
            cols = r.get("metrics_columns") or []
            vals = r.get("metrics_values") or []
            m = dict(zip(cols, vals))
            for n, j in name_to_pos.items():
                v = float(m.get(n, 0.0))
                X[i, j] = v
        return X

    def _nonfinite_cols(self, X: np.ndarray, names: List[str]) -> List[int]:
        bad = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if not np.all(np.isfinite(col)):
                bad.append(j)
        return bad

    def _low_variance_cols(self, X: np.ndarray, thr: float) -> List[int]:
        var = X.var(axis=0)
        return [int(i) for i, v in enumerate(var) if float(v) < thr]

    def _alias_collapse(
        self, names: List[str]
    ) -> Tuple[List[str], Dict[str, str]]:
        out, mp = [], {}
        for n in names:
            base = _ALIAS_STRIP_RE.sub("", n)
            mp[n] = base
            out.append(base)
        return out, mp

    def _merge_exact_dups(
        self, X: np.ndarray, names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        # combine identical-named columns by max (post-alias collapse)
        order = {}
        merged = []
        cols = []
        for j, n in enumerate(names):
            if n in order:
                jj = order[n]
                cols[jj] = np.maximum(cols[jj], X[:, j])
            else:
                order[n] = len(merged)
                merged.append(n)
                cols.append(X[:, j].copy())
        X2 = np.stack(cols, axis=1)
        return X2, merged

    def _safe_minmax(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        col_min = np.nanmin(X, axis=0)
        col_max = np.nanmax(X, axis=0)
        span = np.maximum(col_max - col_min, 1e-12)
        X = (X - col_min) / span
        X = np.clip(X, 0.0, 1.0)
        X[~np.isfinite(X)] = 0.0
        return X

    def _drop_near_duplicates(
        self, X: np.ndarray, names: List[str], thr: float
    ) -> Tuple[List[int], List[Tuple[str, str, float]]]:
        kept = []
        dup_pairs = []
        norms = np.linalg.norm(X, axis=0) + 1e-12
        for j in range(X.shape[1]):
            v = X[:, j]
            v /= norms[j]
            is_dup = False
            for k in kept:
                sim = float(
                    np.dot(v, X[:, k] / (np.linalg.norm(X[:, k]) + 1e-12))
                )
                if sim >= thr:
                    dup_pairs.append((names[k], names[j], sim))
                    is_dup = True
                    break
            if not is_dup:
                kept.append(j)
        return kept, dup_pairs

    def _rank_columns(
        self, X: np.ndarray, names: List[str], labels: Optional[Sequence[int]]
    ) -> Tuple[np.ndarray, str]:
        if labels is not None and _SK:
            y = np.asarray(labels, dtype=int)
            if (
                len(y) == X.shape[0]
                and (y.min() >= 0)
                and (y.max() <= 1)
                and (y.sum() not in (0, len(y)))
            ):
                # 1) MI for non-linear signal; 2) tie-break with AUC when possible
                mi = mutual_info_classif(
                    X, y, discrete_features=False, random_state=0
                )
                score = mi.astype(np.float64)
                # optional: boost columns with high AUC
                try:
                    aucs = np.array(
                        [roc_auc_score(y, X[:, j]) for j in range(X.shape[1])],
                        dtype=np.float64,
                    )
                    score = 0.7 * score + 0.3 * np.nan_to_num(aucs, nan=0.5)
                except Exception:
                    pass
                return score, "MI(+AUC)"
        # Fallback: variance (works even unlabeled)
        return X.var(axis=0).astype(np.float64), "variance"
