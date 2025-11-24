# stephanie/components/critic/model/critic_model.py
from __future__ import annotations

import json
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Any, Tuple, Union

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

# Default artifact locations (match your trainer)
DEFAULT_MODEL_PATH = Path("models/critic.joblib")
DEFAULT_META_PATH  = Path("models/critic.meta.json")


@dataclass(frozen=True)
class CriticModelMeta:
    feature_names: List[str]
    core_only: bool
    locked_features_path: Optional[str]
    directionality: Dict[str, int]          # name -> {1 | -1}
    cv_summary: Dict[str, Any]
    holdout_summary: Dict[str, Any]


class CriticModel:
    """
    Lightweight inference wrapper for the Tiny Critic.

    - Loads sklearn Pipeline (StandardScaler + LogisticRegression)
    - Aligns incoming features to `meta.feature_names`
    - Applies the same directionality correction used in training
    - Provides batch & single scoring + simple coefficient-based explanation
    """

    def __init__(self, model: Pipeline, meta: CriticModelMeta):
        if not isinstance(model, Pipeline):
            raise TypeError("Expected an sklearn Pipeline (StandardScaler -> LogisticRegression).")
        self.model = model
        self.meta = meta

        # Convenience handles (ok if names differ, we search them)
        self._scaler = None
        self._clf = None
        for name, step in self.model.named_steps.items():
            cls = step.__class__.__name__.lower()
            if "scaler" in cls:
                self._scaler = step
            if "logistic" in cls:
                self._clf = step

        if self._clf is None:
            raise ValueError("Pipeline must contain a LogisticRegression step.")
        if self._scaler is None:
            # Not strictly required, but explanations need it. We allow None and degrade gracefully.
            pass

        # Precompute index map for fast alignment
        self._name_to_pos = {n: i for i, n in enumerate(self.meta.feature_names)}

    # ---------- Loading ----------

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        meta_path: Union[str, Path] = DEFAULT_META_PATH,
    ) -> CriticModel:
        model_path = Path(model_path)
        meta_path = Path(meta_path)

        model = joblib.load(model_path)
        meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = CriticModelMeta(
            feature_names=list(meta_raw["feature_names"]),
            core_only=bool(meta_raw.get("core_only", False)),
            locked_features_path=meta_raw.get("locked_features_path"),
            directionality=dict(meta_raw.get("directionality", {})),
            cv_summary=dict(meta_raw.get("cv_summary", {})),
            holdout_summary=dict(meta_raw.get("holdout_summary", {})),
        )
        return cls(model, meta)

    # ---------- Feature prep ----------

    def _align_and_direction_correct(
        self,
        features: Union[Dict[str, float], Sequence[float]],
        incoming_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Produce a (1, D) vector aligned to meta.feature_names with directionality applied.
        - If `features` is a dict: pick values by name; missing -> 0.0; extra ignored.
        - If `features` is a sequence: you MUST provide `incoming_names` to map positions.
        """
        target_names = self.meta.feature_names
        x = np.zeros((1, len(target_names)), dtype=np.float32)

        if isinstance(features, dict):
            for name, val in features.items():
                pos = self._name_to_pos.get(name)
                if pos is not None:
                    x[0, pos] = float(val)
        else:
            if incoming_names is None:
                raise ValueError("When passing a numeric sequence, you must provide `incoming_names`.")
            name_to_in = {n: i for i, n in enumerate(incoming_names)}
            for j, name in enumerate(target_names):
                i = name_to_in.get(name)
                if i is not None:
                    x[0, j] = float(features[i])

        # Apply directionality flip (multiply by -1 for features trained as 'higher=worse')
        for j, name in enumerate(target_names):
            d = self.meta.directionality.get(name, 1)
            if d == -1:
                x[0, j] = -x[0, j]

        return x

    def _align_batch(
        self,
        rows: Union[List[Dict[str, float]], np.ndarray],
        incoming_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Batch version:
        - If `rows` is list[dict]: align per row.
        - If ndarray: requires `incoming_names`.
        """
        target_names = self.meta.feature_names
        X = np.zeros((len(rows), len(target_names)), dtype=np.float32)

        if isinstance(rows, list) and (len(rows) == 0 or isinstance(rows[0], dict)):
            # list of dict rows
            for r_idx, row in enumerate(rows):
                for name, val in row.items():
                    pos = self._name_to_pos.get(name)
                    if pos is not None:
                        X[r_idx, pos] = float(val)
        else:
            # ndarray path
            if not isinstance(rows, np.ndarray):
                rows = np.asarray(rows, dtype=np.float32)
            if incoming_names is None:
                raise ValueError("incoming_names is required when passing a numeric matrix.")
            name_to_in = {n: i for i, n in enumerate(incoming_names)}
            for j, name in enumerate(target_names):
                i = name_to_in.get(name)
                if i is not None:
                    X[:, j] = rows[:, i]

        # Directionality
        for j, name in enumerate(target_names):
            d = self.meta.directionality.get(name, 1)
            if d == -1:
                X[:, j] = -X[:, j]
        return X

    # ---------- Scoring API ----------

    def score_one(
        self,
        features: Union[Dict[str, float], Sequence[float]],
        incoming_names: Optional[Sequence[str]] = None,
    ) -> float:
        """
        Returns P(positive) as float in [0, 1].
        """
        X = self._align_and_direction_correct(features, incoming_names)
        p = self.model.predict_proba(X)[0, 1]
        return float(p)

    def score_batch(
        self,
        rows: Union[List[Dict[str, float]], np.ndarray],
        incoming_names: Optional[Sequence[str]] = None,
        return_labels: bool = False,
        threshold: float = 0.5,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns array of P(positive). If return_labels, also returns 0/1 predictions.
        """
        X = self._align_batch(rows, incoming_names)
        P = self.model.predict_proba(X)[:, 1]
        if return_labels:
            yhat = (P >= threshold).astype(np.int32)
            return P, yhat
        return P

    # ---------- Explanations (coef x standardized value) ----------

    def explain_one(
        self,
        features: Union[Dict[str, float], Sequence[float]],
        incoming_names: Optional[Sequence[str]] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Returns list of (feature_name, contribution) sorted by |contribution| desc.
        Contribution is coef * standardized_value (post-directionality).
        """
        if self._scaler is None:
            # If you ever swap out the pipeline, we degrade gracefully.
            return []

        X = self._align_and_direction_correct(features, incoming_names)  # 1 x D
        # Standardize using the scaler stats
        mu = getattr(self._scaler, "mean_", None)
        sc = getattr(self._scaler, "scale_", None)
        if mu is None or sc is None:
            return []

        Z = (X - mu) / (sc + 1e-12)  # 1 x D
        coef = self._clf.coef_.reshape(-1)  # (D,)
        contrib = (Z.reshape(-1) * coef)     # (D,)

        names = self.meta.feature_names
        pairs = list(zip(names, contrib.tolist()))
        pairs.sort(key=lambda t: abs(t[1]), reverse=True)
        return pairs[:top_k]

    # ---------- Introspection ----------

    def info(self) -> Dict[str, Any]:
        return {
            "n_features": len(self.meta.feature_names),
            "feature_names": self.meta.feature_names,
            "core_only": self.meta.core_only,
            "locked_features_path": self.meta.locked_features_path,
            "cv_summary": self.meta.cv_summary,
            "holdout_summary": self.meta.holdout_summary,
        }


# -------------------- CLI --------------------

def _read_jsonl(path: Path) -> List[Dict[str, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _read_csv(path: Path) -> Tuple[List[Dict[str, float]], List[str]]:
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        names = r.fieldnames or []
        rows = []
        for row in r:
            rows.append({k: float(row[k]) if row[k] != "" else 0.0 for k in names})
    return rows, names

def _write_scores(path: Path, probs: np.ndarray, labels: Optional[np.ndarray] = None):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if labels is None:
            w.writerow(["score"])
            for p in probs:
                w.writerow([f"{float(p):.6f}"])
        else:
            w.writerow(["score", "label"])
            for p, y in zip(probs, labels):
                w.writerow([f"{float(p):.6f}", int(y)])


def main_cli():
    """
    Usage:
      python -m stephanie.components.critic.model.tiny_model --in data.jsonl --out scores.csv
      python -m stephanie.components.critic.model.tiny_model --in data.csv --csv --out scores.csv --labels
      python -m stephanie.components.critic.model.tiny_model --explain row.json
    """
    import argparse
    p = argparse.ArgumentParser("TinyModel inference")
    p.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--meta",  type=str, default=str(DEFAULT_META_PATH))
    p.add_argument("--in", dest="in_path", type=str, help="JSONL (default) or CSV if --csv set")
    p.add_argument("--csv", action="store_true", help="Treat input as CSV")
    p.add_argument("--out", type=str, default="", help="Write scores to CSV")
    p.add_argument("--labels", action="store_true", help="Also emit 0/1 labels with threshold 0.5")
    p.add_argument("--explain", type=str, default="", help="Explain a single JSON row instead of batch scoring")
    args = p.parse_args()

    model = CriticModel.load(args.model, args.meta)

    if args.explain:
        row = json.loads(Path(args.explain).read_text(encoding="utf-8"))
        top = model.explain_one(row, top_k=15)
        print(json.dumps({
            "meta": model.info(),
            "explanation": [{"feature": n, "contribution": v} for n, v in top]
        }, indent=2))
        return

    if not args.in_path:
        print("Missing --in path", file=sys.stderr)
        sys.exit(2)

    in_path = Path(args.in_path)
    if args.csv:
        rows, names = _read_csv(in_path)
        probs = model.score_batch(rows)  # dicts already aligned by name
    else:
        rows = _read_jsonl(in_path)
        probs = model.score_batch(rows)

    if args.labels:
        labels = (probs >= 0.5).astype(np.int32)
    else:
        labels = None

    if args.out:
        _write_scores(Path(args.out), probs, labels)
    else:
        # print to stdout
        if labels is None:
            for p in probs:
                print(f"{float(p):.6f}")
        else:
            for p, y in zip(probs, labels):
                print(f"{float(p):.6f},{int(y)}")


if __name__ == "__main__":
    main_cli()
