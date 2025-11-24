from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score

from stephanie.components.critic.model.critic_model import CriticModel


def _group_indices(groups: np.ndarray) -> Dict[Any, np.ndarray]:
    mapping: Dict[Any, list] = {}
    for idx, g in enumerate(groups):
        mapping.setdefault(g, []).append(idx)
    return {k: np.array(v, dtype=int) for k, v in mapping.items()}


def evaluate_reranking(npz_path: str | Path, model_path: str | Path, out_dir: str | Path) -> Dict[str, Any]:
    """
    Reranking proof on NPZ dataset:

      - For each group, choose the candidate with max TinyCritic score.
      - Compare accuracy to a random-choice baseline (averaged over 100 trials).
      - Also report an oracle upper bound: proportion of groups that contain at least one positive.

    Expects NPZ with keys: X, y, metric_names, groups
      * X: (N, D) features
      * y: (N,) labels in {0,1}
      * metric_names: list/array of D feature names
      * groups: (N,) group identifiers (e.g., problem IDs)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=True) as data:
        X = data["X"]
        y = data["y"].astype(int)
        metric_names = data["metric_names"].tolist()
        groups = data.get("groups", None)

    if groups is None:
        raise ValueError("Dataset NPZ must contain 'groups' to run reranking proof.")

    # Load model
    tc = CriticModel.load(model_path)

    # Compute probabilities for all samples
    p = tc.score_batch(X, incoming_names=metric_names)

    # Grouped rerank: pick the best candidate per group according to TinyCritic
    gmap = _group_indices(groups)
    chosen_idx = []
    for gid, idxs in gmap.items():
        sub = p[idxs]
        k = idxs[np.argmax(sub)]
        chosen_idx.append(k)
    chosen_idx = np.array(chosen_idx, dtype=int)

    # Evaluate: accuracy of TinyCritic’s chosen candidates (compare predicted label vs ground truth)
    critic_pred = (p[chosen_idx] >= 0.5).astype(int)
    critic_acc = float(accuracy_score(y[chosen_idx], critic_pred))

    # Random-choice baseline (repeat to estimate expectation)
    rng = np.random.default_rng(0)
    rand_accs = []
    for _ in range(100):
        picks = np.array([rng.choice(idxs) for idxs in gmap.values()], dtype=int)
        rand_pred = (p[picks] >= 0.5).astype(int)
        rand_accs.append(accuracy_score(y[picks], rand_pred))
    rand_acc = float(np.mean(rand_accs))

    # Oracle: fraction of groups that contain at least one positive
    oracle_acc = float(np.mean([int(y[idxs].max() > 0) for idxs in gmap.values()]))

    report = {
        "dataset": str(npz_path),
        "model": str(model_path),
        "n_samples": int(X.shape[0]),
        "n_groups": int(len(gmap)),
        "critic_accuracy": critic_acc,
        "baseline_random_accuracy": rand_acc,
        "oracle_group_accuracy": oracle_acc,
        "absolute_lift_vs_random": critic_acc - rand_acc,
    }

    (out_dir / "reranking_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse, pprint
    p = argparse.ArgumentParser(description="Verify Critic utility via reranking proof")
    p.add_argument("--npz", type=str, default="data/critic.npz")
    p.add_argument("--model", type=str, default="models/critic.joblib")
    p.add_argument("--out", type=str, default="runs/critic_proof")
    args = p.parse_args()

    res = evaluate_reranking(args.npz, args.model, args.out)
    pprint.pprint(res)
    print("✅ Reranking proof completed. Report written.")
