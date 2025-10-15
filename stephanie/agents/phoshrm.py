# stephanie/agents/phoshrm.py
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy
import numpy as np
import pandas as pd

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL, GOAL_TEXT, PIPELINE_RUN_ID
# ---- import your matrix builder (as given) ----
from stephanie.eval.score_matrix import build_score_matrix
from stephanie.scoring.scorable import Scorable, ScorableFactory, ScorableType
from stephanie.services.scoring_service import ScoringService
from stephanie.services.zeromodel_service import ZeroModelService

if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stephanie.zeromodel.vpm_phos import (brightness_concentration,
                                          phos_sort_pack, robust01, save_img)


# ------------------------
# Agent
# ------------------------
class PhoshrmAgent(BaseAgent):
    """
    Compare HRM vs Tiny via PHOS VPMs over ~N chat responses.
    Produces:
      - hrm_vpm_raw.png / tiny_vpm_raw.png
      - hrm_vpm_phos.png / tiny_vpm_phos.png
      - vpm_phos_diff.png
      - manifest.json (metrics + paths)
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.max_responses = int(cfg.get("max_responses", 500))
        self.tl_fracs = list(cfg.get("tl_fracs", [0.25, 0.16, 0.36]))
        self.delta_guard = float(cfg.get("delta_guard", 0.02))
        self.dimensions = list(cfg.get("dimensions", ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"]))
        self.out_dir = Path(cfg.get("out_dir", "data/vpm"))
        self.interleave = bool(cfg.get("interleave", False))





    async def run(self, context: dict) -> dict:
        results_train: Dict[str, Any] = {}
        eval_stats: Dict[str, Any] = {}

        # knobs
        test_ratio   = float(self.cfg.get("test_ratio", 0.2))
        scorer_name  = str(self.cfg.get("eval_scorer", self.trainer.model_type))  # e.g. "sicql", "mrq", "hrm", "tiny"
        seed         = int(self.cfg.get("seed", 42))

        scoring_service = self.container.get("scoring")  # ScoringService

        for dimension in self.dimensions:
            # 1) pull raw samples (your existing builder)
            pairs_by_dim = self.pair_builder.get_training_pairs_by_dimension(dimension=dimension)
            samples_full = pairs_by_dim.get(dimension, [])
            if not samples_full:
                self.logger.log("NoSamplesFound", {"dimension": dimension})
                continue

            # 2) split train/test on raw samples so evaluation uses held-out
            train_samples, test_samples = _train_test_split(samples_full, test_ratio=test_ratio, seed=seed)
            if not train_samples:
                self.logger.log("NoTrainAfterSplit", {"dimension": dimension})
                continue

            # 3) train
            stats = self.trainer.train(train_samples, dimension)
            if "error" in stats:
                self.logger.log("TrainError", {"dimension": dimension, "reason": stats.get("error")})
                continue
            results_train[dimension] = stats

            # 4) build held-out triples for evaluation
            triples = _flatten_samples_for_eval(test_samples)
            if not triples:
                self.logger.log("NoEvalTriples", {"dimension": dimension})
                continue

            # 5) score with the *production path* (scoring service -> scorer -> model files you just saved)
            preds, targs = [], []
            for goal_text, out_text, target in triples:
                scorable = ScorableFactory.from_text(out_text, target_type=ScorableType.DOCUMENT)
                ctx = {GOAL: {GOAL_TEXT: goal_text}}
                try:
                    bundle = scoring_service.score(scorer_name, scorable=scorable, context=ctx, dimensions=[dimension])
                    sr = bundle.results.get(dimension)
                    if sr is None: 
                        continue
                    preds.append(float(sr.score))
                    targs.append(float(target))
                except Exception as e:
                    self.logger.log("EvalScoreError", {"dimension": dimension, "error": str(e)})

            if len(preds) < 2:
                self.logger.log("EvalTooSmall", {"dimension": dimension, "count": len(preds)})
                continue

            # 6) compute metrics
            y_pred = np.array(preds, dtype=np.float64)
            y_true = np.array(targs, dtype=np.float64)

            mae  = float(np.mean(np.abs(y_pred - y_true)))
            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            # optional R^2 (guard degenerate)
            if np.var(y_true) > 1e-12:
                r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            else:
                r2 = None
            corrs = _safe_corr(y_pred, y_true)

            eval_stats[dimension] = {
                "n": int(len(y_true)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                **corrs,
                "scorer": scorer_name,
            }

            # 7) persist to TrainingStats (optional but nice)
            try:
                self.memory.training_stats.add_from_result(
                    stats={"avg_loss": mae, "train_mae": stats.get("train_mae"), "eval_rmse": rmse,
                           "eval_r2": r2, "eval_pearson_r": corrs["pearson_r"], "eval_spearman_rho": corrs["spearman_rho"]},
                    model_type=self.trainer.model_type,
                    target_type=self.trainer.target_type,
                    dimension=dimension,
                    version=self.trainer.version,
                    embedding_type=self.trainer.embedding_type,
                    config={"test_ratio": test_ratio, "eval_scorer": scorer_name},
                    sample_count=len(samples_full),
                    valid_samples=len(train_samples),
                    invalid_samples=len(samples_full) - len(train_samples),
                )
            except Exception as e:
                self.logger.log("TrainingStatsPersistError", {"dimension": dimension, "error": str(e)})

        # return both train and eval summaries
        context["training_stats"] = results_train
        context["eval_stats"] = eval_stats
        return context



def _flatten_samples_for_eval(samples: List[dict]) -> List[Tuple[str, str, float]]:
    """
    Normalize various sample schemas into (goal_text, output_text, target_value).
    Supports:
      - {"title"/"goal_text", "output", "score"}
      - {"title"/"goal_text", "output_a"/"output_b", "value_a"/"value_b"}
      - {"goal_text", "scorable_text", "target_score" or "score"}
    """
    triples = []
    for s in samples:
        title = (s.get("goal_text") or s.get("title") or "").strip()
        # singleton
        if "output" in s and ("score" in s or "target_score" in s):
            out = (s.get("output") or "").strip()
            val = s.get("target_score", s.get("score"))
            if title and out and val is not None:
                triples.append((title, out, float(val)))
            continue
        # pairwise
        if all(k in s for k in ("output_a","output_b","value_a","value_b")):
            a_out, b_out = (s.get("output_a") or "").strip(), (s.get("output_b") or "").strip()
            a_val, b_val = s.get("value_a", None), s.get("value_b", None)
            if title and a_out and a_val is not None:
                triples.append((title, a_out, float(a_val)))
            if title and b_out and b_val is not None:
                triples.append((title, b_out, float(b_val)))
            continue
        # explicit HRM/MRQ form
        if ("goal_text" in s and "scorable_text" in s and ("target_score" in s or "score" in s)):
            out = (s.get("scorable_text") or "").strip()
            val = s.get("target_score", s.get("score"))
            if title and out and val is not None:
                triples.append((title, out, float(val)))
            continue
    return triples

def _train_test_split(xs: List[dict], test_ratio: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    if not xs: return [], []
    rnd = random.Random(seed)
    idx = list(range(len(xs)))
    rnd.shuffle(idx)
    cut = max(1, int(len(idx) * (1 - test_ratio)))
    train_ix, test_ix = set(idx[:cut]), set(idx[cut:])
    return [xs[i] for i in train_ix], [xs[i] for i in test_ix]

def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict:
    """Returns Pearson r and Spearman rho (rank corr) without scipy."""
    out = {"pearson_r": None, "spearman_rho": None}
    if len(x) < 2: return out
    # Pearson
    try:
        r = float(np.corrcoef(x, y)[0,1])
        if math.isfinite(r): out["pearson_r"] = r
    except Exception: pass
    # Spearman via rank corr
    try:
        rx = x.argsort().argsort().astype(np.float64)
        ry = y.argsort().argsort().astype(np.float64)
        rho = float(np.corrcoef(rx, ry)[0,1])
        if math.isfinite(rho): out["spearman_rho"] = rho
    except Exception: pass
    return out



    # ---------------- Private: collect candidate responses ----------------
    def _collect_texts(self, goal_text: str, *, top_k: int) -> List[str]:
        """
        Pull a blend of 'good', 'medium', 'opposite' responses using your embedding store.
        """
        texts: List[str] = []

        # Good: previously scored on the same goal
        try:
            good_rows = self._gather_runs_by_goal(goal_text, limit=top_k // 3)
            texts.extend(good_rows)
        except Exception as e:
            self.logger.log("PHOSGoodGatherError", {"error": str(e)})

        # Medium: similarity band
        try:
            med_rows = self.memory.embedding.search_scorables_in_similarity_band(
                goal_text, ScorableType.RESPONSE, 0.15, 0.80, top_k // 3
            )
            texts.extend([r["text"] for r in med_rows])
        except Exception as e:
            self.logger.log("PHOSMediumGatherError", {"error": str(e)})

        # Opposite: unrelated
        try:
            opp_rows = self.memory.embedding.search_unrelated_scorables(
                goal_text, ScorableType.RESPONSE, top_k=top_k // 3
            )
            texts.extend([r["text"] for r in opp_rows])
        except Exception as e:
            self.logger.log("PHOSOppositeGatherError", {"error": str(e)})

        # Dedup & cap
        seen = set()
        uniq = []
        for t in texts:
            if t and t not in seen:
                seen.add(t); uniq.append(t)
                if len(uniq) >= top_k: break
        return uniq

    def _gather_runs_by_goal(self, goal_text: str, limit: int) -> List[str]:
        """
        Heuristic: pull responses that were previously linked to goals similar to goal_text.
        If you have a DB table for successful runs, you can query it here. For now, use
        nearest neighbors from embedding store tagged as RESPONSE with a higher band.
        """
        rows = self.memory.embedding.search_scorables_in_similarity_band(
            goal_text, ScorableType.RESPONSE, 0.70, 1.00, limit
        )
        return [r["text"] for r in rows]


    # ---------------- Private: VPM build & compare ------------------------
    def _build_and_compare_vpms(
        self,
        *,
        df: pd.DataFrame,
        dimensions: List[str],
        out_prefix: str,
        tl_fracs: Iterable[float],
        interleave: bool,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        artifacts: Dict[str, str] = {}
        metrics: Dict[str, Any] = {}

        # Raw vectors → raw images
        hrm_vec = vpm_vector_from_df(df, model="hrm", dimensions=dimensions, interleave=interleave)
        tiny_vec = vpm_vector_from_df(df, model="tiny", dimensions=dimensions, interleave=interleave)

        hrm_raw = hrm_vec.reshape(int(np.ceil(np.sqrt(len(hrm_vec)))),-1) if hrm_vec.size else np.zeros((1,1))
        tiny_raw = tiny_vec.reshape(int(np.ceil(np.sqrt(len(tiny_vec)))),-1) if tiny_vec.size else np.zeros((1,1))

        save_img(hrm_raw, f"{out_prefix}_hrm_vpm_raw.png", title="HRM VPM (raw)")
        save_img(tiny_raw, f"{out_prefix}_tiny_vpm_raw.png", title="Tiny VPM (raw)")
        artifacts["hrm_vpm_raw"] = f"{out_prefix}_hrm_vpm_raw.png"
        artifacts["tiny_vpm_raw"] = f"{out_prefix}_tiny_vpm_raw.png"

        # Sweep PHOS tl_fracs; pick best concentration per model
        best = {}
        for name, vec in [("hrm", hrm_vec), ("tiny", tiny_vec)]:
            raw_img = hrm_raw if name == "hrm" else tiny_raw
            raw_c = brightness_concentration(raw_img, frac=0.25)

            sweeps = []
            for tl in tl_fracs:
                img = phos_sort_pack(vec, tl_frac=tl)
                c = brightness_concentration(img, frac=0.25)
                path = f"{out_prefix}_{name}_vpm_phos_tl{tl:.2f}.png"
                save_img(img, path, title=f"{name.upper()} PHOS tl={tl:.2f}")
                sweeps.append({"tl_frac": float(tl), "conc": float(c), "path": path})

            sweeps.sort(key=lambda d: d["conc"], reverse=True)
            chosen = next((s for s in sweeps if s["conc"] > raw_c * (1.0 + self.delta_guard)), sweeps[0])

            artifacts[f"{name}_vpm_phos"] = chosen["path"]
            metrics[f"{name}_raw_conc"] = float(raw_c)
            metrics[f"{name}_phos_conc"] = float(chosen["conc"])
            metrics[f"{name}_phos_tl_frac"] = float(chosen["tl_frac"])
            metrics[f"{name}_sweep"] = sweeps
            best[name] = chosen

        # Chosen diff (only for a common tl_frac if you want apples-to-apples; here we recompute both at their chosen tl)
        hrm_img = plt_imread_gray(artifacts["hrm_vpm_phos"])
        tiny_img = plt_imread_gray(artifacts["tiny_vpm_phos"])
        # align shapes
        s = min(hrm_img.shape[0], tiny_img.shape[0])
        diff = hrm_img[:s,:s] - tiny_img[:s,:s]
        dmin, dmax = float(diff.min()), float(diff.max())
        # normalize to [0,1] just for visualization
        vis = (diff - dmin) / (dmax - dmin + 1e-12)
        diff_path = f"{out_prefix}_vpm_phos_diff.png"
        save_img(vis, diff_path, title="PHOS(HRM) − PHOS(Tiny)")
        artifacts["vpm_phos_diff"] = diff_path
        metrics["diff_min"] = dmin
        metrics["diff_max"] = dmax

        return artifacts, metrics


# ---- tiny helper to read grayscale back for diff ----
def plt_imread_gray(path: str) -> np.ndarray:
    import matplotlib.image as mpimg
    img = mpimg.imread(path)
    if img.ndim == 3:  # RGB(A) → gray
        return img[..., :3].mean(axis=-1).astype(np.float64)
    return img.astype(np.float64)

# lightweight aligned scorer using your inline worker
async def build_aligned_score_df(responses, goal_text, scorer_service, dimensions):
    # build per-response dict: {"hrm.reasoning": score, ...}
    rows = []
    for i, txt in enumerate(responses):
        scorable = ScorableFactory.from_text(txt, ScorableType.RESPONSE)
        bundle_hrm = scorer_service.score("hrm", scorable, { "goal": {"goal_text": goal_text} }, dimensions)
        bundle_tiny = scorer_service.score("tiny", scorable, { "goal": {"goal_text": goal_text} }, dimensions)
        row = {}
        for d in dimensions:
            if d in bundle_hrm.results:  row[("hrm", d)]  = float(bundle_hrm.results[d].score)
            if d in bundle_tiny.results: row[("tiny", d)] = float(bundle_tiny.results[d].score)
        rows.append(row)

    import pandas as pd
    cols = pd.MultiIndex.from_product([["hrm","tiny"], dimensions], names=["model","dimension"])
    df = pd.DataFrame(rows).reindex(columns=cols)
    # drop rows with any missing to keep shapes identical
    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    return df

