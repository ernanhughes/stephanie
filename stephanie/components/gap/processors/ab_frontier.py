# stephanie/components/gap/processors/ab_frontier.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from stephanie.utils.embed_utils import as_list_floats, cos_safe


@dataclass
class ABFrontierConfig:
    out_dir: Path
    # If empty, we will auto-pick pairs based on variance or common names.
    metric_pairs: List[Tuple[str, str]] = None
    # Pairing threshold in cosine similarity for item matching across sets
    min_pair_sim: float = 0.35
    # Max arrows to draw (to avoid overplot)
    max_arrows: int = 600


class ABFrontierProcessor:
    """
    Builds visual 'frontier' comparisons for Baseline vs Targeted Nexus runs:

    - Pairs Targeted items to nearest Baseline items using cosine on embeddings.global
      (fallback: metrics_vector when embeddings missing)
    - Creates 2D plots (chosen metric pairs or PCA fallback) with arrows from Baseline→Targeted
    - Writes PNGs and an index.json manifest under <out_dir>
    """

    def __init__(self, cfg, container, logger):
        self.cfg = cfg
        self.container = container
        self.logger = logger

    # ---- public entrypoint ----
    def build_frontiers(
        self,
        *,
        run_id: str,
        score_out: Dict[str, Any],           # orchestrator A/B table (columns, table_a/b, items_a/b)
        bl_manifest: Dict[str, Any],
        tg_manifest: Dict[str, Any],
        out_dir: Path,
        metric_pairs: Optional[List[Tuple[str, str]]] = None,
        min_pair_sim: float = 0.35,
    ) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)

        cols = list(score_out.get("columns") or [])
        A = np.array(score_out.get("table_a") or [], dtype=float)  # Baseline
        B = np.array(score_out.get("table_b") or [], dtype=float)  # Targeted
        ids_a: List[str] = list(score_out.get("items_a") or [])
        ids_b: List[str] = list(score_out.get("items_b") or [])

        # --- Extract embeddings for pairing ---
        emb_a, emb_b = self._extract_embeds(bl_manifest), self._extract_embeds(tg_manifest)
        # fallback to metrics_vector embeddings if needed
        if not any(len(v) > 0 for v in emb_a.values()) or not any(len(v) > 0 for v in emb_b.values()):
            self.logger.log("ABFrontierWarn", {"msg": "Missing embeddings; falling back to metrics_vector"})
            emb_a = self._extract_metric_vectors(bl_manifest)
            emb_b = self._extract_metric_vectors(tg_manifest)

        # Build aligned matrices for paired items
        pairs = self._pair_items(ids_a, ids_b, emb_a, emb_b, min_pair_sim=min_pair_sim)
        if not pairs:
            self.logger.log("ABFrontierNoPairs", {"msg": "No cross-set pairs above threshold", "thr": min_pair_sim})
            return {"pairs": 0, "images": [], "index_path": None}

        # align rows to the paired id order
        idx_a = {i: k for k, i in enumerate(ids_a)}
        idx_b = {i: k for k, i in enumerate(ids_b)}
        rows_a = np.array([A[idx_a[ia]] for ia, _ in pairs], dtype=float)
        rows_b = np.array([B[idx_b[ib]] for _, ib in pairs], dtype=float)
        pair_ids = [(ia, ib) for ia, ib in pairs]

        # Decide metric pairs
        m_pairs = metric_pairs or self._choose_metric_pairs(cols, rows_a, rows_b)

        images = []
        for (mx, my) in m_pairs:
            if mx not in cols or my not in cols:
                continue
            xi, yi = cols.index(mx), cols.index(my)

            img_path = out_dir / f"frontier__{_safe_name(mx)}__{_safe_name(my)}.png"
            self._plot_frontier(
                rows_a[:, xi], rows_a[:, yi],
                rows_b[:, xi], rows_b[:, yi],
                ids_a=[ia for ia, _ in pair_ids],
                ids_b=[ib for _, ib in pair_ids],
                title=f"Frontier (Targeted vs Baseline) — {mx} vs {my}",
                out_path=img_path,
            )
            images.append(img_path.as_posix())

        # Write index.json
        index = {
            "run_id": run_id,
            "pairs": len(pairs),
            "columns": cols,
            "images": images,
            "metric_pairs": m_pairs,
        }
        (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        return {"pairs": len(pairs), "images": images, "index_path": (out_dir / "index.json").as_posix()}

    # ---- helpers ----
    def _extract_embeds(self, manifest: Dict[str, Any]) -> Dict[str, List[float]]:
        out = {}
        for it in manifest.get("items", []) or []:
            k = str(it.get("item_id") or it.get("scorable_id") or "")
            v = (it.get("embeddings") or {}).get("global")
            out[k] = as_list_floats(v)
        return out

    def _extract_metric_vectors(self, manifest: Dict[str, Any]) -> Dict[str, List[float]]:
        out = {}
        for it in manifest.get("items", []) or []:
            k = str(it.get("item_id") or it.get("scorable_id") or "")
            mv = it.get("metrics_vector") or {}
            if isinstance(mv, dict) and mv:
                out[k] = [float(x) for x in mv.values()]
            else:
                out[k] = []
        return out

    def _pair_items(
        self,
        ids_a: List[str], ids_b: List[str],
        emb_a: Dict[str, List[float]], emb_b: Dict[str, List[float]],
        *, min_pair_sim: float
    ) -> List[Tuple[str, str]]:
        """
        Greedy one-to-one matching by cosine similarity. Returns list of (id_a, id_b).
        """
        # build matrices
        A = [emb_a.get(i, []) for i in ids_a]
        B = [emb_b.get(i, []) for i in ids_b]
        if not any(A) or not any(B):
            return []

        # normalize (avoid div-by-zero; cos_safe will protect anyway)
        sims = []
        for jb, vb in enumerate(B):
            for ia, va in enumerate(A):
                s = cos_safe(va, vb)
                if s >= min_pair_sim:
                    sims.append((s, ia, jb))
        sims.sort(key=lambda t: -t[0])  # high→low

        used_a, used_b, pairs = set(), set(), []
        for s, ia, jb in sims:
            if ia in used_a or jb in used_b:
                continue
            used_a.add(ia)
            used_b.add(jb)
            pairs.append((ids_a[ia], ids_b[jb]))
        return pairs

    def _choose_metric_pairs(self, cols: List[str], A: np.ndarray, B: np.ndarray) -> List[Tuple[str, str]]:
        """
        If specific pairs not provided, pick:
        - Known nice pairs if present.
        - Else top-variance 2D combos (up to 4 plots).
        """
        preferred = [
            ("clarity", "faithfulness"),
            ("reasoning", "knowledge"),
            ("coverage", "faithfulness"),
            ("alignment", "clarity"),
        ]
        pairs = []
        for x, y in preferred:
            cx = _find_col(cols, x)
            cy = _find_col(cols, y)
            if cx and cy:
                pairs.append((cx, cy))
        if pairs:
            return pairs

        # variance heuristic
        if len(cols) < 2 or A.size == 0 or B.size == 0:
            return []
        C = np.vstack([A, B])
        var = np.var(C, axis=0)
        order = np.argsort(var)[::-1]
        out = []
        for i in range(min(4, len(order) // 2)):
            j = 2 * i
            if j + 1 < len(order):
                out.append((cols[order[j]], cols[order[j + 1]]))
        return out[:4]

    def _plot_frontier(
        self,
        ax0_a: np.ndarray, ay0_a: np.ndarray,
        ax0_b: np.ndarray, ay0_b: np.ndarray,
        *, ids_a: Sequence[str], ids_b: Sequence[str],
        title: str, out_path: Path
    ) -> None:
        """
        Scatter two clouds and draw arrows per matched pair Baseline→Targeted.
        """
        # aesthetics
        plt.figure(figsize=(8, 6), dpi=140)
        ax = plt.gca()
        ax.set_facecolor("#0f0f10")
        plt.grid(alpha=0.15, color="#666")

        # draw point clouds
        plt.scatter(ax0_a, ay0_a, s=12, alpha=0.45, label="Baseline", edgecolors="none")
        plt.scatter(ax0_b, ay0_b, s=14, alpha=0.55, label="Targeted", edgecolors="none")

        # arrows
        n = min(len(ax0_a), len(ax0_b))
        stride = max(1, n // 600)  # sparsify if too many
        for i in range(0, n, stride):
            x1, y1 = ax0_a[i], ay0_a[i]
            x2, y2 = ax0_b[i], ay0_b[i]
            plt.arrow(x1, y1, (x2 - x1), (y2 - y1), length_includes_head=True,
                      head_width=0.02 * max(1.0, (ax.get_xlim()[1]-ax.get_xlim()[0])),
                      head_length=0.02 * max(1.0, (ax.get_ylim()[1]-ax.get_ylim()[0])),
                      alpha=0.25)

        plt.title(title, color="#ddd", fontsize=12)
        plt.legend(facecolor="#141414", edgecolor="#333", labelcolor="#ddd")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path.as_posix(), facecolor="#0f0f10")
        plt.close()


# ---- small helpers ----
def _find_col(cols: List[str], needle: str) -> Optional[str]:
    nl = needle.lower()
    # prefer exact
    for c in cols:
        if c.lower() == nl:
            return c
    # else contains
    for c in cols:
        if nl in c.lower():
            return c
    return None


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)
