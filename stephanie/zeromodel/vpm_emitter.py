# stephanie/zero_model/vpm_emitter.py
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    # Defer importing pyplot until used (helps in headless envs)
    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # We'll guard where needed


# Optional import (not required here, but kept for type hints)
try:
    from stephanie.services.zero_model import ZeroModelService
except Exception:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from stephanie.services.zero_model import ZeroModelService
    else:
        ZeroModelService = None  # type: ignore


# -------------------------
# Metrics container
# -------------------------
@dataclass
class VPMMetrics:
    """Standardized VPM metrics structure for consistent visualization."""
    coverage: float = 0.0
    correctness: float = 0.0
    coherence: float = 0.0
    citation_support: float = 0.0
    entity_consistency: float = 0.0
    readability: float = 0.0
    novelty: float = 0.0
    stickiness: float = 0.0
    tests_pass_rate: float = 0.0
    mutation_score: float = 0.0
    complexity: float = 0.0
    type_safe: float = 0.0
    lint_clean: float = 0.0
    faithfulness: float = 0.0
    overall: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> VPMMetrics:
        """Create VPMMetrics from a dictionary of metrics."""
        # Normalize common aliases
        data = dict(data or {})
        if "coverage" not in data and "claim_coverage" in data:
            data["coverage"] = data["claim_coverage"]
        if "no_halluc" in data and "hallucination_rate" not in data:
            # Leave as-is; packing handles it
            pass
        return cls(
            coverage=data.get("coverage", 0.0),
            correctness=data.get("correctness", 0.0),
            coherence=data.get("coherence", 0.0),
            citation_support=data.get("citation_support", 0.0),
            entity_consistency=data.get("entity_consistency", 0.0),
            readability=data.get("readability", 0.0),
            novelty=data.get("novelty", 0.0),
            stickiness=data.get("stickiness", 0.0),
            tests_pass_rate=data.get("tests_pass_rate", 0.0),
            mutation_score=data.get("mutation_score", 0.0),
            complexity=data.get("complexity", 0.0),
            type_safe=data.get("type_safe", 0.0),
            lint_clean=data.get("lint_clean", 0.0),
            faithfulness=data.get("faithfulness", 0.0),
            overall=data.get("overall", 0.0),
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "coverage": float(self.coverage),
            "correctness": float(self.correctness),
            "coherence": float(self.coherence),
            "citation_support": float(self.citation_support),
            "entity_consistency": float(self.entity_consistency),
            "readability": float(self.readability),
            "novelty": float(self.novelty),
            "stickiness": float(self.stickiness),
            "tests_pass_rate": float(self.tests_pass_rate),
            "mutation_score": float(self.mutation_score),
            "complexity": float(self.complexity),
            "type_safe": float(self.type_safe),
            "lint_clean": float(self.lint_clean),
            "faithfulness": float(self.faithfulness),
            "overall": float(self.overall),
        }


# -------------------------
# VPM Emitter
# -------------------------
class VPMEmitter:
    """
    VPM (Visual Progress Map) Emitter: Generates visualizations of AI processing steps.

    - Uses ZeroModel service for high-fidelity tiles if available
    - Falls back to matplotlib for PNG generation
    - Normalizes metrics across domains (text/code/image)
    - Outputs:
        * ABC tile (A/B/C comparison)
        * Iteration timeline
        * PACS panel heatmap
        * Knowledge progress (claim/evidence curves)
    """

    def __init__(
        self,
        logger: logging.Logger,
        zero_model_service: Optional[ZeroModelService] = None,
        output_dir: str = "reports/vpm",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.logger = logger
        self.zm = zero_model_service
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = {
            "default_metrics": [
                "overall",
                "coverage",
                "faithfulness",
                "coherence",
                "structure",
                "no_halluc",     # derived from hallucination_rate when packing
                "figure_ground", # derived if present
            ],
            "panel_metrics": [
                "overall",
                "coverage",
                "faithfulness",
                "structure",
                "no_halluc",
                "figure_ground",
            ],
            "thresholds": {"good": 0.75, "medium": 0.5, "bad": 0.25},
        }
        if config:
            # shallow merge for convenience
            self.config.update(config)

        # Use the project's structured logger if available
        if hasattr(self.logger, "log"):
            self.logger.log("VPMEmitterInit", {
                "output_dir": str(self.output_dir),
                "zero_model_available": bool(self.zm),
            })
        else:
            self.logger.info("VPMEmitter initialized at %s (ZeroModel=%s)",
                             str(self.output_dir), bool(self.zm))

    # ---------- public API ----------

    def emit_abc_tile(
        self,
        doc_id: str,
        metrics_a: Dict[str, float],
        metrics_b: Dict[str, float],
        metrics_c: Dict[str, float],
        title: str = "A/B/C Comparison",
    ) -> Optional[str]:
        """Emit a tile comparing metrics A, B, and C."""
        try:
            data = {
                "doc_id": str(doc_id),
                "title": title,
                "metrics": {
                    "A": self._pack(metrics_a),
                    "B": self._pack(metrics_b),
                    "C": self._pack(metrics_c),
                },
                "iterations": [],
                "timestamp": time.time(),
            }

            # Try ZeroModel service first
            if self.zm and hasattr(self.zm, "generate_summary_vpm_tiles"):
                payload = {"vpm_data": data, "output_dir": str(self.output_dir)}
                result = self.zm.generate_summary_vpm_tiles(**payload) or {}
                tile_path = result.get("quality_tile_path")
                if tile_path:
                    self._elog("VPMEmitABCComplete", {"doc_id": doc_id, "tile_path": tile_path, "service": "zero_model"})
                    return tile_path

            # Fallback
            return self._matplotlib_abc_tile(doc_id, data["metrics"]["A"], data["metrics"]["B"], data["metrics"]["C"], title)

        except Exception as e:
            self._elog("VPMEmitABCTileError", {"doc_id": doc_id, "error": str(e)})
            return None

    def emit_iteration_timeline(
        self,
        doc_id: str,
        iterations: List[Dict[str, Any]],
        title: str = "Iteration Progress",
    ) -> Optional[str]:
        """Emit a timeline showing overall score progress across iterations."""
        try:
            # ZeroModel (optional)
            if self.zm and hasattr(self.zm, "generate_summary_vpm_tiles"):
                data = {
                    "doc_id": str(doc_id),
                    "title": title,
                    "metrics": {"A": {}, "B": {}, "C": {}},
                    "iterations": iterations or [],
                    "timestamp": time.time(),
                }
                result = self.zm.generate_summary_vpm_tiles(vpm_data=data, output_dir=str(self.output_dir)) or {}
                timeline_path = result.get("iter_timeline")
                if timeline_path:
                    self._elog("VPMEmitIterTimelineComplete", {"doc_id": doc_id, "timeline_path": timeline_path, "service": "zero_model"})
                    return timeline_path

            # Fallback
            return self._matplotlib_iteration_line(doc_id, iterations, title)

        except Exception as e:
            self._elog("VPMEmitIterTimelineError", {"doc_id": doc_id, "error": str(e)})
            return None

    def emit_panel_heatmap(
        self,
        doc_id: str,
        panel_detail: Dict[str, Any],
        title: str = "PACS Panel",
    ) -> Optional[str]:
        """Emit a heatmap for PACS panel outputs (skeptic/editor/risk)."""
        try:
            # ZeroModel (optional)
            if self.zm and hasattr(self.zm, "generate_summary_vpm_tiles"):
                data = {
                    "doc_id": str(doc_id),
                    "title": title,
                    "metrics": {"A": {}, "B": {}, "C": {}},
                    "iterations": [],
                    "panel_detail": panel_detail or {},
                    "timestamp": time.time(),
                }
                result = self.zm.generate_summary_vpm_tiles(vpm_data=data, output_dir=str(self.output_dir)) or {}
                heatmap_path = result.get("panel_heatmap")
                if heatmap_path:
                    self._elog("VPMEmitPanelHeatmapComplete", {"doc_id": doc_id, "heatmap_path": heatmap_path, "service": "zero_model"})
                    return heatmap_path

            # Fallback
            return self._matplotlib_panel_heatmap(doc_id, panel_detail, title)

        except Exception as e:
            self._elog("VPMEmitPanelHeatmapError", {"doc_id": doc_id, "error": str(e)})
            return None

    def emit_knowledge_progress(
        self,
        doc_id: str,
        iterations: List[Dict[str, Any]],
        title: str = "Knowledge Progress",
    ) -> Optional[str]:
        """
        Emit knowledge progression curves (claim coverage, evidence strength).
        Falls back to whatever keys are available in iterations.
        """
        try:
            # ZeroModel (optional)
            if self.zm and hasattr(self.zm, "generate_summary_vpm_tiles"):
                data = {
                    "doc_id": str(doc_id),
                    "title": title,
                    "metrics": {"A": {}, "B": {}, "C": {}},
                    "iterations": iterations or [],
                    "timestamp": time.time(),
                    "knowledge_progress": True,
                }
                result = self.zm.generate_summary_vpm_tiles(vpm_data=data, output_dir=str(self.output_dir)) or {}
                progress_path = result.get("knowledge_progress")
                if progress_path:
                    self._elog("VPMEmitKnowledgeProgressComplete", {"doc_id": doc_id, "progress_path": progress_path, "service": "zero_model"})
                    return progress_path

            # Fallback
            return self._matplotlib_knowledge_progress(doc_id, iterations, title)

        except Exception as e:
            self._elog("VPMEmitKnowledgeProgressError", {"doc_id": doc_id, "error": str(e)})
            return None

    # ---------- helpers ----------

    def _elog(self, event: str, payload: Dict[str, Any]):
        if hasattr(self.logger, "log"):
            self.logger.log(event, payload)
        else:
            self.logger.info("%s: %s", event, json.dumps(payload))

    def _pack(self, m: Dict[str, Any]) -> Dict[str, float]:
        """
        Normalize metric keys to the canonical set used by tiles:
        - coverage: prefer 'claim_coverage' if present
        - no_halluc: derived from 'hallucination_rate', else 0/1 if present
        - figure_ground: nested 'figure_results.overall_figure_score'
        """
        m = dict(m or {})
        coverage = float(m.get("coverage", m.get("claim_coverage", 0.0)))
        faithfulness = float(m.get("faithfulness", 0.0))
        structure = float(m.get("structure", 0.0))
        overall = float(m.get("overall", 0.0))

        # hallucination
        if "hallucination_rate" in m:
            no_halluc = float(1.0 - float(m.get("hallucination_rate", 1.0)))
        else:
            no_halluc = float(m.get("no_halluc", 0.0))

        # figure grounding nested metric
        fig = 0.0
        fr = m.get("figure_results", {})
        if isinstance(fr, dict):
            fig = float(fr.get("overall_figure_score", 0.0))

        return {
            "overall": overall,
            "coverage": coverage,
            "faithfulness": faithfulness,
            "structure": structure,
            "no_halluc": no_halluc,
            "figure_ground": fig,
        }

    # ---------- matplotlib fallbacks ----------

    def _matplotlib_abc_tile(
        self,
        doc_id: str,
        A: Dict[str, float],
        B: Dict[str, float],
        C: Dict[str, float],
        title: str,
    ) -> Optional[str]:
        if plt is None:
            self._elog("MatplotlibMissing", {"for": "abc_tile"})
            return None

        names = self.config["default_metrics"]
        # build matrix rows A/B/C
        mat = np.array(
            [
                [A.get(k, 0.0) for k in names],
                [B.get(k, 0.0) for k in names],
                [C.get(k, 0.0) for k in names],
            ],
            dtype=np.float32,
        )

        fig, ax = plt.subplots(figsize=(9.2, 3.0))
        im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_yticks([0, 1, 2], labels=["A", "B", "C"])
        ax.set_xticks(range(len(names)), labels=names, rotation=20, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out = str(self.output_dir / f"{doc_id}_abc.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def _matplotlib_iteration_line(
        self,
        doc_id: str,
        iters: List[Dict[str, Any]],
        title: str,
    ) -> Optional[str]:
        if plt is None:
            self._elog("MatplotlibMissing", {"for": "iteration_timeline"})
            return None
        if not iters:
            return None

        xs = [int(i.get("iteration", idx + 1)) for idx, i in enumerate(iters)]
        current_scores = [float(i.get("current_score", 0.0)) for i in iters]
        cand_scores = [float(i.get("best_candidate_score", 0.0)) for i in iters]

        fig, ax = plt.subplots(figsize=(9.2, 4.0))
        ax.plot(xs, current_scores, linewidth=2, label="current score")
        ax.plot(xs, cand_scores, linewidth=2, label="candidate score")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Overall")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        out = str(self.output_dir / f"{doc_id}_iteration.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def _matplotlib_panel_heatmap(
        self,
        doc_id: str,
        panel_detail: Dict[str, Any],
        title: str,
    ) -> Optional[str]:
        if plt is None:
            self._elog("MatplotlibMissing", {"for": "panel_heatmap"})
            return None

        panel = (panel_detail or {}).get("panel") or []
        if not panel:
            return None

        roles = [p.get("role", "?") for p in panel]
        metrics = self.config["panel_metrics"]

        # Assemble matrix of normalized [0..1] values per role x metric
        mat = np.zeros((len(roles), len(metrics)), dtype=np.float32)
        for i, entry in enumerate(panel):
            m = entry.get("metrics", {}) or {}
            packed = self._pack(m)
            for j, key in enumerate(metrics):
                mat[i, j] = float(packed.get(key, 0.0))

        # Normalize by column (optional; packed already 0..1, but keep stable)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            cmax, cmin = float(np.max(col)), float(np.min(col))
            if cmax > cmin:
                mat[:, j] = (col - cmin) / (cmax - cmin)

        fig, ax = plt.subplots(figsize=(9.2, 3.0 + 0.25 * len(roles)))
        im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_yticks(range(len(roles)), labels=roles)
        ax.set_xticks(range(len(metrics)), labels=metrics, rotation=20, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out = str(self.output_dir / f"{doc_id}_panel.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def _matplotlib_knowledge_progress(
        self,
        doc_id: str,
        iters: List[Dict[str, Any]],
        title: str,
    ) -> Optional[str]:
        if plt is None:
            self._elog("MatplotlibMissing", {"for": "knowledge_progress"})
            return None
        if not iters:
            return None

        xs = [int(i.get("iteration", idx + 1)) for idx, i in enumerate(iters)]

        # Accept both your earlier keys and alternates
        coverage = [float(i.get("claim_coverage", i.get("coverage", 0.0))) for i in iters]
        evidence = [float(i.get("evidence_strength", i.get("citation_support", 0.0))) for i in iters]

        fig, ax = plt.subplots(figsize=(9.2, 4.0))
        ax.plot(xs, coverage, linewidth=2, label="claim coverage")
        ax.plot(xs, evidence, linewidth=2, label="evidence strength / citation")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        out = str(self.output_dir / f"{doc_id}_knowledge.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close(fig)
        return out
