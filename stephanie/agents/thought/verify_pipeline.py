# stephanie/agents/thought/verify_pipeline.py
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ----- External deps from your codebase -----
from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.summary.paper_summarizer import SimplePaperBlogAgent
from stephanie.knowledge.anti_hallucination import AntiHallucination
from stephanie.knowledge.figure_grounding import FigureGrounding
from stephanie.models.strategy import StrategyProfile
from stephanie.scoring.scorable import ScorableFactory, ScorableType
from stephanie.utils.casebook_utils import generate_casebook_name
from stephanie.utils.json_sanitize import sanitize_for_json


# ==========================
# 1) Config
# ==========================
@dataclass
class VerifierConfig:
    max_iters: int = 5
    min_gain: float = 0.015
    min_overall: float = 0.80
    target_confidence: float = 0.95
    min_figure_score: float = 0.80
    verification_threshold: float = 0.90
    convergence_window: int = 2
    knowledge_graph_conf: float = 0.70
    sents_min: int = 4
    sents_max: int = 20
    cbr_cases: int = 3
    hrm_weight: float = 0.10
    use_cbr: bool = True
    use_hrm: bool = True
    use_zeromodel: bool = True
    use_descendants_metric: bool = False
    strategy_scope: str = "track_c"
    report_dir: str = "reports/track_c"
    vpm_dir: str = "reports/vpm"
    enable_audit_report: bool = True

    model_key_ranker: str = "ranker.sicql.v1"
    model_key_retriever: str = "retriever.mrq.v1"


# ==========================
# 2) Visualization (VPM) Emitter
# ==========================
class VPMEmitter:
    """
    Emits VPM images. Uses your zero_model_service if available;
    otherwise falls back to matplotlib PNG/GIFs.
    """
    def __init__(self, logger, zeromodel_service, out_dir: str = "reports/vpm"):
        self.logger = logger
        self.zm = zeromodel_service
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def emit_abc_tile(self, doc_id: str, metrics_a: dict, metrics_b: dict, metrics_c: dict) -> Optional[str]:
        try:
            if self.zm and hasattr(self.zm, "generate_summary_vpm_tiles"):
                payload = {
                    "vpm_data": {
                        "doc_id": str(doc_id),
                        "title": "",
                        "metrics": {
                            "A": self._pack(metrics_a),
                            "B": self._pack(metrics_b),
                            "C": self._pack(metrics_c),
                        },
                        "iterations": [],
                        "timestamp": time.time(),
                    },
                    "output_dir": self.out_dir,
                }
                res = self.zm.generate_summary_vpm_tiles(**payload) or {}
                return res.get("quality_tile_path")
            # fallback to simple matplotlib strip
            return self._matplotlib_abc_tile(doc_id, metrics_a, metrics_b, metrics_c)
        except Exception as e:
            self.logger.log("VPMEmitABCTileError", {"doc_id": doc_id, "error": str(e)})
            return None

    def emit_iteration_timeline(self, doc_id: str, iterations: List[Dict[str, Any]]) -> Optional[str]:
        try:
            return self._matplotlib_iteration_line(doc_id, iterations)
        except Exception as e:
            self.logger.log("VPMEmitIterTimelineError", {"doc_id": doc_id, "error": str(e)})
            return None

    def emit_panel_heatmap(self, doc_id: str, panel_detail: Dict[str, Any]) -> Optional[str]:
        """
        Heatmap rows = roles (skeptic/editor/risk), cols = key sub-metrics,
        values = normalized 0..1 for quick visual compare.
        """
        try:
            return self._matplotlib_panel_heatmap(doc_id, panel_detail or {})
        except Exception as e:
            self.logger.log("VPMEmitPanelHeatmapError", {"doc_id": doc_id, "error": str(e)})
            return None

    def emit_knowledge_progress(self, doc_id: str, iterations: List[Dict[str, Any]]) -> Optional[str]:
        """
        Line plot of knowledge_verification (or claim_coverage/evidence_strength) across iterations.
        """
        try:
            return self._matplotlib_knowledge_progress(doc_id, iterations)
        except Exception as e:
            self.logger.log("VPMEmitKnowledgeProgressError", {"doc_id": doc_id, "error": str(e)})
            return None

    # ---- helpers ----
    def _pack(self, m: dict) -> dict:
        return {
            "overall": float(m.get("overall", 0.0)),
            "coverage": float(m.get("claim_coverage", m.get("coverage", 0.0))),
            "faithfulness": float(m.get("faithfulness", 0.0)),
            "structure": float(m.get("structure", 0.0)),
            "no_halluc": float(1.0 - m.get("hallucination_rate", 1.0)),
            "figure_ground": float(
                (m.get("figure_results", {}) or {}).get("overall_figure_score", 0.0)
            ) if isinstance(m.get("figure_results"), dict) else 0.0,
        }

    def _matplotlib_abc_tile(self, doc_id: str, A: dict, B: dict, C: dict) -> Optional[str]:
        try:
            import matplotlib
            if matplotlib.get_backend().lower() != "agg":
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            self.logger.log("MatplotlibMissing", {"for": "abc_tile"})
            return None

        names = ["overall", "coverage", "faithfulness", "structure", "no_halluc", "figure_ground"]
        mat = np.array([
            [self._pack(A)[k] for k in names],
            [self._pack(B)[k] for k in names],
            [self._pack(C)[k] for k in names],
        ], dtype=np.float32)

        fig, ax = plt.subplots(figsize=(8, 2.6))
        im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_yticks([0,1,2], labels=["A", "B", "C"])
        ax.set_xticks(range(len(names)), labels=names, rotation=20, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out = os.path.join(self.out_dir, f"{doc_id}_abc.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def _matplotlib_iteration_line(self, doc_id: str, iters: List[Dict[str, Any]]) -> Optional[str]:
        if not iters:
            return None
        try:
            import matplotlib
            if matplotlib.get_backend().lower() != "agg":
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            self.logger.log("MatplotlibMissing", {"for": "iteration_timeline"})
            return None

        xs = [it["iteration"] for it in iters]
        cs = [float(it.get("current_score", 0.0)) for it in iters]
        ys = [float(it.get("best_candidate_score", 0.0)) for it in iters]

        fig, ax = plt.subplots(figsize=(8.6, 4.0))
        ax.plot(xs, cs, linewidth=2, label="current score")
        ax.plot(xs, ys, linewidth=2, label="candidate score")
        ax.set_title("Per-Iteration Scores (Track C)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Overall")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        out = os.path.join(self.out_dir, f"{doc_id}_timeline.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def _matplotlib_panel_heatmap(self, doc_id: str, panel_detail: Dict[str, Any]) -> Optional[str]:
        panel = panel_detail.get("panel") or []
        if not panel:
            return None
        try:
            import matplotlib
            if matplotlib.get_backend().lower() != "agg":
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            self.logger.log("MatplotlibMissing", {"for": "panel_heatmap"})
            return None

        roles = [p.get("role","?") for p in panel]
        cols = ["overall", "claim_coverage", "faithfulness", "structure", "hallucination_rate"]
        M = []
        for p in panel:
            m = p.get("metrics", {}) or {}
            row = [
                float(m.get("overall", 0.0)),
                float(m.get("claim_coverage", m.get("coverage", 0.0))),
                float(m.get("faithfulness", 0.0)),
                float(m.get("structure", 0.0)),
                1.0 - float(m.get("hallucination_rate", 1.0)),
            ]
            M.append(row)
        M = np.array(M, dtype=np.float32)

        fig, ax = plt.subplots(figsize=(7, 2.2 + 0.3*len(roles)))
        im = ax.imshow(M, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_yticks(range(len(roles)), labels=roles)
        ax.set_xticks(range(len(cols)), labels=cols, rotation=20, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("PACS Panel Metrics")
        out = os.path.join(self.out_dir, f"{doc_id}_panel.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def _matplotlib_knowledge_progress(self, doc_id: str, iters: List[Dict[str, Any]]) -> Optional[str]:
        try:
            import matplotlib
            if matplotlib.get_backend().lower() != "agg":
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            self.logger.log("MatplotlibMissing", {"for": "knowledge_progress"})
            return None

        xs = [it["iteration"] for it in iters]
        kv = []
        es = []
        for it in iters:
            kv.append(float(it.get("claim_coverage", it.get("knowledge_verification", 0.0))))
            es.append(float(it.get("evidence_strength", 0.0)))

        fig, ax = plt.subplots(figsize=(8.6, 4.0))
        ax.plot(xs, kv, linewidth=2, label="claim/evidence coverage")
        ax.plot(xs, es, linewidth=2, label="evidence strength")
        ax.set_title("Knowledge Verification Progress")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        out = os.path.join(self.out_dir, f"{doc_id}_knowledge.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close(fig)
        return out


# ==========================
# 3) Knowledge Graph Builder
# ==========================
class KnowledgeGraphBuilder:
    def __init__(self, container, logger):
        self.container = container
        self.logger = logger

    async def build(self, doc_id: str, paper_text: str, chat_corpus: List[dict], trajectories: List[dict], domains: List[dict]) -> Dict[str, Any]:
        def _empty() -> Dict[str, Any]:
            return {
                "nodes": [], "relationships": [], "claims": [],
                "claim_coverage": 0.0, "evidence_strength": 0.0,
                "temporal_coherence": 0.0, "domain_alignment": 0.0,
                "knowledge_gaps": [], "meta": {"paper_id": str(doc_id)}
            }

        svc = self.container.get("knowledge_graph")
        if not (svc and hasattr(svc, "build_tree")):
            self.logger.log("KGMissingBuildTree", {"doc_id": doc_id})
            return _empty()

        try:
            kg = await asyncio.to_thread(
                svc.build_tree,
                paper_text=paper_text or "",
                paper_id=str(doc_id),
                chat_corpus=chat_corpus or [],
                trajectories=trajectories or [],
                domains=domains or [],
            )
            if not isinstance(kg, dict):
                kg = {}
            kg = kg.get("knowledge_graph") or kg
            for k, v in _empty().items():
                kg.setdefault(k, v)
            kg["meta"].setdefault("paper_id", str(doc_id))
            return kg
        except Exception as e:
            self.logger.log("KnowledgeGraphBuildFailed", {"doc_id": doc_id, "error": str(e), "traceback": traceback.format_exc()})
            return _empty()


# ==========================
# 4) CBR Retriever
# ==========================
class CBRRetriever:
    def __init__(self, cbr_service, logger):
        self.cbr = cbr_service
        self.logger = logger

    def retrieve(self, goal_text: str, k: int = 3) -> List[Dict[str, Any]]:
        if not self.cbr:
            return []
        try:
            cases = self.cbr.retrieve(goal_text=goal_text, top_k=k) or []
            out = []
            for c in cases:
                out.append({
                    "title": (c.get("goal_text") or "")[:160],
                    "why_it_won": (c.get("scores", {}).get("winner_rationale") or "")[:240],
                    "patch": (c.get("lessons") or "")[:240],
                    "summary": (c.get("best_text") or c.get("summary") or "")[:400],
                })
            return out
        except Exception as e:
            self.logger.log("CBRRetrieveError", {"error": str(e)})
            return []


# ==========================
# 5) Prompt Builder
# ==========================
class PromptBuilder:
    def __init__(self, logger):
        self.logger = logger

    def build(self, *, current_summary: str, claims: List[dict], title: str,
              domain: str, kb_ctx: Dict[str, Any], sents_min: int, sents_max: int,
              case_pack: Optional[List[dict]] = None) -> str:

        claims_text = "\n".join(f"- {c.get('text','').strip()}" for c in (claims or [])[:5] if c.get("text"))
        tmpl_text = ""
        if (kb_ctx or {}).get("templates"):
            bullets = []
            for t in kb_ctx["templates"]:
                bullets.append("- " + " ".join(t.get("outline", [])[:3]))
            tmpl_text = "\n\nTemplates that worked before:\n" + "\n".join(bullets)

        hints_text = ""
        if (kb_ctx or {}).get("hints"):
            hints_text = "\n\nStrategy hints:\n" + "\n".join(f"- {h}" for h in kb_ctx["hints"])

        examples = ""
        if case_pack:
            ex_lines = []
            for ex in case_pack[:3]:
                ex_lines.append(f"- Lesson: {ex.get('patch','')}\n  Why it won: {ex.get('why_it_won','')}")
            if ex_lines:
                examples = "\n\nPrior improvements to emulate:\n" + "\n".join(ex_lines)

        return f"""
You are a verification expert checking this academic paper summary against the paper's key claims.

Title: {title}   (domain: {domain})

Key Claims:
{claims_text}{examples}{tmpl_text}{hints_text}

Current summary:
\"\"\"{current_summary}\"\"\"

Improve the summary by:
1) Ensuring all key claims are accurately represented
2) Citing figures/tables for quantitative claims when warranted
3) Removing unsupported statements 
4) Preserving clarity and neutrality

Constraints:
- Keep to {sents_min}-{sents_max} sentences
- Use ONLY facts present in the paper and allowed context
- Do not invent numbers or facts

Verified summary:
""".strip()


# ==========================
# 6) PACS Refiner
# ==========================
class PACSRefiner:
    def __init__(self, agent: BaseAgent, metrics_calc: SimplePaperBlogAgent, figure_grounding: FigureGrounding, logger):
        self.agent = agent
        self.metrics_calc = metrics_calc
        self.figure_grounding = figure_grounding
        self.logger = logger

    def refine(self, candidate: str, abstract: str, context: Dict[str, Any],
               paper_data: Dict[str, Any], knowledge_tree: Dict[str, Any],
               pacs_weights: Dict[str, float], sents_min: int, sents_max: int,
               kbase_ctx: Dict[str, Any] | None = None,
               return_panel: bool = False) -> str | Tuple[str, Dict[str, Any]]:

        roles = [
            ("skeptic", "remove speculation; flag ungrounded claims"),
            ("editor", f"tighten structure; keep {sents_min}-{sents_max} sentences"),
            ("risk",   "require figure/table citation for any numeric claim"),
        ]
        panel: List[Dict[str, Any]] = []

        for role, brief in roles:
            prompt = f"""Role: {role.title()}. Brief: {brief}
Abstract:
\"\"\"{abstract[:1000]}\"\"\"

Text to review:
\"\"\"{candidate}\"\"\"

Return ONLY the revised paragraph."""
            try:
                out = self.agent.call_llm(prompt, context=context)
                if not out:
                    continue
                text = out.strip()
                m = self.metrics_calc._compute_metrics(text, abstract, "")
                if role == "risk":
                    m["figure_results"] = self._figure_score(text, paper_data, knowledge_tree)
                panel.append({"role": role, "text": text, "metrics": m})
            except Exception as e:
                self.logger.log("PACSRoleError", {"role": role, "error": str(e)})

        if not panel:
            return (candidate, {}) if return_panel else candidate

        # choose best by role-weighted score
        best_text, best_score, best_entry = candidate, -1.0, None
        for entry in panel:
            score = self._role_weighted_score(entry["role"], entry["metrics"], pacs_weights)
            entry["score"] = score
            if score > best_score:
                best_text, best_score, best_entry = entry["text"], score, entry

        detail = {
            "panel": panel,
            "weights_used": dict(pacs_weights or {}),
            "kb_hints": (kbase_ctx or {}).get("hints", []),
            "kb_templates_count": len((kbase_ctx or {}).get("templates", [])),
        }
        return (best_text, detail) if return_panel else best_text

    def _role_weighted_score(self, role: str, m: Dict[str, float], w: Dict[str, float]) -> float:
        skeptic_focus = 0.6 * (1.0 - float(m.get("hallucination_rate", 0.0))) + 0.4 * float(m.get("faithfulness", 0.0))
        editor_focus  = 0.5 * float(m.get("coherence", 0.0)) + 0.5 * float(m.get("structure", 0.0))
        risk_focus    = float((m.get("figure_results", {}) or {}).get("overall_figure_score", 0.0)) if isinstance(m.get("figure_results"), dict) else 0.0
        base          = float(m.get("overall", 0.0))
        alpha         = w.get(role, 0.33)
        role_focus    = skeptic_focus if role == "skeptic" else editor_focus if role == "editor" else risk_focus
        return alpha * (0.5 * base + 0.5 * role_focus)

    def _figure_score(self, text: str, paper_data: Dict[str, Any], knowledge_tree: Dict[str, Any]) -> Dict[str, Any]:
        # quick heuristic (same as your prior)
        quant_claims = []
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
        for sent in sents:
            matches = re.findall(r"(\b\d+\.?\d*\s*(%|percent|accuracy|precision|recall|f1|auc|rmse|mae|bleu)\b)", sent, flags=re.I)
            if matches:
                quant_claims.append({
                    "claim": sent,
                    "value": matches[0][0],
                    "metric": matches[0][1],
                    "has_citation": any(k in sent.lower() for k in ["figure","fig.","table","as shown","see"]),
                })
        cited = sum(1 for c in quant_claims if c["has_citation"])
        rate = cited / max(1, len(quant_claims))
        return {"total_claims": len(quant_claims), "properly_cited": cited, "citation_rate": rate, "overall_figure_score": rate, "claims": quant_claims}


# ==========================
# 7) Metrics Scorer (base + knowledge + HRM)
# ==========================
class MetricsScorer:
    def __init__(self, metrics_calc: SimplePaperBlogAgent, scoring_service, logger, hrm_weight: float = 0.10):
        self.calc = metrics_calc
        self.scoring = scoring_service
        self.logger = logger
        self.hrm_weight = hrm_weight

    def score(self, summary: str, abstract: str, author_summary: str,
              knowledge_tree: Dict[str, Any], goal_title: Optional[str], context: Optional[Dict[str, Any]],
              verification_threshold: float) -> Dict[str, float]:

        base = self.calc._compute_metrics(summary, abstract, author_summary)
        ver  = self._verify_against_knowledge(summary, knowledge_tree, verification_threshold)

        hrm_score = None
        if self.scoring:
            try:
                scorable = ScorableFactory.from_dict({"text": summary, "goal": goal_title or "", "type": "document"})
                bundle = self.scoring.score("hrm", context=context, scorable=scorable, dimensions=["alignment"])
                res = getattr(bundle, "results", {}).get("alignment")
                if getattr(res, "score", None) is not None:
                    hs = float(res.score)
                    hrm_score = 1.0/(1.0+math.exp(-hs)) if hs < 0 or hs > 1 else hs
            except Exception as e:
                self.logger.log("HRMScoreError", {"error": str(e)})

        overall = 0.8 * base.get("overall", 0.0) + 0.2 * ver
        if hrm_score is not None:
            overall = (1.0 - self.hrm_weight) * overall + self.hrm_weight * hrm_score

        out = dict(base)
        out["knowledge_verification"] = float(ver)
        if hrm_score is not None:
            out["hrm_score"] = float(hrm_score)
        out["overall"] = float(overall)
        return out

    def _verify_against_knowledge(self, summary: str, tree: Dict[str, Any], threshold: float) -> float:
        if not tree:
            return 0.5
        claims = tree.get("claims", []) or []
        covered = sum(1 for c in claims if c.get("text") and self.calc._contains_concept(summary, c["text"]))
        claim_cov = covered / max(1, len(claims))
        rels = tree.get("relationships", []) or []
        strong = [r for r in rels if float(r.get("confidence", 0.0)) >= threshold]
        evidence = len(strong)/max(1, len(rels))
        return 0.7*claim_cov + 0.3*evidence


# ==========================
# 8) Guardrails
# ==========================
class Guardrails:
    def __init__(self, anti_hallucination: AntiHallucination, figure_grounding: FigureGrounding, logger):
        self.anti = anti_hallucination
        self.fig = figure_grounding
        self.logger = logger

    def hallucinations(self, summary: str, abstract: str, author_summary: str, tree: Dict[str, Any]) -> Tuple[bool, List[str]]:
        try:
            ok, issues = self.anti.verify_section(summary, tree, {"abstract": abstract, "summary": author_summary})
            return bool(ok), (issues or [])
        except Exception as e:
            self.logger.log("AntiHallucinationError", {"error": str(e)})
            return True, ["anti_hallucination_failed_soft"]


# ==========================
# 9) Strategy Manager
# ==========================
class StrategyManager:
    def __init__(self, strategy_store, agent_name: str, scope: str, logger):
        self.store = strategy_store
        self.agent_name = agent_name
        self.scope = scope
        self.logger = logger

    def load(self) -> StrategyProfile:
        if self.store:
            return self.store.load(agent_name=self.agent_name, scope=self.scope)
        return StrategyProfile()

    def save(self, profile: StrategyProfile):
        if self.store:
            self.store.save(agent_name=self.agent_name, profile=profile, scope=self.scope)


# ==========================
# 10) Persistence (casebooks + scorables + signals)
# ==========================
class Persistence:
    def __init__(self, memory, logger):
        self.memory = memory
        self.logger = logger

    def save_case_and_scorable(self, *, doc_id: str, paper_title: str, track_b_id: Any,
                               prompt_text: str, raw_llm: str, candidate: str,
                               best_summary: str, best_metrics: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, Any]:

        out: Dict[str, Any] = {}
        try:
            # ensure blog casebook (action_type='blog')
            casebook_name = generate_casebook_name("blog", paper_title)
            pipeline_run_id = context.get("pipeline_run_id")
            casebook = self.memory.casebooks.ensure_casebook(name=casebook_name, tag="blog",
                                                             pipeline_run_id=pipeline_run_id,
                                                             meta={"paper_id": str(doc_id), "title": paper_title})
            case = self.memory.casebooks.add_case(
                casebook_id=casebook.id,
                goal_id=casebook.goal_id,
                prompt_text=prompt_text,
                agent_name=context.get("agent_name") or "KnowledgeInfusedVerifier",
                response_texts=[raw_llm, candidate],
                meta={},
            )
            out["case_id"] = getattr(case, "id", None)

            # dynamic scorable
            safe_meta = sanitize_for_json({
                "paper_id": str(doc_id),
                "title": paper_title,
                "metrics": best_metrics,
                "origin": "track_c_verified",
                "origin_ids": [track_b_id],
            })
            scorable = self.memory.dynamic_scorables.add(
                pipeline_run_id=context.get("pipeline_run_id"),
                scorable_type=ScorableType.DYNAMIC,
                source=context.get("agent_name") or "KnowledgeInfusedVerifier",
                text=best_summary,
                meta=safe_meta,
                source_scorable_id=track_b_id,
                source_scorable_type="dynamic",
            )
            out["scorable_id"] = getattr(scorable, "id", None)

            # link scorable to case
            try:
                self.memory.casebooks.add_scorable(
                    case_id=out["case_id"],
                    pipeline_run_id=context.get("pipeline_run_id"),
                    role="text",
                    scorable_id=out["scorable_id"],
                    text=best_summary,
                    scorable_type=ScorableType.DYNAMIC,
                    meta={},
                )
            except Exception:
                pass

        except Exception as e:
            self.logger.log("PersistenceError", {"doc_id": doc_id, "error": str(e)})
        return out

    def capture_signal(self, *, paper_id: str, domain: str, strategy: StrategyProfile,
                       metrics: Dict[str, Any], iterations: List[Dict[str, Any]]):
        payload = {
            "paper_id": str(paper_id),
            "domain": domain,
            "strategy_version": int(getattr(strategy, "strategy_version", 0)),
            "verification_threshold": float(getattr(strategy, "verification_threshold", 0.0)),
            "pacs_weights": dict(getattr(strategy, "pacs_weights", {})),
            "final_quality": float(metrics.get("overall", 0.0)),
            "knowledge_verification": float(metrics.get("knowledge_verification", 0.0)),
            "iterations": len(iterations or []),
            "first_iter_score": float((iterations or [{}])[0].get("current_score", 0.0)) if iterations else None,
            "last_iter_score": float((iterations or [{}])[-1].get("best_candidate_score", 0.0)) if iterations else None,
            "ts": time.time(),
        }
        try:
            if hasattr(self.memory, "calibration_events"):
                self.memory.calibration_events.add({
                    "domain": domain or "general",
                    "query": f"{paper_id}:{domain}",
                    "raw_similarity": payload["final_quality"],
                    "is_relevant": bool(payload["final_quality"] >= 0.80),
                    "scorable_id": str(paper_id),
                    "scorable_type": "paper",
                    "entity_type": "summary_verification",
                    "features": {k: payload.get(k) for k in ["final_quality","knowledge_verification","iterations"]},
                })
        except Exception:
            pass
        try:
            if hasattr(self.memory, "casebooks") and hasattr(self.memory.casebooks, "add"):
                self.memory.casebooks.add(
                    casebook_name="verification_signals",
                    case_id=str(paper_id),
                    role="signal",
                    text=json.dumps(payload),
                    meta={"domain": domain, "timestamp": payload["ts"]},
                )
        except Exception:
            pass


# ==========================
# 11) Audit Reporter
# ==========================
class AuditReporter:
    def __init__(self, report_dir: str, logger):
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        self.logger = logger

    def write(self, *, doc_id: str, title: str, baseline: Dict[str, Any],
              final: Dict[str, Any], iterations: List[Dict[str, Any]],
              images: Dict[str, Optional[str]], strategy_before: Dict[str, Any], strategy_after: Dict[str, Any]) -> str:

        def f(x):
            try:
                return f"{float(x):.3f}"
            except Exception:
                return str(x)

        lines = []
        lines.append(f"# Verification Report — {title or doc_id}\n")
        if images.get("abc_tile"):
            lines.append(f"![ABC tile]({os.path.relpath(images['abc_tile'], self.report_dir)})\n")

        lines.append("## Overview (Baseline → Final)\n")
        rows = [
            ("overall", baseline.get("overall"), final.get("overall")),
            ("knowledge_verification", baseline.get("knowledge_verification"), final.get("knowledge_verification")),
            ("coverage", baseline.get("claim_coverage", baseline.get("coverage")), final.get("claim_coverage", final.get("coverage"))),
            ("faithfulness", baseline.get("faithfulness"), final.get("faithfulness")),
            ("structure", baseline.get("structure"), final.get("structure")),
            ("hallucination_rate (↓)", baseline.get("hallucination_rate"), final.get("hallucination_rate")),
            ("figure_grounding", (baseline.get("figure_results") or {}).get("overall_figure_score") if isinstance(baseline.get("figure_results"), dict) else None,
                                  (final.get("figure_results") or {}).get("overall_figure_score") if isinstance(final.get("figure_results"), dict) else None),
        ]
        lines.append("| metric | baseline | final |\n|---|---:|---:|")
        for k, b, c in rows:
            lines.append(f"| {k} | {f(b)} | {f(c)} |")
        lines.append("")

        if images.get("iter_timeline"):
            lines.append("## Iteration Timeline\n")
            lines.append(f"![Iteration scores]({os.path.relpath(images['iter_timeline'], self.report_dir)})\n")

        if images.get("panel_heatmap"):
            lines.append("## PACS Panel Snapshot\n")
            lines.append(f"![Panel]({os.path.relpath(images['panel_heatmap'], self.report_dir)})\n")

        if images.get("knowledge_progress"):
            lines.append("## Knowledge Progress\n")
            lines.append(f"![Knowledge]({os.path.relpath(images['knowledge_progress'], self.report_dir)})\n")

        lines.append("## Strategy\n")
        lines.append(f"- Before: `{json.dumps(strategy_before)}`")
        lines.append(f"- After:  `{json.dumps(strategy_after)}`\n")

        out_md = os.path.join(self.report_dir, f"{doc_id}.md")
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return out_md


# ==========================
# 12) Orchestrator (the new Agent)
# ==========================
class KnowledgeInfusedVerifier(BaseAgent):
    """
    Thin orchestrator that wires all components, runs the loop,
    and emits VPM images for each doc.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.cfg = VerifierConfig(
            max_iters=int(cfg.get("max_iters", 5)),
            min_gain=float(cfg.get("min_gain", 0.015)),
            min_overall=float(cfg.get("min_overall", 0.80)),
            target_confidence=float(cfg.get("target_confidence", 0.95)),
            min_figure_score=float(cfg.get("min_figure_score", 0.80)),
            verification_threshold=float(cfg.get("verification_threshold", 0.90)),
            convergence_window=int(cfg.get("convergence_window", 2)),
            knowledge_graph_conf=float(cfg.get("knowledge_graph_conf", 0.70)),
            sents_min=int(cfg.get("min_sents", 4)),
            sents_max=int(cfg.get("max_sents", 20)),
            cbr_cases=int(cfg.get("cbr_cases", 3)),
            hrm_weight=float(cfg.get("hrm_weight", 0.10)),
            use_cbr=bool(cfg.get("use_cbr", True)),
            use_hrm=bool(cfg.get("use_hrm", True)),
            use_zeromodel=bool(cfg.get("use_zeromodel", True)),
            use_descendants_metric=bool(cfg.get("use_descendants_metric", False)),
            strategy_scope=cfg.get("strategy_scope", "track_c"),
            report_dir=str(cfg.get("audit_report_dir", "reports/track_c")),
            vpm_dir=str(cfg.get("vpm_dir", "reports/vpm")),
            enable_audit_report=bool(cfg.get("enable_audit_report", True)),
            model_key_ranker=cfg.get("model_key_ranker", "ranker.sicql.v1"),
            model_key_retriever=cfg.get("model_key_retriever", "retriever.mrq.v1"),
        )

        # services
        self.metrics_calc = SimplePaperBlogAgent(cfg, memory, container, logger)
        self.kg_builder = KnowledgeGraphBuilder(container, logger)
        self.prompt_builder = PromptBuilder(logger)
        self.pacs_refiner = PACSRefiner(agent=self, metrics_calc=self.metrics_calc,
                                        figure_grounding=FigureGrounding(logger), logger=logger)
        self.guardrails = Guardrails(AntiHallucination(logger), FigureGrounding(logger), logger)
        self.scorer = MetricsScorer(self.metrics_calc, container.get("scoring") if self.cfg.use_hrm else None,
                                    logger, hrm_weight=self.cfg.hrm_weight)

        self.cbr = CBRRetriever(container.get("cbr") if self.cfg.use_cbr else None, logger)
        self.strategy_mgr = StrategyManager(container.get("strategy"), self.name, self.cfg.strategy_scope, logger)
        self.strategy = self.strategy_mgr.load()

        self.vpm = VPMEmitter(logger, container.get("zeromodel") if self.cfg.use_zeromodel else None, out_dir=self.cfg.vpm_dir)
        self.persist = Persistence(memory, logger)
        self.reporter = AuditReporter(self.cfg.report_dir, logger)

        self.model_key_ranker = self.cfg.model_key_ranker
        self.model_key_retriever = self.cfg.model_key_retriever

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        documents = context.get("documents", []) or context.get(self.input_key, [])
        chat_corpus = context.get("chat_corpus", [])
        out: Dict[Any, Dict[str, Any]] = {}

        for doc in documents:
            doc_id = doc.get("id") or doc.get("paper_id")
            if not doc_id:
                continue

            # Load Track A/B artifacts (optional: keep your existing getters)
            track_a, track_b = self._load_tracks(doc_id)
            if not track_b:
                self.logger.log("TrackBMissing", {"doc_id": doc_id})
                continue
            a_meta = self._safe_meta(track_a) if track_a else {}
            b_meta = self._safe_meta(track_b)
            title = doc.get("title", "") or a_meta.get("title","")
            abstract = a_meta.get("abstract") or b_meta.get("abstract") or self._fetch_abstract(doc_id)
            author_sum = a_meta.get("arxiv_summary") or b_meta.get("arxiv_summary") or (doc.get("summary","") or "")
            b_text = (getattr(track_b, "text", "") or "").strip()
            baseline = self._extract_summary(b_text) or (b_meta.get("summary") or b_text)
            baseline_metrics = b_meta.get("metrics") or self.scorer.calc._compute_metrics(baseline, abstract, author_sum)

            # Build knowledge graph
            kg = await self.kg_builder.build(
                doc_id=str(doc_id),
                paper_text=(doc.get("text") or ""),
                chat_corpus=chat_corpus,
                trajectories=context.get("conversation_trajectories", []),
                domains=context.get("domains", []),
            )

            # Iterative loop
            best_summary, best_metrics, iterations, panel_detail, prompt_used = await self._loop(
                doc_id=str(doc_id),
                title=title,
                baseline=baseline,
                abstract=abstract,
                author_summary=author_sum,
                knowledge_graph=kg,
                context=context
            )

            # Guardrails
            ok, issues = self.guardrails.hallucinations(best_summary, abstract, author_sum, kg)
            figure = self.pacs_refiner._figure_score(best_summary, doc, kg)
            passes = bool(ok) and (figure.get("overall_figure_score", 0.0) >= self.cfg.min_figure_score)
            best_metrics["figure_results"] = figure
            out_doc = {
                "summary": best_summary,
                "metrics": best_metrics,
                "iterations": iterations,
                "passes_guardrails": passes,
                "knowledge_graph": kg,
            }

            # Persist + VPM images + report
            images = {
                "abc_tile": self.vpm.emit_abc_tile(str(doc_id), a_meta.get("metrics",{}), baseline_metrics, best_metrics),
                "iter_timeline": self.vpm.emit_iteration_timeline(str(doc_id), iterations),
                "panel_heatmap": self.vpm.emit_panel_heatmap(str(doc_id), panel_detail or {}),
                "knowledge_progress": self.vpm.emit_knowledge_progress(str(doc_id), iterations),
            }

            try:
                saved = self.persist.save_case_and_scorable(
                    doc_id=str(doc_id),
                    paper_title=title,
                    track_b_id=getattr(track_b, "id", None),
                    prompt_text=prompt_used,
                    raw_llm=iterations[0].get("raw_llm","") if iterations else "",
                    candidate=iterations[0].get("candidate","") if iterations else "",
                    best_summary=best_summary,
                    best_metrics=best_metrics,
                    context={"pipeline_run_id": context.get("pipeline_run_id"),
                             "agent_name": self.name},
                )
                out_doc.update(saved)
            except Exception:
                pass

            if self.cfg.enable_audit_report:
                try:
                    report_md = self.reporter.write(
                        doc_id=str(doc_id),
                        title=title,
                        baseline=baseline_metrics,
                        final=best_metrics,
                        iterations=iterations,
                        images=images,
                        strategy_before=self.strategy.to_dict(),
                        strategy_after=self.strategy.to_dict(),  # updated below if changed
                    )
                    out_doc["report_md"] = report_md
                except Exception:
                    pass

            # Signals
            self.persist.capture_signal(
                paper_id=str(doc_id),
                domain=self._domain(context),
                strategy=self.strategy,
                metrics=best_metrics,
                iterations=iterations
            )

            out[doc_id] = out_doc

        context.setdefault("summary_v2", {})
        context["summary_v2"].update(out)
        return context

    # ----- core loop -----
    async def _loop(self, *, doc_id: str, title: str, baseline: str, abstract: str,
                    author_summary: str, knowledge_graph: Dict[str, Any], context: Dict[str, Any]) \
                    -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], str]:

        domain = self._domain(context)
        # kbase context (optional)
        kbase = self.container.get("kbase")
        kb_ctx = kbase.context_for_paper(title=title, abstract=abstract, domain=domain) if kbase else {}

        current = baseline
        current_m = self.scorer.score(current, abstract, author_summary, knowledge_graph, title, context, self.cfg.verification_threshold)
        best_s, best_m = current, current_m
        iterations: List[Dict[str, Any]] = []
        panel_detail: Dict[str, Any] = {}
        no_improve = 0

        for i in range(self.cfg.max_iters):
            case_pack = self.cbr.retrieve(goal_text=title, k=self.cfg.cbr_cases) if self.cfg.use_cbr else []
            prompt = PromptBuilder(self.logger).build(
                current_summary=current,
                claims=knowledge_graph.get("claims", []),
                title=title,
                domain=domain,
                kb_ctx=kb_ctx,
                sents_min=self.cfg.sents_min,
                sents_max=self.cfg.sents_max,
                case_pack=case_pack
            )

            raw_llm = self.call_llm(prompt, context=context) or current
            candidate, detail = self.pacs_refiner.refine(
                raw_llm, abstract, context, {"title": title}, knowledge_graph,
                pacs_weights=self.strategy.pacs_weights, sents_min=self.cfg.sents_min, sents_max=self.cfg.sents_max,
                kbase_ctx=kb_ctx, return_panel=True
            )
            panel_detail = detail or {}

            cand_m = self.scorer.score(candidate, abstract, author_summary, knowledge_graph, title, context, self.cfg.verification_threshold)
            gain = cand_m["overall"] - current_m["overall"]

            iterations.append({
                "iteration": i+1,
                "current_score": current_m["overall"],
                "best_candidate_score": cand_m["overall"],
                "gain": gain,
                "claim_coverage": knowledge_graph.get("claim_coverage", 0.0),
                "evidence_strength": knowledge_graph.get("evidence_strength", 0.0),
                "raw_llm": raw_llm,
                "candidate": candidate,
            })

            if cand_m["overall"] >= self.cfg.min_overall and gain >= self.cfg.min_gain:
                current, current_m = candidate, cand_m
                if cand_m["overall"] > best_m["overall"]:
                    best_s, best_m = current, current_m
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            if best_m["overall"] >= self.cfg.target_confidence or no_improve >= 2:
                break

        # (Optional) strategy nudge when we truly improved
        if best_m["overall"] >= baseline and (best_m["overall"] - baseline) >= self.cfg.min_gain:
            try:
                new_w = dict(self.strategy.pacs_weights)
                # tiny heuristic
                if float(best_m.get("hallucination_rate", 1.0)) > 0.2:
                    new_w["skeptic"] = min(0.4, new_w.get("skeptic", 0.33) + 0.03)
                self.strategy.update(pacs_weights=new_w, verification_threshold=min(0.99, self.strategy.verification_threshold + 0.01))
                StrategyManager(self.container.get("strategy"), self.name, self.cfg.strategy_scope, self.logger).save(self.strategy)
            except Exception:
                pass

        return best_s, best_m, iterations, panel_detail, prompt

    # ----- misc helpers -----
    def _domain(self, context: Dict[str, Any]) -> str:
        doms = context.get("domains") or []
        if doms and isinstance(doms, list):
            d = doms[0]
            return str((d.get("domain") if isinstance(d, dict) else d) or "unknown")
        return "unknown"

    def _load_tracks(self, doc_id: Any):
        a = b = None
        try:
            a = self.memory.dynamic_scorables.get_latest_by_source_pointer(
                source="paper_summarizer", source_scorable_type="document", source_scorable_id=int(doc_id)
            )
        except Exception:
            pass
        try:
            if a:
                b = self.memory.dynamic_scorables.get_latest_by_source_pointer(
                    source="sharpened_paper_summarizer", source_scorable_type="dynamic", source_scorable_id=int(a.id)
                )
        except Exception:
            pass
        return a, b

    def _safe_meta(self, obj) -> dict:
        meta = getattr(obj, "meta", {}) or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        return meta

    def _fetch_abstract(self, doc_id) -> str:
        try:
            sections = self.memory.document_sections.get_by_document(doc_id)
            for s in sections:
                sd = s.to_dict()
                if (sd.get("section_name") or "").lower().strip() == "abstract":
                    return sd.get("section_text", "") or ""
        except Exception:
            pass
        return ""

    def _extract_summary(self, text: str) -> str:
        m = re.search(r"(?mi)^##\s*Summary\s*\n(.+?)(?=^##|\Z)", text or "", re.S)
        return m.group(1).strip() if m else (text or "").strip()
