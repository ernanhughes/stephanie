# stephanie/agents/summary/knowledge_infused_summarizer.py
from __future__ import annotations

import inspect
import time
import traceback
from typing import Dict, Any, List, Tuple, Optional
import re
import numpy as np
import math
import json
from dataclasses import dataclass, field
import inspect
import asyncio
import traceback
from typing import Any, Dict, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.summary.paper_summarizer import SimplePaperSummarizerAgent
from stephanie.knowledge.anti_hallucination import AntiHallucination
from stephanie.knowledge.figure_grounding import FigureGrounding
from stephanie.agents.knowledge.knowledge_tree_builder import KnowledgeTreeBuilderAgent
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType

# -------------------- Defaults --------------------
MAX_ITERS_DEFAULT = 5
MIN_GAIN_DEFAULT = 0.015
MIN_OVERALL_DEFAULT = 0.80
TARGET_CONFIDENCE_DEFAULT = 0.95
MIN_FIGURE_SCORE_DEFAULT = 0.80
VERIFICATION_THRESHOLD_DEFAULT = 0.90
CONVERGENCE_WINDOW_DEFAULT = 2
KNOWLEDGE_GRAPH_CONF_DEFAULT = 0.70
SENTS_MIN_DEFAULT = 4
SENTS_MAX_DEFAULT = 20
CBR_CASES_DEFAULT = 3
PACS_WEIGHTS_DEFAULT = {"skeptic": 0.34, "editor": 0.33, "risk": 0.33}


@dataclass
class StrategyProfile:
    """Evolving verification strategy state persisted across runs."""
    verification_threshold: float = 0.90
    pacs_weights: Dict[str, float] = field(default_factory=lambda: PACS_WEIGHTS_DEFAULT.copy())
    strategy_version: int = 1
    last_updated: float = field(default_factory=time.time)

    def update(self, new_weights: Dict[str, float], new_threshold: float):
        self.pacs_weights = new_weights
        self.verification_threshold = new_threshold
        self.strategy_version += 1
        self.last_updated = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verification_threshold": self.verification_threshold,
            "pacs_weights": self.pacs_weights,
            "strategy_version": self.strategy_version,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyProfile":
        return cls(
            verification_threshold=float(data.get("verification_threshold", 0.90)),
            pacs_weights=dict(data.get("pacs_weights", PACS_WEIGHTS_DEFAULT.copy())),
            strategy_version=int(data.get("strategy_version", 1)),
            last_updated=float(data.get("last_updated", time.time())),
        )


class KnowledgeInfusedVerifierAgent(BaseAgent):
    """
    Track C: Knowledge-Infused Verifier with true Learning-From-Learning

    Adds:
      • CBR reuse of prior wins (patches/lessons)
      • PACS multi-critic refinement with role-aware re-ranking
      • HRM epistemic judge blended into overall score
      • ZeroModel visibility (ABC quality tile + iteration strips)
      • Strategy evolution (thresholds + PACS weights persisted)
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # --- config knobs
        self.max_iters = int(cfg.get("max_iters", MAX_ITERS_DEFAULT))
        self.min_gain = float(cfg.get("min_gain", MIN_GAIN_DEFAULT))
        self.min_overall = float(cfg.get("min_overall", MIN_OVERALL_DEFAULT))
        self.target_confidence = float(cfg.get("target_confidence", TARGET_CONFIDENCE_DEFAULT))
        self.min_figure_score = float(cfg.get("min_figure_score", MIN_FIGURE_SCORE_DEFAULT))
        self.verification_threshold = float(cfg.get("verification_threshold", VERIFICATION_THRESHOLD_DEFAULT))
        self.convergence_window = int(cfg.get("convergence_window", CONVERGENCE_WINDOW_DEFAULT))
        self.knowledge_tree_conf = float(cfg.get("knowledge_tree_conf", KNOWLEDGE_GRAPH_CONF_DEFAULT))
        self.cbr_cases = int(cfg.get("cbr_cases", CBR_CASES_DEFAULT))

        # feature flags
        self.use_cbr = bool(cfg.get("use_cbr", True))
        self.use_hrm = bool(cfg.get("use_hrm", True))
        self.use_zeromodel = bool(cfg.get("use_zeromodel", True))
        self.use_descendants_metric = bool(cfg.get("use_descendants_metric", False))
        self.hrm_weight = float(cfg.get("hrm_weight", 0.10))

        # services
        self.cbr = container.get("cbr") if self.use_cbr else None
        self.scoring = container.get("scoring")  # exposes HRM scorer if configured
        self.zero_model_service = container.get("zeromodel") if self.use_zeromodel else None

        # strategy state (persist across runs)
        self.strategy_store = container.get("strategy")  # StrategyProfileService
        self.strategy = self._load_strategy_profile()


        # dependencies
        self.metrics_calculator = SimplePaperSummarizerAgent(cfg, memory, container, logger)
        self.anti_hallucination = AntiHallucination(logger)
        self.figure_grounding = FigureGrounding(logger)

        # sentence window (align with A/B)
        self.min_sents = int(cfg.get("min_sents", SENTS_MIN_DEFAULT))
        self.max_sents = int(cfg.get("max_sents", SENTS_MAX_DEFAULT))

        # model keys
        self.model_key_ranker = cfg.get("model_key_ranker", "ranker.sicql.v1")
        self.model_key_retriever = cfg.get("model_key_retriever", "retriever.mrq.v1")

        self.logger.log("KnowledgeInfusedVerifierInit", {
            "max_iters": self.max_iters,
            "verification_threshold": self.verification_threshold,
            "convergence_window": self.convergence_window,
            "cbr_cases": self.cbr_cases,
            "use_cbr": self.use_cbr,
            "use_hrm": self.use_hrm,
            "use_zeromodel": self.use_zeromodel,
            "strategy_version": self.strategy.strategy_version,
        })

    # -------------------- strategy persistence --------------------
    def _load_strategy_profile(self) -> StrategyProfile:
        # Prefer service; never assume memory.meta exists
        if getattr(self, "strategy_store", None):
            return self.strategy_store.load(agent_name=self.name, scope="track_c")
        # ephemeral fallback (won't persist across runs)
        return StrategyProfile()

    def _save_strategy_profile(self, strategy: StrategyProfile):
        if getattr(self, "strategy_store", None):
            self.strategy_store.save(agent_name=self.name, profile=strategy, scope="track_c")
            self.strategy = strategy


    # -------------------- entrypoint --------------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.report({"event": "start", "step": "KnowledgeInfusedVerifier", "details": "Track C verification loop with learning"})

        documents = context.get("documents", []) or context.get(self.input_key, [])
        chat_corpus = context.get("chat_corpus", [])
        verified_outputs: Dict[Any, Dict[str, Any]] = {}

        def _extract_summary_from_text(text: str) -> str:
            m = re.search(r"(?mi)^##\s*Summary\s*\n(.+?)(?=^##|\Z)", text or "", re.S)
            return (m.group(1).strip() if m else (text or "").strip())

        for doc in documents:
            doc_id = doc.get("id") or doc.get("paper_id")
            if doc_id is None:
                self.logger.log("TrackCSkipNoDocId", {"doc": str(doc)[:200]})
                continue

            # --- Track A (baseline)
            try:
                track_a_obj = self.memory.dynamic_scorables.get_latest_by_source_pointer(
                    source="paper_summarizer",
                    source_scorable_type="document",
                    source_scorable_id=int(doc_id),
                )
            except Exception as e:
                track_a_obj = None
                self.logger.log("TrackALoadError", {"doc_id": doc_id, "error": str(e)})
            if not track_a_obj:
                self.logger.log("TrackAMissing", {"doc_id": doc_id, "hint": "Ensure Track A persisted with source_scorable_id=document_id and type=document"})
                continue
            a_meta = self._safe_meta(track_a_obj)
            a_metrics = a_meta.get("metrics") or {}

            # --- Track B (sharpened)
            try:
                track_b_obj = self.memory.dynamic_scorables.get_latest_by_source_pointer(
                    source="sharpened_paper_summarizer",
                    source_scorable_type="dynamic",
                    source_scorable_id=int(track_a_obj.id),
                )
            except Exception as e:
                track_b_obj = None
                self.logger.log("TrackBLoadError", {"doc_id": doc_id, "error": str(e)})
            if not track_b_obj:
                self.logger.log("TrackBMissing", {"doc_id": doc_id, "hint": "Ensure Track B persisted with source_scorable_id=<Track A dynamic id> and type=dynamic"})
                continue
            b_meta = self._safe_meta(track_b_obj)

            b_text = (getattr(track_b_obj, "text", "") or "").strip()
            baseline_b_summary = _extract_summary_from_text(b_text) or (b_meta.get("summary") or b_text)

            title = doc.get("title", "") or (a_meta.get("title") or "")
            abstract = a_meta.get("abstract") or b_meta.get("abstract") or self._fetch_abstract(doc_id)
            arxiv_summary = a_meta.get("arxiv_summary") or b_meta.get("arxiv_summary") or (doc.get("summary", "") or "")

            baseline_b_metrics = b_meta.get("metrics")
            if not baseline_b_metrics:
                baseline_b_metrics = self._score_summary(baseline_b_summary, abstract, arxiv_summary, {}, title, context)

            # --- Track C (verify + learn)
            try:
                verified = await self._verify_summary(
                    doc_id=str(doc_id),
                    enhanced_summary=baseline_b_summary,
                    paper_data=doc,
                    chat_corpus=chat_corpus,
                    context=context,
                    track_a=track_a_obj,
                    track_b=track_b_obj,
                )
            except Exception as e:
                self.logger.log("TrackCVerifyError", {"doc_id": doc_id, "error": str(e), "traceback": traceback.format_exc()})
                continue

            verified_outputs[doc_id] = verified

            # --- training events + VPM tiles
            try:
                v_metrics = verified.get("metrics") or {}
                if v_metrics.get("overall", 0.0) >= self.min_overall and verified.get("passes_guardrails", False):
                    self._emit_training_events(
                        paper={
                            "paper_id": doc.get("paper_id", doc_id),
                            "title": title,
                            "abstract": abstract,
                            "author_summary": arxiv_summary,
                        },
                        baseline_summary=baseline_b_summary,
                        verified_summary=verified.get("summary", ""),
                        baseline_metrics=baseline_b_metrics,
                        verified_metrics=v_metrics,
                        context=context,
                    )

                self._emit_vpm_tiles(
                    doc_id=doc_id,
                    title=title,
                    metrics_a=a_metrics,
                    metrics_b=baseline_b_metrics or {},
                    metrics_c=v_metrics,
                    iterations_c=verified.get("iterations", []),
                    out_dir="reports/vpm",
                    lineage_ids=[getattr(track_a_obj, "id", None), getattr(track_b_obj, "id", None)],
                )
            except Exception as e:
                try:
                    self.memory.session.rollback()
                except Exception:
                    pass
                self.logger.log("TrackCPostProcessError", {"doc_id": doc_id, "error": str(e)})

        context.setdefault("summary_v2", {})
        context["summary_v2"] = verified_outputs
        return context

    # -------------------- core verification loop --------------------
    async def _verify_summary(
        self,
        doc_id: str,
        enhanced_summary: str,
        paper_data: Dict[str, Any],
        chat_corpus: List[Dict[str, Any]],
        context: Dict[str, Any],
        track_a: Any,
        track_b: Any,
    ) -> Dict[str, Any]:
        start_time = time.time()

        abstract = self._fetch_abstract(doc_id)
        arxiv_summary = paper_data.get("summary", "")
        goal_title = paper_data.get("title", "")

        knowledge_graph = context.get("knowledge_graph")
        if not knowledge_graph:
            knowledge_graph = await self._build_knowledge_graph(doc_id, paper_data, chat_corpus, context)

        current_summary = enhanced_summary
        current_metrics = self._score_summary(current_summary, abstract, arxiv_summary, knowledge_graph, goal_title, context)
        start_overall = current_metrics.get("overall", 0.0)
        best_summary, best_metrics = current_summary, current_metrics

        iterations: List[Dict[str, Any]] = []
        no_improve_count = 0
        convergence_track: List[float] = []
        lineage_ids = [getattr(track_a, "id", None), getattr(track_b, "id", None)]

        for iter_idx in range(self.max_iters):
            iter_start = time.time()

            # CBR pack
            case_pack = self._retrieve_case_pack(goal_title, k=self.cbr_cases)
            self.report({"event": "cbr_pack", "k": len(case_pack)})

            # prompt with CBR
            prompt = self._build_verification_prompt(
                current_summary=current_summary,
                claims=(knowledge_graph or {}).get("claims", []),
                paper_data=paper_data,
                case_pack=case_pack,
            )
            candidate = self.call_llm(prompt, context=context) or current_summary

            # PACS refinement
            candidate = self._pacs_refine(candidate, abstract, context) or candidate

            # score candidate
            cand_metrics = self._score_summary(candidate, abstract, arxiv_summary, knowledge_graph, goal_title, context)
            gain = cand_metrics["overall"] - current_metrics["overall"]

            # emit iteration tile
            try:
                if self.zero_model_service and hasattr(self.zero_model_service, "emit_iteration_tile"):
                    self.zero_model_service.emit_iteration_tile(
                        doc_id=str(doc_id),
                        iteration=iter_idx + 1,
                        metrics={
                            "overall": cand_metrics.get("overall", 0.0),
                            "knowledge_verification": cand_metrics.get("knowledge_verification", 0.0),
                            "hrm_score": cand_metrics.get("hrm_score", 0.0) if cand_metrics.get("hrm_score") is not None else 0.0,
                        },
                        output_dir="reports/vpm/iters",
                    )
            except Exception as e:
                self.logger.log("VPMIterTileWarn", {"doc_id": doc_id, "error": str(e)})

            # record iteration
            iter_payload = {
                "iteration": iter_idx + 1,
                "current_score": current_metrics["overall"],
                "best_candidate_score": cand_metrics["overall"],
                "gain": gain,
                "processing_time": time.time() - iter_start,
                "knowledge_graph_conf": self.knowledge_graph_conf,
            }
            if knowledge_graph:
                iter_payload["claim_coverage"] = knowledge_graph.get("claim_coverage", 0.0)
                iter_payload["evidence_strength"] = knowledge_graph.get("evidence_strength", 0.0)
            iterations.append(iter_payload)

            # accept if improves enough
            if cand_metrics["overall"] >= self.min_overall and gain >= self.min_gain:
                current_summary = candidate
                current_metrics = cand_metrics
                if cand_metrics["overall"] > best_metrics["overall"]:
                    best_summary, best_metrics = current_summary, current_metrics
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            convergence_track.append(best_metrics["overall"])

            # stops
            if best_metrics["overall"] >= self.target_confidence:
                self.report({"event": "verification_converged", "reason": "target_confidence"})
                break
            if no_improve_count >= 2:
                self.report({"event": "verification_converged", "reason": "no_improve"})
                break
            if len(convergence_track) >= self.convergence_window:
                recent = convergence_track[-self.convergence_window:]
                if np.std(recent) < 1e-2:
                    self.report({"event": "verification_converged", "reason": "convergence_window"})
                    break

        # guardrails
        is_valid, hallucination_issues = self._verify_hallucinations(best_summary, abstract, arxiv_summary, knowledge_graph)
        figure_results = self._verify_figure_grounding(best_summary, paper_data, knowledge_graph)

        # strategy evolution
        if best_metrics["overall"] > start_overall + self.min_gain:
            new_weights = self._adjust_pacs_weights({**best_metrics, "figure_results": figure_results})
            new_threshold = min(0.99, self.strategy.verification_threshold + 0.01)
            self.strategy.update(pacs_weights=new_weights, verification_threshold=new_threshold)
            self._save_strategy_profile(self.strategy)
            self.report({"event": "strategy_updated", "new_weights": new_weights, "new_threshold": new_threshold})

        result = {
            "summary": best_summary,
            "metrics": best_metrics,
            "iterations": iterations,
            "processing_time": time.time() - start_time,
            "hallucination_issues": hallucination_issues,
            "figure_results": figure_results,
            "passes_guardrails": bool(is_valid) and (figure_results.get("overall_figure_score", 0.0) >= self.min_figure_score),
            "converged": best_metrics["overall"] >= self.target_confidence,
            "knowledge_graph": knowledge_graph,
            "verification_trace": {
                "iterations": len(iterations),
                "final_score": best_metrics["overall"],
                "converged": len(convergence_track) >= self.convergence_window and np.std(convergence_track[-self.convergence_window:]) < 1e-2,
            },
        }

        # persist as dynamic scorable
        try:
            scorable_id = self.memory.dynamic_scorables.add(
                pipeline_run_id=context.get("pipeline_run_id"),
                scorable_type=TargetType.DYNAMIC,
                source=self.name,
                text=best_summary,
                meta={
                    "paper_id": paper_data.get("paper_id", doc_id),
                    "title": paper_data.get("title", ""),
                    "metrics": best_metrics,
                    "origin": "track_c_verified",
                    "verification_trace": result["verification_trace"],
                    "hallucination_issues": hallucination_issues,
                    "figure_results": figure_results,
                },
                source_scorable_id=getattr(track_b, "id", None),
                source_scorable_type="dynamic",
            )
            result["scorable_id"] = scorable_id
        except Exception as e:
            self.logger.log("DynamicScorableSaveError", {"doc_id": doc_id, "error": str(e), "traceback": traceback.format_exc()})

        return result

    # -------------------- CBR / PACS / HRM helpers --------------------
    def _retrieve_case_pack(self, title: str, k: int = 3) -> List[Dict[str, Any]]:
        if not self.use_cbr or not self.cbr:
            return []
        try:
            cases = self.cbr.retrieve(goal_text=title, top_k=k)
            pack = []
            for c in cases or []:
                pack.append({
                    "title": (c.get("goal_text") or "")[:160],
                    "why_it_won": (c.get("scores", {}).get("winner_rationale") or "")[:240],
                    "patch": (c.get("lessons") or "")[:240],
                    "summary": (c.get("best_text") or c.get("summary") or "")[:400],
                })
            return pack
        except Exception as e:
            self.logger.log("CBRRetrieveError", {"error": str(e)})
            return []

    def _pacs_refine(self, candidate: str, abstract: str, context: Dict[str, Any]) -> str:
        roles = [
            ("skeptic", "remove speculation; flag ungrounded claims"),
            ("editor", f"tighten structure; keep {self.min_sents}-{self.max_sents} sentences"),
            ("risk", "require figure/table citation for any numeric claim"),
        ]
        panel: List[Tuple[str, str]] = []
        for role, brief in roles:
            prompt = f"""Role: {role.title()}. Brief: {brief}\nAbstract:\n\"\"\"{abstract[:1000]}\"\"\"\n\nText to review:\n\"\"\"{candidate}\"\"\"\n\nReturn ONLY the revised paragraph."""
            try:
                out = self.call_llm(prompt, context=context)
                if out:
                    panel.append((role, out.strip()))
            except Exception:
                continue
        if not panel:
            return candidate

        # role-aware scoring: emphasize different sub-metrics
        best_text = candidate
        best_score = -1.0
        for role, text in panel:
            m = self.metrics_calculator._compute_metrics(text, abstract, "")
            role_score = self._role_weighted_score(role, m)
            if role_score > best_score:
                best_text, best_score = text, role_score
        return best_text

    def _role_weighted_score(self, role: str, m: Dict[str, float]) -> float:
        # Map metrics to role intent
        skeptic_focus = 0.6 * (1.0 - float(m.get("hallucination_rate", 0.0))) + 0.4 * float(m.get("faithfulness", 0.0))
        editor_focus = 0.5 * float(m.get("coherence", 0.0)) + 0.5 * float(m.get("structure", 0.0))
        risk_focus = float(m.get("figure_results", {}).get("overall_figure_score", 0.0)) if isinstance(m.get("figure_results"), dict) else 0.0
        base = float(m.get("overall", 0.0))
        w = self.strategy.pacs_weights.get(role, 0.33)
        # blend base with role focus
        if role == "skeptic":
            score = 0.5 * base + 0.5 * skeptic_focus
        elif role == "editor":
            score = 0.5 * base + 0.5 * editor_focus
        else:  # risk
            score = 0.5 * base + 0.5 * risk_focus
        return w * score

    def _hrm_epistemic(self, text: str, goal: str, context: Dict[str, Any]) -> Tuple[Optional[float], str]:
        if not self.use_hrm or not self.scoring:
            return None, ""
        try:
            scorable = ScorableFactory.from_dict({"text": text, "goal": goal, "type": "document"})
            bundle = self.scoring.score("hrm", context=context, scorable=scorable, dimensions=["alignment"])
            res = getattr(bundle, "results", {}).get("alignment")
            if res is None:
                return None, ""
            score = float(getattr(res, "score", None)) if getattr(res, "score", None) is not None else None
            rationale = getattr(res, "rationale", "")
            return score, rationale
        except Exception as e:
            self.logger.log("HRMScoreError", {"error": str(e)})
            return None, ""

    # -------------------- prompt & scoring --------------------
    def _build_verification_prompt(
        self,
        current_summary: str,
        claims: List[Dict[str, Any]],
        paper_data: Dict[str, Any],
        case_pack: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        title = paper_data.get("title", "")
        abstract = self._fetch_abstract(paper_data.get("id") or paper_data.get("paper_id"))
        claims_text = "\n".join(f"- {c.get('text','').strip()}" for c in (claims or [])[:5] if c.get("text"))
        examples = ""
        if case_pack:
            ex_lines = []
            for ex in case_pack[:3]:
                ex_lines.append(f"- Lesson: {ex.get('patch','')}\n  Why it won: {ex.get('why_it_won','')}")
            if ex_lines:
                examples = "\n\nPrior improvements to emulate:\n" + "\n".join(ex_lines)
        return f"""
You are a verification expert checking this academic paper summary against the paper's key claims.

Title: {title}

Key Claims:
{claims_text}{examples}

Current summary:
\"\"\"{current_summary}\"\"\"

Improve the summary by:
1) Ensuring all key claims are accurately represented
2) Citing figures/tables for quantitative claims when warranted
3) Removing unsupported statements 
4) Preserving clarity and neutrality

Constraints:
- Keep to {self.min_sents}-{self.max_sents} sentences
- Use ONLY facts present in the paper and allowed context
- Do not invent numbers or facts

Verified summary:
""".strip()

    def _score_summary(
        self,
        summary: str,
        abstract: str,
        author_summary: str,
        knowledge_tree: Dict[str, Any],
        goal_title: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        base = self.metrics_calculator._compute_metrics(summary, abstract, author_summary)
        ver = self._verify_against_knowledge_tree(summary, knowledge_tree)
        hrm_score = None
        if self.use_hrm:
            hrm_score, _ = self._hrm_epistemic(summary, goal_title or "", context or {})
        overall = (0.7 * float(base.get("overall", 0.0))) + (0.2 * float(ver))
        if hrm_score is not None:
            overall += (self.hrm_weight * float(hrm_score))
        out = {**base, "knowledge_verification": ver, "overall": overall}
        if hrm_score is not None:
            out["hrm_score"] = float(hrm_score)
        return out

    def _verify_against_knowledge_tree(self, summary: str, knowledge_tree: Dict[str, Any]) -> float:
        if not knowledge_tree:
            return 0.5
        claims = knowledge_tree.get("claims", []) or []
        covered = 0
        for claim in claims:
            text = claim.get("text", "")
            if text and self.metrics_calculator._contains_concept(summary, text):
                covered += 1
        claim_coverage = covered / max(1, len(claims))
        rels = knowledge_tree.get("relationships", []) or []
        strong = [r for r in rels if float(r.get("confidence", 0.0)) >= self.verification_threshold]
        evidence_strength = len(strong) / max(1, len(rels))
        return (0.7 * claim_coverage) + (0.3 * evidence_strength)

    # -------------------- guardrails --------------------
    def _verify_hallucinations(self, summary: str, abstract: str, author_summary: str, knowledge_tree: Dict[str, Any]) -> Tuple[bool, List[str]]:
        # Make AntiHallucination resilient to key-type mismatches in figure maps etc.
        try:
            is_valid, issues = self.anti_hallucination.verify_section(summary, knowledge_tree, {"abstract": abstract, "summary": author_summary})
            return (bool(is_valid), issues or [])
        except Exception as e:
            self.logger.log("AntiHallucinationError", {"error": str(e)})
            return True, ["anti_hallucination_failed_soft"]

    def _verify_figure_grounding(self, summary: str, paper_data: Dict[str, Any], knowledge_tree: Dict[str, Any]) -> Dict[str, Any]:
        # Simple heuristic extractor for quant claims → expected to be replaced by FigureGrounding
        quant_claims = []
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary or "") if s.strip()]
        for sent in sentences:
            matches = re.findall(r"(\b\d+\.?\d*\s*(%|percent|accuracy|precision|recall|f1|auc|rmse|mae|bleu)\b)", sent, flags=re.I)
            if matches:
                quant_claims.append({
                    "claim": sent,
                    "value": matches[0][0],
                    "metric": matches[0][1],
                    "has_citation": any(marker in sent.lower() for marker in ["figure", "fig.", "table", "as shown", "see"]),
                })
        properly_cited = sum(1 for c in quant_claims if c.get("has_citation"))
        citation_rate = properly_cited / max(1, len(quant_claims))
        return {
            "total_claims": len(quant_claims),
            "properly_cited": properly_cited,
            "citation_rate": citation_rate,
            "overall_figure_score": citation_rate,
            "claims": quant_claims,
        }

    # -------------------- strategy evolution --------------------
    def _adjust_pacs_weights(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        weights = dict(self.strategy.pacs_weights)
        if float(metrics.get("knowledge_verification", 0.0)) > 0.8:
            weights["editor"] = min(0.4, weights.get("editor", 0.33) + 0.05)
            weights["skeptic"] = max(0.2, weights.get("skeptic", 0.33) - 0.05)
        if float(metrics.get("hallucination_rate", 1.0)) > 0.2:
            weights["skeptic"] = min(0.4, weights.get("skeptic", 0.33) + 0.05)
            weights["editor"] = max(0.2, weights.get("editor", 0.33) - 0.05)
        if float(metrics.get("coverage", 0.0)) < 0.7:
            weights["skeptic"] = min(0.4, weights.get("skeptic", 0.33) + 0.03)
        if float(metrics.get("coherence", 0.0)) < 0.7:
            weights["editor"] = min(0.4, weights.get("editor", 0.33) + 0.03)
        fig_score = 0.0
        fr = metrics.get("figure_results")
        if isinstance(fr, dict):
            fig_score = float(fr.get("overall_figure_score", 0.0))
        if fig_score < 0.7:
            weights["risk"] = min(0.4, weights.get("risk", 0.33) + 0.03)
        # normalize
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}

    # -------------------- utilities --------------------
    def _safe_meta(self, obj) -> dict:
        meta = getattr(obj, "meta", {}) or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        return meta

    def _emit_vpm_tiles(
        self,
        *,
        doc_id,
        title: str,
        metrics_a: dict | None,
        metrics_b: dict | None,
        metrics_c: dict | None,
        iterations_c: list[dict] | None,
        out_dir: str = "reports/vpm",
        lineage_ids: List[Any] | None = None,
    ):
        try:
            svc = self.zero_model_service
            if not svc:
                self.logger.log("VPMSkipServiceMissing", {"doc_id": doc_id, "reason": "zero_model service missing"})
                return
            vpm_data = self._prepare_vpm_data(doc_id, title, metrics_a or {}, metrics_b or {}, metrics_c or {}, iterations_c or [])
            if hasattr(svc, "generate_summary_vpm_tiles"):
                result = svc.generate_summary_vpm_tiles(vpm_data=vpm_data, output_dir=out_dir)
            else:
                # minimal fallback: ABC triptych only
                names = ["overall","coverage","faithfulness","structure","no_halluc"]
                import numpy as _np
                rows = []
                for label, mm in ("A",metrics_a),("B",metrics_b),("C",metrics_c):
                    mm = mm or {}
                    rows.append([
                        float(mm.get("overall", 0.0)),
                        float(mm.get("claim_coverage", 0.0)),
                        float(mm.get("faithfulness", 0.0)),
                        float(mm.get("structure", 0.0)),
                        float(1.0 - mm.get("hallucination_rate", 1.0)),
                    ])
                mat = _np.asarray(rows, dtype=_np.float32)
                out = f"{out_dir}/{doc_id}_abc.gif"
                if hasattr(svc, "_emit_timeline"):
                    svc._emit_timeline(mat, out)
                result = {"quality_tile_path": out}
            self.logger.log("VPMTilesGenerated", {"doc_id": doc_id, **(result or {})})
        except Exception as e:
            self.logger.log("VPMTileGenerationError", {"doc_id": doc_id, "error": str(e), "traceback": traceback.format_exc()})

    def _prepare_vpm_data(self, doc_id, title, metrics_a, metrics_b, metrics_c, iterations_c):
        def pack(m):
            # Map into a compact, consistent bundle
            return {
                "overall": float(m.get("overall", 0.0)),
                "coverage": float(m.get("claim_coverage", m.get("coverage", 0.0))),
                "faithfulness": float(m.get("faithfulness", 0.0)),
                "structure": float(m.get("structure", 0.0)),
                "no_halluc": float(1.0 - m.get("hallucination_rate", 1.0)),
                "figure_ground": float((m.get("figure_results", {}) or {}).get("overall_figure_score", 0.0)) if isinstance(m.get("figure_results"), dict) else 0.0,
            }
        return {
            "doc_id": doc_id,
            "title": title[:80],
            "metrics": {"A": pack(metrics_a), "B": pack(metrics_b), "C": pack(metrics_c)},
            "iterations": iterations_c or [],
            "timestamp": time.time(),
        }

    def _fetch_abstract(self, doc_id) -> str:
        try:
            sections = self.memory.document_sections.get_by_document(doc_id)
            for s in sections:
                sd = s.to_dict()
                if (sd.get("section_name") or "").lower().strip() == "abstract":
                    return sd.get("section_text", "") or ""
        except Exception as e:
            self.logger.log("AbstractFetchFailed", {"doc_id": doc_id, "error": str(e)})
        return ""

    def _emit_training_events(
        self,
        paper: Dict[str, Any],
        baseline_summary: str,
        verified_summary: str,
        baseline_metrics: Dict[str, float],
        verified_metrics: Dict[str, float],
        context: Dict[str, Any],
    ):
        title = paper.get("title", "paper")
        gain = float(verified_metrics.get("overall", 0.0) - (baseline_metrics or {}).get("overall", 0.0))
        w = max(0.1, min(1.0, gain + 0.3))

        # pointwise
        self.memory.training_events.add_pointwise(
            model_key=self.model_key_retriever,
            dimension="alignment",
            query_text=title,
            cand_text=verified_summary,
            label=1,
            weight=float(verified_metrics.get("overall", 0.7)),
            trust=float(verified_metrics.get("overall", 0.7)),
            goal_id=context.get("goal", {}).get("id"),
            pipeline_run_id=context.get("pipeline_run_id"),
            agent_name=self.name,
            source="track_c",
            meta={"stage": "track_c", "gain": gain, "knowledge_verification": verified_metrics.get("knowledge_verification", 0.0)},
        )

        # pairwise vs. Track B
        self.memory.training_events.add_pairwise(
            model_key=self.model_key_ranker,
            dimension="alignment",
            query_text=title,
            pos_text=verified_summary,
            neg_text=baseline_summary,
            weight=w,
            trust=w * 0.6,
            goal_id=context.get("goal", {}).get("id"),
            pipeline_run_id=context.get("pipeline_run_id"),
            agent_name=self.name,
            source="track_c",
            meta={
                "stage": "track_c",
                "verified_score": verified_metrics.get("overall"),
                "baseline_score": (baseline_metrics or {}).get("overall"),
                "gain": gain,
            },
        )

        # optional pairwise vs author/arXiv summary
        author_summary = paper.get("author_summary", "") or ""
        if author_summary.strip():
            author_metrics = self._score_summary(author_summary, paper.get("abstract", ""), author_summary, {}, title, context)
            prefer_verified = verified_metrics.get("overall", 0.0) > author_metrics.get("overall", 0.0)
            pos = verified_summary if prefer_verified else author_summary
            neg = author_summary if prefer_verified else verified_summary
            self.memory.training_events.add_pairwise(
                model_key=self.model_key_ranker,
                dimension="alignment",
                query_text=title,
                pos_text=pos,
                neg_text=neg,
                weight=0.5,
                trust=0.3,
                goal_id=context.get("goal", {}).get("id"),
                pipeline_run_id=context.get("pipeline_run_id"),
                agent_name=self.name,
                source="track_c",
                meta={
                    "stage": "track_c",
                    "verified_score": verified_metrics.get("overall", 0.0),
                    "author_score": author_metrics.get("overall", 0.0),
                    "prefer_verified": prefer_verified,
                },
            )

    async def _build_knowledge_graph(
        self,
        doc_id: str,
        paper_data: Dict[str, Any],
        chat_corpus: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build (or fetch) a knowledge tree for this paper.
        If a service is registered, call its `.build(...)`.
        Otherwise, use the builder agent's `.run(...)`.
        Handles both sync and async call styles and normalizes the result.
        """
        def _as_graph(raw: Any) -> Dict[str, Any]:
            # Accept {"knowledge_graph": ...}, or a raw dict
            if not isinstance(raw, dict):
                return {}
            knowledge_graph = raw.get("knowledge_graph") or raw
            if not isinstance(knowledge_graph, dict):
                return {}
            # Ensure expected fields exist
            knowledge_graph.setdefault("claims", [])
            knowledge_graph.setdefault("relationships", [])
            knowledge_graph.setdefault("claim_coverage", 0.0)
            knowledge_graph.setdefault("evidence_strength", 0.0)
            return knowledge_graph

        try:
            # 1) Prepare context for builders
            tree_context = {
                "paper_section": {
                    "section_name": "Full Paper",
                    "section_text": paper_data.get("text", ""),
                    "paper_id": paper_data.get("paper_id") or doc_id,
                },
                "chat_corpus": chat_corpus,
                "critical_messages": context.get("critical_messages", []),
                "conversation_trajectories": context.get("conversation_trajectories", []),
                "domains": context.get("domains", []),
                "fusion_entities": context.get("fusion_entities", {}),
            }

            builder = self.container.get("knowledge_graph")
            result = None

            if builder is not None and hasattr(builder, "build"):
                # Service path: .build(...)
                maybe = builder.build(
                    paper_data=paper_data,
                    chat_corpus=chat_corpus,
                    context=context,
                    tree_context=tree_context,
                    doc_id=doc_id,
                )
                result = await maybe if inspect.iscoroutine(maybe) else maybe

            elif builder is not None and hasattr(builder, "run"):
                # Some services/older agents expose .run(context)
                maybe = builder.run(tree_context)
                result = await maybe if inspect.iscoroutine(maybe) else maybe

            else:
                # 3) Fallback: local agent with .run(context)
                agent = KnowledgeTreeBuilderAgent(self.cfg, self.memory, self.container, self.logger)
                maybe = agent.run(tree_context)
                result = await maybe if inspect.iscoroutine(maybe) else maybe

            return _as_graph(result)

        except Exception as e:
            self.logger.log("KnowledgeTreeBuildFailed", {
                "doc_id": doc_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            return {}

    # --- Emit one VPM tile per iteration (safe wrapper) ---
    def _emit_vpm_tile(
        self,
        doc_id: str,
        stage: str,
        metrics: Dict[str, Any],
        lineage_ids: List[Any] | None,
        context: Dict[str, Any],
    ) -> None:
        """
        Best-effort tile emission. If a ZeroModel/visualization service is
        available, we forward the request; otherwise we no-op.
        """
        try:
            zm = getattr(self, "zero_model_service", None) or self.container.get("zeromodel")
        except Exception:
            zm = None

        if not zm:
            # Nothing to do; keep pipeline robust
            self.logger.log("VPMTileSkipNoService", {"doc_id": doc_id, "stage": stage})
            return

        try:
            payload = {
                "doc_id": str(doc_id),
                "stage": stage,
                "metrics": dict(metrics or {}),
                "lineage": [x for x in (lineage_ids or []) if x is not None],
                "vpf": {
                    "pipeline_run_id": context.get("pipeline_run_id"),
                    "agent": self.name,
                    "stage": stage,
                },
            }
            # Delegate to the service; tolerate either method name
            fn = (
                getattr(zm, "create_tile", None)
                or getattr(zm, "generate_summary_vpm_tiles", None)
                or getattr(zm, "emit_tile", None)
            )
            if callable(fn):
                fn(**payload) if fn.__code__.co_kwonlyargcount else fn(payload)
            else:
                self.logger.log("VPMTileSkipNoAPI", {"doc_id": doc_id, "stage": stage})
        except Exception as e:
            self.logger.log("VPMTileEmitError", {
                "doc_id": doc_id,
                "stage": stage,
                "error": str(e),
            })

    # --- Scoring (with knowledge + optional HRM) ---
    def _score_summary(
        self,
        summary: str,
        abstract: str,
        author_summary: str,
        knowledge_tree: Dict[str, Any],
        goal_title: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Base Track-A metrics + knowledge verification (+ optional HRM).
        Compatible with older callers that don’t pass goal_title/context.
        """
        # 1) deterministic base metrics
        base = self.metrics_calculator._compute_metrics(summary, abstract, author_summary)

        # 2) knowledge verification (claim coverage + evidence strength)
        ver = self._verify_against_knowledge_tree(summary, knowledge_tree)

        # 3) optional HRM epistemic judge
        hrm_score = None
        try:
            if getattr(self, "use_hrm", False):
                _ctx = context or {}
                _goal = goal_title or (_ctx.get("goal", {}) or {}).get("goal_text", "")
                hrm_score, _ = self._hrm_epistemic(summary, _goal, _ctx)
        except Exception:
            hrm_score = None

        # 4) blend
        #   keep your prior weighting; add a small HRM term if present
        overall = (0.8 * base.get("overall", 0.0)) + (0.2 * ver)
        if hrm_score is not None:
            overall = 0.7 * base.get("overall", 0.0) + 0.2 * ver + 0.1 * float(hrm_score)

        out = dict(base)
        out["knowledge_verification"] = float(ver)
        if hrm_score is not None: 
            out["hrm_score"] = float(hrm_score)
        out["overall"] = float(overall)
        return out
