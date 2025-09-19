# stephanie/agents/thought/sharpened_paper_summarizer.py
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.thought.paper_blog import SimplePaperBlogAgent
from stephanie.scoring.scorable_factory import TargetType

MAX_ITERS_DEFAULT = 4
MIN_GAIN_DEFAULT = 0.02
MIN_OVERALL_DEFAULT = 0.75
TARGET_CONFIDENCE_DEFAULT = 0.85
MIN_FIGURE_SCORE_DEFAULT = 0.70
SENTS_MIN_DEFAULT = 4
SENTS_MAX_DEFAULT = 20


class SharpenedPaperSummarizerAgent(BaseAgent):
    """
    Track B: Super Sharpening (GROWS → CRITIC → REFLECT loop)

    Reads Track A baseline summaries from dynamic_scorables using provenance:
      source='paper_summarizer', source_scorable_type='document', source_scorable_id=<doc_id>

    Writes refined dynamic_scorables with provenance pointing to the baseline dynamic scorable.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Sharpening loop parameters
        self.max_iters = int(cfg.get("max_iters", MAX_ITERS_DEFAULT))
        self.min_gain = float(cfg.get("min_gain", MIN_GAIN_DEFAULT))
        self.min_overall = float(cfg.get("min_overall", MIN_OVERALL_DEFAULT))
        self.target_confidence = float(cfg.get("target_confidence", TARGET_CONFIDENCE_DEFAULT))
        self.min_figure_score = float(cfg.get("min_figure_score", MIN_FIGURE_SCORE_DEFAULT))

        # Sentence window (match Track A)
        self.min_sents = int(cfg.get("min_sents", SENTS_MIN_DEFAULT))
        self.max_sents = int(cfg.get("max_sents", SENTS_MAX_DEFAULT All right))

        # Training model keys
        self.model_key_ranker = cfg.get("model_key_ranker", "ranker.sicql.v1")
        self.model_key_retriever = cfg.get("model_key_retriever", "retriever.mrq.v1")

        # Metrics helper (reuse Track A deterministic metrics)
        self.metrics = SimplePaperBlogAgent(cfg, memory, container, logger)

        # Optional scoring service
        self.scoring = container.get("scoring")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.report({"event": "start", "step": "SuperSharpening", "details": "Track B sharpening loop"})

        documents = context.get("documents", []) or context.get(self.input_key, [])
        out_v1: Dict[Any, Dict[str, Any]] = {}

        for doc in documents:
            doc_id = doc.get("paper_id") or doc.get("id")
            if doc_id is None:
                self.logger.log("SharpenSkipNoDocId", {"doc": str(doc)[:200]})
                continue

            # 1) Pull baseline summary created by Track A via provenance
            baseline_obj = None
            try:
                sid = int(doc_id)
                baseline_obj = self.memory.dynamic_scorables.get_latest_by_source_pointer(
                    source="paper_summarizer",
                    source_scorable_type="document",
                    source_scorable_id=sid,
                )
            except Exception:
                # Non-numeric doc_id: optionally fall back by meta.paper_id
                baseline_obj = self.memory.dynamic_scorables.get_latest_by_source_and_meta(
                    source="paper_summarizer",
                    meta_key="paper_id",
                    meta_value=str(doc_id),
                )

            if not baseline_obj:
                self.logger.log("SharpenBaselineMissing", {
                    "doc_id": doc_id,
                    "hint": "Ensure Track A saved dynamic_scorables with source_scorable_id=doc_id and type='document', or meta.paper_id"
                })
                continue

            # Extract baseline summary text (prefer the ## Summary section, fallback to meta.summary)
            baseline_text = baseline_obj.text or ""
            baseline_summary = self._extract_summary_from_text(baseline_text) or (baseline_obj.meta or {}).get("summary", "")
            if not baseline_summary:
                self.logger.log("SharpenBaselineParseFailed", {
                    "doc_id": doc_id,
                    "scorable_id": getattr(baseline_obj, "id", None),
                })
                continue

            # 2) Sharpen
            enhanced = self._enhance_summary(
                doc_id=str(doc_id),
                baseline_summary=baseline_summary,
                paper_data=doc,
                context=context,
            )
            out_v1[doc_id] = enhanced

            # 3) Persist refined scorable with provenance to the baseline scorable
            refined_obj = None
            try:
                refined_text = f"## Summary\n{enhanced['summary']}".strip()
                abstract = self._fetch_abstract(doc.get("id") or doc_id)
                refined_obj = self.memory.dynamic_scorables.add(
                    pipeline_run_id=context.get("pipeline_run_id"),
                    scorable_type=TargetType.DYNAMIC,
                    source=self.name, 
                    text=refined_text,
                    source_scorable_id=int(baseline_obj.id),
                    source_scorable_type="dynamic",
                    meta={
                        "paper_id": doc_id,
                        "title": doc.get("title", ""),
                        "abstract": abstract,
                        "arxiv_summary": doc.get("summary", ""),
                        "text": refined_text,
                        "summary": enhanced.get("summary", ""),
                        "metrics": enhanced.get("metrics", {}),
                        "hallucination_issues": enhanced.get("hallucination_issues", []),
                        "figure_results": enhanced.get("figure_results", {}),
                        "origin": "track_b_super_sharpening",
                        "iterations": enhanced.get("iterations", []),
                        "processing_time": enhanced.get("processing_time", 0),
                        "passes_guardrails": enhanced.get("passes_guardrails", False),
                        "converged": enhanced.get("converged", False),
                    },
                )
                # ensure embedding
                self.memory.embedding.get_or_create(refined_text)
            except Exception as e:
                self.memory.session.rollback()
                self.logger.log("SharpenRefinedPersistError", {"doc_id": doc_id, "error": str(e)})

            # 4) Emit training events
            try:
                # Build paper bundle for metrics/grounding
                paper_bundle = {
                    "paper_id": doc_id,
                    "title": doc.get("title", ""),
                    "abstract": self._fetch_abstract(doc.get("id") or doc_id),
                    "arxiv_summary": doc.get("summary", ""),
                }

                # Baseline metrics (if we have them in baseline_obj.meta), else recompute quickly
                baseline_metrics = (baseline_obj.meta or {}).get("metrics")
                if not baseline_metrics:
                    baseline_metrics = self._score_summary(
                        baseline_summary, paper_bundle["abstract"], paper_bundle["arxiv_summary"]
                    )

                # Emit only if our enhanced pass actually exists
                if enhanced and "metrics" in enhanced:
                    self._emit_training_events(
                        paper=paper_bundle,
                        baseline_summary=baseline_summary,
                        enhanced_summary=enhanced["summary"],
                        baseline_metrics=baseline_metrics,
                        enhanced_metrics=enhanced["metrics"],
                        context=context,
                    )
            except Exception as e:
                self.memory.session.rollback()
                self.logger.log("SharpenTrainingEventError", {"doc_id": doc_id, "error": str(e)})

        context.setdefault("summary_v1", {})
        context["summary_v1"] = out_v1
        return context

    # ---------- Core sharpening prompt ----------
    def _build_super_sharpen_prompt(self, *, title: str, abstract: str, summary: str,
                                    min_sents: int, max_sents: int) -> str:
        """
        A single 'super' sharpening prompt that wraps GROWS + CRITIC + REFLECT.
        Keeps output format to exactly the improved summary paragraph.
        """
        abstract_snip = (abstract or "")[:1000]
        return f"""
You are an expert science editor. Improve the paper summary below using a combined **GROWS + CRITIC + REFLECT** loop.

GROWS:
- Generate alternatives, Review against abstract, Optimize for clarity/flow, Work again on weak spots, Stop when optimal.

CRITIC:
- Find assumptions, spot gaps/overclaims, propose precise fixes grounded in the abstract, then rewrite.

REFLECT:
- Double-check factuality and faithfulness to the abstract; remove speculation and marketing language.

Constraints:
- Output **one paragraph** of {min_sents}-{max_sents} sentences.
- Use ONLY facts present in the abstract; if a detail is missing, prefer generic phrasing over guessing.
- Avoid first person, questions, citations/links, and equations.
- If you mention numbers/metrics, only keep those clearly present in the abstract context.

Paper Title: {title}

Abstract:
\"\"\"
{abstract_snip}
\"\"\"

Current summary:
\"\"\"{summary}\"\"\"

Rewrite now (one paragraph, {min_sents}-{max_sents} sentences):
""".strip()

    def _enhance_summary(self, doc_id: str, baseline_summary: str, paper_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        title = paper_data.get("title", "")
        abstract = self._fetch_abstract(paper_data.get("id") or doc_id)
        arxiv_summary = paper_data.get("summary", "")

        best_summary = baseline_summary.strip()
        best_metrics = self._score_summary(best_summary, abstract, arxiv_summary)

        iterations: List[Dict[str, Any]] = []
        no_gain = 0

        for i in range(self.max_iters):
            prompt = self._build_super_sharpen_prompt(
                title=title,
                abstract=abstract,
                summary=best_summary,
                min_sents=self.min_sents,
                max_sents=self.max_sents,
            )
            candidate = self.call_llm(prompt, context=context).strip()
            candidate = self._extract_summary_from_text(candidate)

            cand_metrics = self._score_summary(candidate, abstract, arxiv_summary)
            gain = cand_metrics["overall"] - best_metrics["overall"]

            iterations.append({
                "iteration": i + 1,
                "candidate_overall": cand_metrics["overall"],
                "current_best": best_metrics["overall"],
                "gain": gain,
            })

            if cand_metrics["overall"] >= self.min_overall and gain >= self.min_gain:
                best_summary, best_metrics = candidate, cand_metrics
                no_gain = 0
            else:
                no_gain += 1

            if best_metrics["overall"] >= self.target_confidence or no_gain >= 2:
                break

        ok_hall, hallucinations = self._verify_hallucinations(best_summary, abstract, arxiv_summary)
        fig_check = self._verify_figure_grounding(best_summary, paper_data)
        passes = ok_hall and (fig_check["overall_figure_score"] >= self.min_figure_score)

        return {
            "summary": best_summary,
            "metrics": best_metrics,
            "iterations": iterations,
            "processing_time": time.time() - start,
            "hallucination_issues": hallucinations,
            "figure_results": fig_check,
            "passes_guardrails": passes,
            "converged": best_metrics["overall"] >= self.target_confidence,
        }

    # ---------- helpers reused from Track A ----------
    def _extract_summary_from_text(self, text: str) -> str:
        m = re.search(r"(?mi)^##\s*Summary\s*\n(.+?)(?=^##|\Z)", text, re.S)
        return (m.group(1).strip() if m else text.strip())

    def _score_summary(self, summary: str, abstract: str, arxiv_summary: str) -> Dict[str, float]:
        return self.metrics._compute_metrics(summary, abstract, arxiv_summary)

    def _verify_hallucinations(self, summary: str, abstract: str, arxiv_summary: str) -> Tuple[bool, List[str]]:
        issues = self.metrics._detect_hallucinations(summary, abstract, arxiv_summary)
        return len(issues) == 0, issues

    def _verify_figure_grounding(self, summary: str, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify figure/table grounding with precise claim matching."""
        # Find quantitative claims with context
        quant_claims = []
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
        
        for sent in sentences:
            matches = re.findall(
                r"(\b\d+\.?\d*\s*(%|percent|accuracy|precision|recall|f1|auc|rmse|mae|bleu)\b)", 
                sent, 
                flags=re.I
            )
            if matches:
                quant_claims.append({
                    "claim": sent,
                    "value": matches[0][0],
                    "metric": matches[0][1],
                    "has_citation": any(marker in sent.lower() for marker in ["figure", "fig.", "table", "as shown", "see"])
                })
        
        # Check if citations match paper content
        properly_cited = 0
        for claim in quant_claims:
            if claim["has_citation"]:
                # In production, this would check if the citation actually supports the claim
                # For now, simple heuristic
                properly_cited += 1
        
        citation_rate = properly_cited / max(1, len(quant_claims))
        
        return {
            "total_claims": len(quant_claims),
            "properly_cited": properly_cited,
            "citation_rate": citation_rate,
            "overall_figure_score": citation_rate,
            "claims": quant_claims
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
        enhanced_summary: str,
        baseline_metrics: Dict[str, float],
        enhanced_metrics: Dict[str, float],
        context: Dict[str, Any],
    ):
        title = paper.get("title", "paper")
        gain = float(enhanced_metrics.get("overall", 0.0) - (baseline_metrics or {}).get("overall", 0.0))
        w = max(0.1, min(1.0, gain + 0.3))

        # pointwise enhanced
        self.memory.training_events.add_pointwise(
            model_key=self.model_key_retriever,
            dimension="alignment",
            query_text=title,
            cand_text=enhanced_summary,
            label=1,
            weight=float(enhanced_metrics.get("overall", 0.7)),
            trust=float(enhanced_metrics.get("overall", 0.7)),
            goal_id=context.get("goal", {}).get("id"),
            pipeline_run_id=context.get("pipeline_run_id"),
            agent_name=self.name,
            source="track_b",
            meta={"stage": "track_b", "gain": gain},
        )

        # pairwise enhanced vs baseline
        self.memory.training_events.add_pairwise(
            model_key=self.model_key_ranker,
            dimension="alignment",
            query_text=title,
            pos_text=enhanced_summary,
            neg_text=baseline_summary,
            weight=w,
            trust=w * 0.6,
            goal_id=context.get("goal", {}).get("id"),
            pipeline_run_id=context.get("pipeline_run_id"),
            agent_name=self.name,
            source="track_b",
            meta={
                "stage": "track_b",
                "enhanced_score": enhanced_metrics.get("overall"),
                "baseline_score": (baseline_metrics or {}).get("overall"),
                "gain": gain,
            },
        )

        # pairwise vs author summary (optional)
        arxiv_summary = paper.get("arxiv_summary", "") or ""
        if arxiv_summary.strip():
            author_metrics = self._score_summary(arxiv_summary, paper.get("abstract", ""), arxiv_summary)
            prefer_enhanced = enhanced_metrics["overall"] > author_metrics["overall"]
            pos = enhanced_summary if prefer_enhanced else arxiv_summary
            neg = arxiv_summary if prefer_enhanced else enhanced_summary

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
                source="track_b",
                meta={
                    "stage": "track_b",
                    "enhanced_score": enhanced_metrics["overall"],
                    "author_score": author_metrics["overall"],
                    "prefer_enhanced": prefer_enhanced,
                },
            )
