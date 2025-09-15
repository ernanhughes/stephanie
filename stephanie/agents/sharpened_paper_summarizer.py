from __future__ import annotations

import re
import time
from typing import Dict, Any, List, Tuple, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable_factory import TargetType
from stephanie.agents.paper_summarizer import SimplePaperSummarizerAgent

MAX_ITERS_DEFAULT = 4
MIN_GAIN_DEFAULT = 0.02
MIN_OVERALL_DEFAULT = 0.75
TARGET_CONFIDENCE_DEFAULT = 0.85
MIN_FIGURE_SCORE_DEFAULT = 0.70


class SharpenedPaperSummarizerAgent(BaseAgent):
    """
    Track B: Super Sharpening (GROWS → CRITIC → REFLECT loop)

    Inputs (context):
      - documents: list of papers with fields {id, title, summary (arXiv/author)}
      - Track A outputs (baseline): either
          context['summary_v0'][doc_id]                            OR
          context['<track_a_key>']['summary_v0'][doc_id]
        (configurable via `track_a_key`)

    Outputs (context):
      - context[self.output_key]['summary_v1'][doc_id] = {
            summary, metrics, iterations, processing_time, hallucination_issues,
            figure_results, passes_guardrails, converged, scorable_id?
        }
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        # Sharpening loop parameters
        self.max_iters = int(cfg.get("max_iters", MAX_ITERS_DEFAULT))
        self.min_gain = float(cfg.get("min_gain", MIN_GAIN_DEFAULT))
        self.min_overall = float(cfg.get("min_overall", MIN_OVERALL_DEFAULT))
        self.target_confidence = float(cfg.get("target_confidence", TARGET_CONFIDENCE_DEFAULT))
        self.min_figure_score = float(cfg.get("min_figure_score", MIN_FIGURE_SCORE_DEFAULT))

        # Sents window (kept consistent with Track A)
        self.min_sents = int(cfg.get("min_sents", 4))
        self.max_sents = int(cfg.get("max_sents", 5))

        # Where to read Track A “summary_v0” from (top level or nested under agent key)
        self.track_a_key = cfg.get("track_a_key")  # e.g., "paper_summarizer"

        # Training model keys
        self.model_key_ranker = cfg.get("model_key_ranker", "ranker.sicql.v1")
        self.model_key_retriever = cfg.get("model_key_retriever", "retriever.mrq.v1")

        # Metrics helper: reuse your Track A’s deterministic metrics
        self.metrics = SimplePaperSummarizerAgent(cfg, memory, container, logger)

        # Optional MR.Q pairwise value via ScoringService (auto fallback to cosine)
        self.scoring = container.get("scoring")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track B: Super sharpening loop that:
        1) Pulls baseline summaries from dynamic_scorables via provenance pointer
        2) Runs sharpening (+ scoring + guardrails)
        3) Persists refined dynamic scorable linking back to the baseline dynamic scorable
        4) Emits training events for retriever/ranker
        """
        self.report({"event": "start", "step": "SuperSharpening", "details": "Track B sharpening loop"})

        documents = context.get("documents", []) or context.get(self.input_key, [])
        out_v1: Dict[Any, Dict[str, Any]] = {}

        for doc in documents:
            doc_id = doc.get("paper_id") or doc.get("id")
            if doc_id is None:
                self.logger.log("SharpenSkipNoDocId", {"doc": str(doc)[:200]})
                continue

            # 1) 🔎 Pull baseline summary saved by Track A (SimplePaperSummarizerAgent)
            baseline_obj = self.memory.dynamic_scorables.get_latest_by_source_pointer(
                source="paper_summarizer",            # Track A agent name
                source_scorable_type="document",
                source_scorable_id=int(doc_id),
            )
            if not baseline_obj:
                self.logger.log("SharpenBaselineMissing", {
                    "doc_id": doc_id,
                    "hint": "Ensure Track A persisted with source_scorable_id=doc_id and type=document"
                })
                continue

            # Extract baseline summary text (prefer the ## Summary section, fallback to meta.summary)
            baseline_text = baseline_obj.text or ""
            baseline_summary = self._extract_summary_from_text(baseline_text) or (baseline_obj.meta or {}).get("summary", "")
            if not baseline_summary:
                self.logger.log("SharpenBaselineParseFailed", {
                    "doc_id": doc_id,
                    "scorable_id": baseline_obj.id,
                })
                continue

            # 2) 🚀 Sharpen
            enhanced = self._enhance_summary(
                doc_id=str(doc_id),
                baseline_summary=baseline_summary,
                paper_data=doc,
                context=context,
            )
            out_v1[doc_id] = enhanced

            # 3) 💾 Persist refined scorable with provenance pointing to the baseline dynamic scorable
            try:
                refined_text = f"## Summary\n{enhanced['summary']}".strip()
                refined_obj = self.memory.dynamic_scorables.add(
                    pipeline_run_id=context.get("pipeline_run_id"),
                    scorable_type=TargetType.DYNAMIC,
                    source=self.name,  # e.g., "super_sharpening" / your agent name
                    text=refined_text,
                    meta={
                        "paper_id": doc_id,
                        "title": doc.get("title", ""),
                        "metrics": enhanced.get("metrics", {}),
                        "origin": "track_b_super_sharpening",
                    },
                    # 👇 provenance to the baseline dynamic scorable
                    source_scorable_id=int(baseline_obj.id),
                    source_scorable_type="dynamic",
                )
                # ensure an embedding exists
                self.memory.embedding.get_or_create(refined_text)
            except Exception as e:
                self.memory.session.rollback()
                self.logger.log("SharpenRefinedPersistError", {"doc_id": doc_id, "error": str(e)})
                refined_obj = None

            # 4) 🎓 Emit training events (enhanced vs baseline; enhanced vs author summary if present)
            try:
                # Build paper bundle for metrics/grounding
                paper_bundle = {
                    "paper_id": doc_id,
                    "title": doc.get("title", ""),
                    "abstract": self._fetch_abstract(doc.get("id") or doc_id),
                    "author_summary": doc.get("summary", ""),
                }

                # Baseline metrics (if we have them in baseline_obj.meta), else recompute quickly
                baseline_metrics = (baseline_obj.meta or {}).get("metrics")
                if not baseline_metrics:
                    baseline_metrics = self._score_summary(
                        baseline_summary, paper_bundle["abstract"], paper_bundle["author_summary"]
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

    # ---------------------------------------------------------------------
    # Core GROWS → CRITIC → REFLECT one-iteration prompt
    # ---------------------------------------------------------------------
    def _grow_critic_reflect(self, title: str, abstract: str, arxiv_summary: str, current: str, context: Dict[str, Any]) -> Dict[str, str]:
        prompt = self._super_prompt(
            title=title,
            abstract=abstract,
            arxiv_summary=arxiv_summary,
            current_summary=current,
            min_sents=self.min_sents,
            max_sents=self.max_sents,
        )
        out = self.call_llm(prompt, context=context)
        return self._parse_super_output(out)

    @staticmethod
    def _super_prompt(*, title: str, abstract: str, arxiv_summary: str, current_summary: str, min_sents: int, max_sents: int) -> str:
        # Truncate abstract for token safety; model has full context from embeddings anyway
        abst = abstract.strip()
        if len(abst) > 1200:
            abst = abst[:1200] + "..."

        return f"""You are an expert scientific editor tasked with sharpening a paper summary.

Inputs:
- Title: {title}
- Abstract: {abst}
- arXiv summary: {arxiv_summary}

Current summary (to improve, {min_sents}-{max_sents} sentences):
{current_summary}

Run a single-pass GROWS → CRITIC → REFLECT loop:

1) GROWS — propose a stronger candidate summary in {min_sents}-{max_sents} sentences.
   - Be faithful to the Abstract and arXiv summary only.
   - Prefer concrete, verifiable phrasing; avoid hype.
   - Use conservative wording if details are missing (“not specified”).
2) CRITIC — list 3–6 specific issues in the candidate (factual risk, missing claim, vagueness, structure).
3) REVISION — rewrite the candidate addressing the critiques.
4) REFLECT — sanity-check the revision for faithfulness vs the inputs; flag remaining risks if any.
5) REFINED SUMMARY — the final improved summary ({min_sents}-{max_sents} sentences), one paragraph.

Rules:
- Faithfulness-first. No invented numbers. If you reference a quantitative result, say “(see Fig. X)” only if clearly warranted by the inputs.
- No first-person voice, no questions, no bullet points (except in the CRITIC section).
- Output EXACTLY in this markdown structure:

## Candidate
<paragraph>

## Critique
- <issue 1>
- <issue 2>
- <issue 3>

## Revision
<paragraph>

## Reflection
<short paragraph>

## Refined Summary
<one paragraph, {min_sents}-{max_sents} sentences>
"""

    @staticmethod
    def _parse_super_output(text: str) -> Dict[str, str]:
        def block(name: str) -> str:
            m = re.search(rf"^##\s*{name}\s*\n(.+?)(?=^##|\Z)", text, flags=re.S | re.M)
            return (m.group(1).strip() if m else "").strip()

        candidate = block("Candidate")
        critique = block("Critique")
        revision = block("Revision")
        reflection = block("Reflection")
        refined = block("Refined Summary")

        # If refined missing, fall back to revision; if that’s missing, candidate
        if not refined:
            refined = revision or candidate

        return {
            "candidate": candidate,
            "critique": critique,
            "revision": revision,
            "reflection": reflection,
            "refined_summary": refined,
        }

    # ---------------------------------------------------------------------
    # Persistence & training events (mirrors Track A patterns)
    # ---------------------------------------------------------------------
    def _persist_scorable_document(
        self,
        paper: Dict[str, Any],
        summary_text: str,
        intro_text: str,
        metrics: Dict[str, float],
        context: Dict[str, Any],
    ) -> Optional[str]:
        try:
            scorable = self.memory.dynamic_scorables.add(
                pipeline_run_id=context.get("pipeline_run_id"),
                scorable_type=TargetType.DYNAMIC,
                source=self.name,
                text=summary_text.strip(),
                meta={
                    "paper_id": paper.get("paper_id"),
                    "title": paper.get("title"),
                    "origin": "track_b_super_sharpening",
                    "metrics": metrics,
                },
            )
            self.memory.embedding.get_or_create(summary_text)
            return scorable.id
        except Exception as e:
            self.memory.session.rollback()
            self.logger.log("TrackBPersistFailed", {"error": str(e)})
            return None

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
        w = max(0.1, min(1.0, gain + 0.3))  # soft weight with floor

        # Pointwise positive for enhanced
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

        # Pairwise enhanced vs baseline
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
            meta={"stage": "track_b", "enhanced_score": enhanced_metrics.get("overall"), "baseline_score": (baseline_metrics or {}).get("overall"), "gain": gain},
        )

        # Optional pairwise vs author summary if present
        author_summary = paper.get("author_summary", "") or ""
        if author_summary.strip():
            author_metrics = self._score_summary(author_summary, paper.get("abstract", ""), author_summary)
            prefer_enhanced = enhanced_metrics["overall"] > author_metrics["overall"]
            pos = enhanced_summary if prefer_enhanced else author_summary
            neg = author_summary if prefer_enhanced else enhanced_summary

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
                meta={"stage": "track_b", "enhanced_score": enhanced_metrics["overall"], "author_score": author_metrics["overall"], "prefer_enhanced": prefer_enhanced},
            )

    # ---------------------------------------------------------------------
    # Scoring & guardrails (reuse Track A metrics)
    # ---------------------------------------------------------------------
    def _score_summary(self, summary: str, abstract: str, author_summary: str) -> Dict[str, float]:
        return self.metrics._compute_metrics(summary, abstract, author_summary)

    def _verify_hallucinations(self, summary: str, abstract: str, author_summary: str) -> Tuple[bool, List[str]]:
        issues = self.metrics._detect_hallucinations(summary, abstract, author_summary)
        return len(issues) == 0, issues

    def _verify_figure_grounding(self, summary: str) -> Dict[str, Any]:
        # Heuristic: count quantitative tokens; check for “figure/table” mentions
        quant = re.findall(r"\b\d+\.?\d*\s*(%|percent|accuracy|precision|recall|f1|auc|rmse|mae|bleu)\b", summary, flags=re.I)
        cited = 0
        for _ in quant:
            if any(marker in summary.lower() for marker in ["figure", "fig.", "table", "as shown", "see "]):
                cited += 1
        rate = (cited / max(1, len(quant)))
        return {"total_claims": len(quant), "properly_cited": cited, "citation_rate": rate, "overall_figure_score": rate}

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

    def _extract_summary_from_text(self, text: str) -> str:
        m = re.search(r"(?mi)^##\s*Summary\s*\n(.+?)(?=^##|\Z)", text, re.S)
        return (m.group(1).strip() if m else "").strip()
    
    # Put these inside SharpenedPaperSummarizerAgent

    def _extract_summary_from_text(self, text: str) -> str:
        """
        Try to pull the '## Summary' section; if not present, return the whole text.
        """
        m = re.search(r"^##\s*Summary\s*\n(.+?)(?=^##|\Z)", text, re.S | re.M)
        return (m.group(1).strip() if m else text.strip())

    def _score_summary(self, summary: str, abstract: str, author_summary: str) -> Dict[str, float]:
        """
        Proxy to the Track A metrics so sharpening scores are consistent.
        NOTE: In __init__, add:
            self.metrics_calculator = SimplePaperSummarizerAgent(cfg, memory, container, logger)
        """
        return self.metrics_calculator._compute_metrics(summary, abstract, author_summary)

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

    def _verify_hallucinations(self, summary: str, abstract: str, author_summary: str) -> Tuple[bool, List[str]]:
        issues = self.metrics_calculator._detect_hallucinations(summary, abstract, author_summary)
        return len(issues) == 0, issues

    def _verify_figure_grounding(self, summary: str, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple heuristic: any quantitative claim should have a figure/table mention nearby.
        """
        quant_claims = re.findall(
            r"\b\d+\.?\d*\s*(%|percent|accuracy|precision|recall|f1|auc|rmse|mae|bleu)\b",
            summary, re.I
        )
        properly_cited = 0
        low = summary.lower()
        for _ in quant_claims:
            if any(k in low for k in ["figure", "table", "as shown", "see "]):
                properly_cited += 1
        rate = properly_cited / max(1, len(quant_claims))
        return {
            "total_claims": len(quant_claims),
            "properly_cited": properly_cited,
            "citation_rate": rate,
            "overall_figure_score": rate,
        }

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
        """
        Iterative sharpening with the super prompt, metric gating, and simple guardrails.
        """
        start = time.time()
        title = paper_data.get("title", "")
        abstract = self._fetch_abstract(paper_data.get("id") or doc_id)
        arxiv_summary = paper_data.get("summary", "")

        min_sents = int(self.cfg.get("min_sents", 4))
        max_sents = int(self.cfg.get("max_sents", 5))

        # Initial scores
        best_summary = baseline_summary.strip()
        best_metrics = self._score_summary(best_summary, abstract, arxiv_summary)

        iterations: List[Dict[str, Any]] = []
        no_gain = 0

        for i in range(self.max_iters):
            prompt = self._build_super_sharpen_prompt(
                title=title,
                abstract=abstract,
                summary=best_summary,
                min_sents=min_sents,
                max_sents=max_sents,
            )
            candidate = self.call_llm(prompt, context=context).strip()
            # Candidate might include headers; keep it to the paragraph only.
            candidate = self._extract_summary_from_text(candidate)

            cand_metrics = self._score_summary(candidate, abstract, arxiv_summary)
            gain = cand_metrics["overall"] - best_metrics["overall"]

            iterations.append({
                "iteration": i + 1,
                "candidate_overall": cand_metrics["overall"],
                "current_best": best_metrics["overall"],
                "gain": gain,
            })

            # Accept only if it clears quality thresholds and min gain
            if cand_metrics["overall"] >= self.min_overall and gain >= self.min_gain:
                best_summary, best_metrics = candidate, cand_metrics
                no_gain = 0
            else:
                no_gain += 1

            # Early stops
            if best_metrics["overall"] >= self.target_confidence:
                break
            if no_gain >= 2:
                break

        # Guardrails
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
