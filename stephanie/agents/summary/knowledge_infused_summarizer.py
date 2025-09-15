# stephanie/agents/summary/knowledge_infused_summarizer.py
from __future__ import annotations

import inspect
import time
import traceback
from typing import Dict, Any, List, Tuple

import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.summary.paper_summarizer import SimplePaperSummarizerAgent
from stephanie.agents.knowledge.knowledge_tree_builder import KnowledgeTreeBuilderAgent
from stephanie.knowledge.anti_hallucination import AntiHallucination
from stephanie.knowledge.figure_grounding import FigureGrounding

# Defaults
MAX_ITERS_DEFAULT = 5
MIN_GAIN_DEFAULT = 0.015
MIN_OVERALL_DEFAULT = 0.80
TARGET_CONFIDENCE_DEFAULT = 0.95
MIN_FIGURE_SCORE_DEFAULT = 0.80
VERIFICATION_THRESHOLD_DEFAULT = 0.90
CONVERGENCE_WINDOW_DEFAULT = 2
KNOWLEDGE_TREE_CONF_DEFAULT = 0.70
SENTS_MIN_DEFAULT = 4
SENTS_MAX_DEFAULT = 20


class KnowledgeInfusedVerifierAgent(BaseAgent):
    """
    Track C: Knowledge-Infused Verifier
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Config
        self.max_iters = int(cfg.get("max_iters", MAX_ITERS_DEFAULT))
        self.min_gain = float(cfg.get("min_gain", MIN_GAIN_DEFAULT))
        self.min_overall = float(cfg.get("min_overall", MIN_OVERALL_DEFAULT))
        self.target_confidence = float(cfg.get("target_confidence", TARGET_CONFIDENCE_DEFAULT))
        self.min_figure_score = float(cfg.get("min_figure_score", MIN_FIGURE_SCORE_DEFAULT))
        self.verification_threshold = float(cfg.get("verification_threshold", VERIFICATION_THRESHOLD_DEFAULT))
        self.convergence_window = int(cfg.get("convergence_window", CONVERGENCE_WINDOW_DEFAULT))
        self.knowledge_tree_conf = float(cfg.get("knowledge_tree_conf", KNOWLEDGE_TREE_CONF_DEFAULT))

        # Sentence window (keep aligned with A/B)
        self.min_sents = int(cfg.get("min_sents", SENTS_MIN_DEFAULT))
        self.max_sents = int(cfg.get("max_sents", SENTS_MAX_DEFAULT))

        # Dependencies
        # Use Track A metrics util (deterministic, lightweight)
        self.metrics_calculator = SimplePaperSummarizerAgent(cfg, memory, container, logger)
        self.anti_hallucination = AntiHallucination(logger)
        self.figure_grounding = FigureGrounding(logger)

        # Model keys
        self.model_key_ranker = cfg.get("model_key_ranker", "ranker.sicql.v1")
        self.model_key_retriever = cfg.get("model_key_retriever", "retriever.mrq.v1")

        self.logger.log("KnowledgeInfusedVerifierInit", {
            "max_iters": self.max_iters,
            "verification_threshold": self.verification_threshold,
            "convergence_window": self.convergence_window
        })

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.report({
            "event": "start",
            "step": "KnowledgeInfusedVerifier",
            "details": "Track C verification loop"
        })


        documents = context.get("documents", [])
        chat_corpus = context.get("chat_corpus", [])

        verified_outputs: Dict[Any, Dict[str, Any]] = {}

        for doc in documents:
            doc_id = doc.get("id")

            track_a_output = self.memory.dynamic_scorables.get_latest_by_source_pointer(
                source="paper_summarizer",
                source_scorable_type="document",
                source_scorable_id=doc_id,
            )

            track_b_output = self.memory.dynamic_scorables.get_latest_by_source_pointer(
                source="sharpened_paper_summarizer",
                source_scorable_type="dynamic",
                source_scorable_id=track_a_output.id,
            )

            # Track B doesn’t set "valid"; only gate on guardrails
            # if not track_b_output.get("passes_guardrails", False):
            #     self.logger.log("TrackBSkipped", {
            #         "doc_id": doc_id,
            #         "reason": "guardrails_not_passed"
            #     })
            #     continue

            try:
                verified = await self._verify_summary(
                    doc_id=str(doc_id),
                    enhanced_summary=track_b_output["summary"],
                    paper_data=doc,
                    chat_corpus=chat_corpus,
                    context=context
                )
            except Exception as e:
                self.logger.log("TrackCVerifyError", {
                    "doc_id": doc_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                continue

            verified_outputs[doc_id] = verified

            # Emit training events if high quality
            try:
                if verified["metrics"]["overall"] >= self.min_overall and verified["passes_guardrails"]:
                    self._emit_training_events(
                        paper={
                            "paper_id": doc.get("paper_id", doc_id),
                            "title": doc.get("title", ""),
                            "abstract": self._fetch_abstract(doc_id),
                            "author_summary": doc.get("summary", "")
                        },
                        baseline_summary=track_b_output["summary"],
                        verified_summary=verified["summary"],
                        baseline_metrics=track_b_output["metrics"],
                        verified_metrics=verified["metrics"],
                        context=context
                    )
            except Exception as e:
                self.memory.session.rollback()
                self.logger.log("TrainingEventEmitError", {"doc_id": doc_id, "error": str(e)})

        context.setdefault("summary_v2", {})
        context["summary_v2"] = verified_outputs
        return context

    # -------------------- Core verification loop --------------------

    async def _verify_summary(
        self,
        doc_id: str,
        enhanced_summary: str,
        paper_data: Dict[str, Any],
        chat_corpus: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        start_time = time.time()

        abstract = self._fetch_abstract(doc_id)
        arxiv_summary = paper_data.get("summary", "")

        # Build or get knowledge tree
        knowledge_tree = context.get("knowledge_tree")
        if not knowledge_tree:
            knowledge_tree = await self._build_knowledge_tree(
                doc_id=doc_id,
                paper_data=paper_data,
                chat_corpus=chat_corpus,
                context=context
            )

        current_summary = enhanced_summary
        current_metrics = self._score_summary(current_summary, abstract, arxiv_summary, knowledge_tree)
        best_summary, best_metrics = current_summary, current_metrics

        iterations: List[Dict[str, Any]] = []
        no_improve_count = 0
        convergence_track: List[float] = []

        for iter_idx in range(self.max_iters):
            iter_start = time.time()

            candidates = self._generate_verification_candidates(current_summary, knowledge_tree, paper_data)
            scored_candidates = [
                (cand, self._score_summary(cand, abstract, arxiv_summary, knowledge_tree))
                for cand in candidates
            ]

            best_candidate, best_score = self._select_best_candidate(
                scored_candidates, current_metrics, self.min_gain
            )

            iter_payload = {
                "iteration": iter_idx + 1,
                "current_score": current_metrics["overall"],
                "best_candidate_score": best_score,
                "processing_time": time.time() - iter_start,
                "knowledge_tree_conf": self.knowledge_tree_conf
            }
            if knowledge_tree:
                iter_payload["claim_coverage"] = knowledge_tree.get("claim_coverage", 0.0)
                iter_payload["evidence_strength"] = knowledge_tree.get("evidence_strength", 0.0)
            iterations.append(iter_payload)

            if best_candidate is None:
                self.logger.log("VerificationNoImprovement", {
                    "doc_id": doc_id,
                    "iteration": iter_idx + 1,
                    "final_score": best_metrics["overall"]
                })
                break

            current_summary = best_candidate
            current_metrics = self._score_summary(current_summary, abstract, arxiv_summary, knowledge_tree)

            if current_metrics["overall"] > best_metrics["overall"]:
                best_summary, best_metrics = current_summary, current_metrics
                no_improve_count = 0
            else:
                no_improve_count += 1

            convergence_track.append(best_metrics["overall"])

            if best_metrics["overall"] >= self.target_confidence:
                break
            if no_improve_count >= 2:
                break
            if len(convergence_track) >= self.convergence_window:
                recent = convergence_track[-self.convergence_window:]
                if np.std(recent) < 1e-2:
                    break

        # Guardrails
        is_valid, hallucination_issues = self._verify_hallucinations(
            best_summary, abstract, arxiv_summary, knowledge_tree
        )
        figure_results = self._verify_figure_grounding(best_summary, paper_data, knowledge_tree)

        result = {
            "summary": best_summary,
            "metrics": best_metrics,
            "iterations": iterations,
            "processing_time": time.time() - start_time,
            "hallucination_issues": hallucination_issues,
            "figure_results": figure_results,
            "passes_guardrails": is_valid and figure_results["overall_figure_score"] >= self.min_figure_score,
            "converged": best_metrics["overall"] >= self.target_confidence,
            "knowledge_tree": knowledge_tree,
            "verification_trace": {
                "iterations": len(iterations),
                "final_score": best_metrics["overall"],
                "converged": len(convergence_track) >= self.convergence_window
                             and np.std(convergence_track[-self.convergence_window:]) < 1e-2
            }
        }
        return result

    async def _build_knowledge_tree(
        self,
        doc_id: str,
        paper_data: Dict[str, Any],
        chat_corpus: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build knowledge tree by delegating to KnowledgeTreeBuilderAgent.
        Handles async/sync run implementations.
        """
        try:
            tree_context = {
                "paper_section": {
                    "section_name": "Full Paper",
                    "section_text": paper_data.get("text", ""),
                    "paper_id": paper_data.get("paper_id", doc_id)
                },
                "chat_corpus": chat_corpus,
                "critical_messages": context.get("critical_messages", []),
                "conversation_trajectories": context.get("conversation_trajectories", []),
                "domains": context.get("domains", []),
                "fusion_entities": context.get("fusion_entities", {})
            }

            builder = KnowledgeTreeBuilderAgent(self.cfg, self.memory, self.container, self.logger)
            maybe = builder.run(tree_context)  # may be coroutine
            result = await maybe if inspect.iscoroutine(maybe) else maybe

            return (result or {}).get("knowledge_tree", {}) or {}
        except Exception as e:
            self.logger.log("KnowledgeTreeBuildFailed", {
                "doc_id": doc_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return {}

    # -------------------- Candidate generation & prompts --------------------

    def _generate_verification_candidates(
        self,
        current_summary: str,
        knowledge_tree: Dict[str, Any],
        paper_data: Dict[str, Any]
    ) -> List[str]:
        # If no knowledge tree/claims, just pass current
        claims = (knowledge_tree or {}).get("claims", [])
        if not claims:
            return [current_summary]

        prompt = self._build_verification_prompt(
            current_summary=current_summary,
            claims=claims,
            paper_data=paper_data
        )
        candidate = self.call_llm(prompt)
        return [candidate, current_summary]

    def _build_verification_prompt(
        self,
        current_summary: str,
        claims: List[Dict[str, Any]],
        paper_data: Dict[str, Any]
    ) -> str:
        title = paper_data.get("title", "")
        abstract = self._fetch_abstract(paper_data.get("id") or paper_data.get("paper_id"))

        claims_text = "\n".join(
            f"- {c.get('text','')}".strip()
            for c in claims[:5]
            if c.get('text')
        )

        return f"""
You are a verification expert checking this academic paper summary against the paper's key claims.

Title: {title}

Key Claims:
{claims_text}

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

    # -------------------- Scoring & verification --------------------

    def _score_summary(
        self,
        summary: str,
        abstract: str,
        author_summary: str,
        knowledge_tree: Dict[str, Any]
    ) -> Dict[str, float]:
        base = self.metrics_calculator._compute_metrics(summary, abstract, author_summary)

        # Knowledge verification (claim coverage + evidence strength)
        ver = self._verify_against_knowledge_tree(summary, knowledge_tree)
        overall = (0.8 * base["overall"]) + (0.2 * ver)

        return {**base, "knowledge_verification": ver, "overall": overall}

    def _verify_against_knowledge_tree(self, summary: str, knowledge_tree: Dict[str, Any]) -> float:
        if not knowledge_tree:
            return 0.5

        claims = knowledge_tree.get("claims", []) or []
        covered = 0
        for claim in claims:
            text = claim.get("text", "")
            if text and self._contains_concept(summary, text):
                covered += 1
        claim_coverage = covered / max(1, len(claims))

        rels = knowledge_tree.get("relationships", []) or []
        strong = [r for r in rels if float(r.get("confidence", 0.0)) >= self.verification_threshold]
        evidence_strength = len(strong) / max(1, len(rels))

        return (0.7 * claim_coverage) + (0.3 * evidence_strength)

    def _contains_concept(self, text: str, concept: str) -> bool:
        # Reuse Track A helper directly
        return self.metrics_calculator._contains_concept(text, concept)

    def _verify_hallucinations(
        self,
        summary: str,
        abstract: str,
        author_summary: str,
        knowledge_tree: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        # Use AntiHallucination component; fall back to Track A detector if it returns None
        is_valid, issues = self.anti_hallucination.verify_section(
            summary,
            knowledge_tree,
            {"abstract": abstract, "summary": author_summary}
        )
        return (bool(is_valid), issues or [])

    def _verify_figure_grounding(
        self,
        summary: str,
        paper_data: Dict[str, Any],
        knowledge_tree: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.figure_grounding.verify_section(
            summary,
            {"section_text": paper_data.get("text", "")},
            knowledge_tree
        )

    # -------------------- Utility --------------------

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
        context: Dict[str, Any]
    ):
        title = paper.get("title", "paper")
        gain = float(verified_metrics.get("overall", 0.0) - (baseline_metrics or {}).get("overall", 0.0))
        w = max(0.1, min(1.0, gain + 0.3))

        # Pointwise
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

        # Pairwise vs Track B
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
            meta={"stage": "track_c", "verified_score": verified_metrics.get("overall"), "baseline_score": (baseline_metrics or {}).get("overall"), "gain": gain},
        )

        # Optional pairwise vs author/arXiv summary
        author_summary = paper.get("author_summary", "") or ""
        if author_summary.strip():
            author_metrics = self._score_summary(author_summary, paper.get("abstract", ""), author_summary, {})
            prefer_verified = verified_metrics["overall"] > author_metrics["overall"]
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
                meta={"stage": "track_c", "verified_score": verified_metrics["overall"], "author_score": author_metrics["overall"], "prefer_verified": prefer_verified},
            )
