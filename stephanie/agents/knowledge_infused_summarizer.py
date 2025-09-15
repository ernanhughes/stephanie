# stephanie/agents/summarization/knowledge_infused_verifier.py
import re
import time
import traceback
from typing import Dict, Any, List, Tuple, Optional, Set
from datetime import datetime, timezone

import numpy as np
from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable_factory import TargetType
from stephanie.agents.sharpened_paper_summarizer import SharpenedPaperSummarizerAgent
from stephanie.agents.knowledge.knowledge_tree_builder import KnowledgeTreeBuilderAgent
from stephanie.knowledge.anti_hallucination import AntiHallucination
from stephanie.knowledge.figure_grounding import FigureGrounding


# Default configuration for Track C
MAX_ITERS_DEFAULT = 5
MIN_GAIN_DEFAULT = 0.015
MIN_OVERALL_DEFAULT = 0.80
TARGET_CONFIDENCE_DEFAULT = 0.95
MIN_FIGURE_SCORE_DEFAULT = 0.80
VERIFICATION_THRESHOLD_DEFAULT = 0.90
CONVERGENCE_WINDOW_DEFAULT = 2
KNOWLEDGE_TREE_CONF_DEFAULT = 0.70


class KnowledgeInfusedVerifierAgent(BaseAgent):
    """
    Track C: Knowledge-Infused Verifier
    
    Takes Track B enhanced summaries and verifies/refines them using:
    1. Knowledge tree (paper claims + verified insights from chat + connections)
    2. Verification loop with thresholded early-stopping
    3. Anti-hallucination guardrails and figure grounding
    4. Learning from previous iterations (convergence detection)
    
    Inputs (context):
      - summary_v1: Output from SharpenedPaperSummarizerAgent (Track B)
      - documents: Original document data
      - chat_corpus: Conversation history for knowledge fusion
      - knowledge_tree: Pre-built knowledge tree (optional)
    
    Outputs (context):
      - summary_v2: Verified and refined summary with metrics and trace
    """
    
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        
        # Configuration
        self.max_iters = int(cfg.get("max_iters", MAX_ITERS_DEFAULT))
        self.min_gain = float(cfg.get("min_gain", MIN_GAIN_DEFAULT))
        self.min_overall = float(cfg.get("min_overall", MIN_OVERALL_DEFAULT))
        self.target_confidence = float(cfg.get("target_confidence", TARGET_CONFIDENCE_DEFAULT))
        self.min_figure_score = float(cfg.get("min_figure_score", MIN_FIGURE_SCORE_DEFAULT))
        self.verification_threshold = float(cfg.get("verification_threshold", VERIFICATION_THRESHOLD_DEFAULT))
        self.convergence_window = int(cfg.get("convergence_window", CONVERGENCE_WINDOW_DEFAULT))
        self.knowledge_tree_conf = float(cfg.get("knowledge_tree_conf", KNOWLEDGE_TREE_CONF_DEFAULT))
        
        # Dependencies
        self.metrics_calculator = SharpenedPaperSummarizerAgent(cfg, memory, container, logger)
        self.anti_hallucination = AntiHallucination(logger)
        self.figure_grounding = FigureGrounding(logger)
        
        # sensible defaults for model keys
        self.model_key_ranker = cfg.get("model_key_ranker", "ranker.sicql.v1")
        self.model_key_retriever = cfg.get("model_key_retriever", "retriever.mrq.v1")
        
        self.logger.info("KnowledgeInfusedVerifierAgent initialized", {
            "max_iters": self.max_iters,
            "verification_threshold": self.verification_threshold,
            "convergence_window": self.convergence_window
        })

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify and refine Track B summaries using knowledge tree and conversation history."""
        self.report({
            "event": "start",
            "step": "KnowledgeInfusedVerifier",
            "details": "Track C verification loop"
        })
        
        # Get Track B outputs
        track_b_outputs = context.get("summary_v1", {})
        documents = context.get("documents", [])
        chat_corpus = context.get("chat_corpus", [])
        
        verified_outputs = {}
        for doc in documents:
            doc_id = doc.get("id")
            if doc_id not in track_b_outputs:
                continue
                
            track_b_output = track_b_outputs[doc_id]
            if not track_b_output.get("valid", False) or not track_b_output.get("passes_guardrails", False):
                self.logger.log("TrackBSkipped", {
                    "doc_id": doc_id,
                    "reason": "invalid_track_b_output"
                })
                continue
                
            # Verify the summary
            verified = self._verify_summary(
                doc_id=doc_id,
                enhanced_summary=track_b_output["summary"],
                paper_data=doc,
                chat_corpus=chat_corpus,
                context=context
            )
            
            verified_outputs[doc_id] = verified
            
            # Persist training events if high quality
            if verified["metrics"]["overall"] >= self.min_overall and verified["passes_guardrails"]:
                try:
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
                    self.logger.log("TrainingEventEmitError", {
                        "doc_id": doc_id,
                        "error": str(e)
                    })
        
        context.setdefault("summary_v2", {})
        context["summary_v2"] = verified_outputs
        return context
    
    def _verify_summary(self, doc_id: str, enhanced_summary: str, paper_data: Dict[str, Any], 
                       chat_corpus: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify and refine a single summary through iterative verification loop."""
        start_time = time.time()
        
        # Get paper data
        abstract = self._fetch_abstract(doc_id)
        arxiv_summary = paper_data.get("summary", "")
        
        # Build knowledge tree (if not provided in context)
        knowledge_tree = context.get("knowledge_tree")
        if not knowledge_tree:
            knowledge_tree = self._build_knowledge_tree(
                doc_id=doc_id,
                paper_data=paper_data,
                chat_corpus=chat_corpus,
                context=context
            )
        
        # Initial state
        current_summary = enhanced_summary
        current_metrics = self._score_summary(
            current_summary, 
            abstract, 
            arxiv_summary,
            knowledge_tree
        )
        best_summary = current_summary
        best_metrics = current_metrics
        iterations = []
        no_improve_count = 0
        convergence_check = []
        
        # Verification loop
        for iter_idx in range(self.max_iters):
            iter_start = time.time()
            
            # Generate verification candidates
            candidates = self._generate_verification_candidates(
                current_summary, 
                knowledge_tree,
                paper_data
            )
            
            # Score all candidates
            scored_candidates = [
                (candidate, self._score_summary(candidate, abstract, arxiv_summary, knowledge_tree))
                for candidate in candidates
            ]
            
            # Select best candidate
            best_candidate, best_score = self._select_best_candidate(
                scored_candidates,
                current_metrics,
                self.min_gain
            )
            
            # Record iteration
            iteration_data = {
                "iteration": iter_idx + 1,
                "current_score": current_metrics["overall"],
                "best_candidate_score": best_score,
                "processing_time": time.time() - iter_start,
                "knowledge_tree_conf": self.knowledge_tree_conf
            }
            
            # Add verification details if available
            if knowledge_tree:
                iteration_data["claim_coverage"] = knowledge_tree.get("claim_coverage", 0.0)
                iteration_data["evidence_strength"] = knowledge_tree.get("evidence_strength", 0.0)
            
            iterations.append(iteration_data)
            
            # Check if we should stop
            if best_candidate is None:
                self.logger.info("Verification stopped: no improvement", {
                    "doc_id": doc_id,
                    "iteration": iter_idx + 1,
                    "final_score": best_metrics["overall"]
                })
                break
                
            # Update current state
            current_summary = best_candidate
            current_metrics = self._score_summary(
                current_summary, 
                abstract, 
                arxiv_summary,
                knowledge_tree
            )
            
            # Track best overall
            if current_metrics["overall"] > best_metrics["overall"]:
                best_summary = current_summary
                best_metrics = current_metrics
                no_improve_count = 0
                convergence_check.append(best_metrics["overall"])
            else:
                no_improve_count += 1
                convergence_check.append(best_metrics["overall"])
            
            # Check stopping conditions
            if best_metrics["overall"] >= self.target_confidence:
                self.logger.info("Verification converged early", {
                    "doc_id": doc_id,
                    "iteration": iter_idx + 1,
                    "final_score": best_metrics["overall"]
                })
                break
                
            if no_improve_count >= 2:
                self.logger.info("Verification stopped: no improvement", {
                    "doc_id": doc_id,
                    "iteration": iter_idx + 1,
                    "final_score": best_metrics["overall"]
                })
                break
                
            # Check convergence (learning from learning)
            if len(convergence_check) >= self.convergence_window:
                recent_scores = convergence_check[-self.convergence_window:]
                std_dev = np.std(recent_scores)
                if std_dev < 0.01:  # Very small variation = converged
                    self.logger.info("Verification stopped: converged", {
                        "doc_id": doc_id,
                        "iteration": iter_idx + 1,
                        "final_score": best_metrics["overall"],
                        "std_dev": std_dev
                    })
                    break
        
        # Final verification
        is_valid, hallucination_issues = self._verify_hallucinations(
            best_summary, 
            abstract,
            arxiv_summary,
            knowledge_tree
        )
        figure_results = self._verify_figure_grounding(best_summary, paper_data, knowledge_tree)
        
        # Create result
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
                "converged": len(convergence_check) >= self.convergence_window and np.std(convergence_check[-self.convergence_window:]) < 0.01
            }
        }
        
        return result
    
    def _build_knowledge_tree(self, doc_id: str, paper_data: Dict[str, Any], 
                             chat_corpus: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Build knowledge tree from paper and conversation history."""
        try:
            # Create context for knowledge tree builder
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
            
            # Run knowledge tree builder
            tree_builder = KnowledgeTreeBuilderAgent(
                cfg=self.cfg,
                memory=self.memory,
                logger=self.logger
            )
            result = tree_builder.run(tree_context)
            
            return result.get("knowledge_tree", {})
        except Exception as e:
            self.logger.log("KnowledgeTreeBuildFailed", {
                "doc_id": doc_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return {}
    
    def _generate_verification_candidates(self, current_summary: str, 
                                        knowledge_tree: Dict[str, Any], 
                                        paper_data: Dict[str, Any]) -> List[str]:
        """Generate verification candidates using knowledge tree."""
        candidates = []
        
        # Only proceed if we have a valid knowledge tree
        if not knowledge_tree:
            return [current_summary]
        
        # Get claims from knowledge tree
        claims = knowledge_tree.get("claims", [])
        
        # If no claims, just return current summary
        if not claims:
            return [current_summary]
        
        # Create verification prompt
        prompt = self._build_verification_prompt(
            current_summary=current_summary,
            claims=claims,
            paper_data=paper_data
        )
        
        # Generate with LLM
        candidate = self.call_llm(prompt)
        candidates.append(candidate)
        
        # Add current summary as fallback
        candidates.append(current_summary)
        
        return candidates
    
    def _build_verification_prompt(self, current_summary: str, claims: List[Dict[str, Any]], 
                                 paper_data: Dict[str, Any]) -> str:
        """Build prompt for verification against knowledge tree."""
        title = paper_data.get("title", "")
        abstract = self._fetch_abstract(paper_data.get("id"))
        
        # Format claims
        claims_text = "\n".join([
            f"- {claim['text']}" 
            for claim in claims[:5]  # Limit to top 5 claims
        ])
        
        return f"""
You are a verification expert checking this academic paper summary for accuracy against the paper's key claims.

Paper Title: {title}

Key Claims from Paper:
{claims_text}

Current summary:
\"\"\"{current_summary}\"\"\"

Verify the summary against the key claims and improve it by:
1. Ensuring all key claims are accurately represented
2. Adding proper citations to figures/tables for quantitative claims
3. Removing unsupported statements
4. Improving clarity while maintaining faithfulness

Constraints:
- Keep the summary to {paper_data.get('min_sents', 4)}-{paper_data.get('max_sents', 5)} sentences
- Use ONLY facts present in the paper and conversation history
- Do not invent numbers or facts not in the paper
- Cite figures/tables for all quantitative claims (e.g., "Figure 3 shows...")

Verified summary:
"""
    
    def _score_summary(self, summary: str, abstract: str, author_summary: str, 
                      knowledge_tree: Dict[str, Any]) -> Dict[str, float]:
        """Score summary using metrics plus knowledge tree verification."""
        # Base metrics from Track A/B
        base_metrics = self.metrics_calculator._score_summary(
            summary, 
            abstract, 
            author_summary
        )
        
        # Knowledge tree verification
        verification_score = self._verify_against_knowledge_tree(
            summary, 
            knowledge_tree
        )
        
        # Blend scores (80% base metrics, 20% verification)
        overall = (0.8 * base_metrics["overall"]) + (0.2 * verification_score)
        
        return {
            **base_metrics,
            "knowledge_verification": verification_score,
            "overall": overall
        }
    
    def _verify_against_knowledge_tree(self, summary: str, knowledge_tree: Dict[str, Any]) -> float:
        """Verify summary against knowledge tree claims and evidence."""
        if not knowledge_tree:
            return 0.5  # Neutral score if no knowledge tree
        
        # Check claim coverage
        claims = knowledge_tree.get("claims", [])
        covered = 0
        for claim in claims:
            if self._contains_concept(summary, claim["text"]):
                covered += 1
        
        claim_coverage = covered / max(1, len(claims))
        
        # Check evidence strength
        evidence_strength = 0.0
        if "relationships" in knowledge_tree:
            # Count high-confidence relationships
            strong_rels = [r for r in knowledge_tree["relationships"] 
                          if r.get("confidence", 0.0) >= self.verification_threshold]
            evidence_strength = len(strong_rels) / max(1, len(knowledge_tree["relationships"]))
        
        # Calculate verification score
        return (0.7 * claim_coverage) + (0.3 * evidence_strength)
    
    def _contains_concept(self, text: str, concept: str) -> bool:
        """Check if text contains a concept (with some variation tolerance)."""
        # Reuse the same method from Track A
        return self.metrics_calculator._metrics._contains_concept(text, concept)
    
    def _select_best_candidate(
        self,
        scored_candidates: List[Tuple[str, Dict[str, float]]],
        current_metrics: Dict[str, float],
        min_gain: float
    ) -> Tuple[Optional[str], float]:
        """Select best candidate that improves over current."""
        # Filter candidates that meet minimum quality
        valid_candidates = [
            (candidate, metrics) 
            for candidate, metrics in scored_candidates
            if metrics["overall"] >= self.min_overall
        ]
        
        if not valid_candidates:
            return None, 0.0
        
        # Sort by score
        valid_candidates.sort(key=lambda x: x[1]["overall"], reverse=True)
        
        # Get best candidate
        best_candidate, best_metrics = valid_candidates[0]
        best_score = best_metrics["overall"]
        
        # Check if it's a significant improvement
        improvement = best_score - current_metrics["overall"]
        if improvement >= min_gain:
            return best_candidate, best_score
        
        return None, 0.0
    
    def _verify_hallucinations(self, summary: str, abstract: str, 
                             author_summary: str, knowledge_tree: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify summary for hallucinations using knowledge tree."""
        # Use AntiHallucination component
        is_valid, issues = self.anti_hallucination.verify_section(
            summary, 
            knowledge_tree,
            {"abstract": abstract, "summary": author_summary}
        )
        
        return is_valid, issues
    
    def _verify_figure_grounding(self, summary: str, paper_data: Dict[str, Any], 
                                knowledge_tree: Dict[str, Any]) -> Dict[str, Any]:
        """Verify figure/table grounding with knowledge tree context."""
        # Use FigureGrounding component
        results = self.figure_grounding.verify_section(
            summary, 
            {"section_text": paper_data.get("text", "")},
            knowledge_tree
        )
        
        return results
    
    def _fetch_abstract(self, doc_id) -> str:
        """Fetch abstract using the same method as Track A/B."""
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
        """Emit training events for knowledge-infused verification."""
        title = paper.get("title", "paper")
        gain = float(verified_metrics.get("overall", 0.0) - (baseline_metrics or {}).get("overall", 0.0))
        w = max(0.1, min(1.0, gain + 0.3))

        # pointwise verified
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
            meta={
                "stage": "track_c",
                "gain": gain,
                "knowledge_verification": verified_metrics.get("knowledge_verification", 0.0)
            },
        )

        # pairwise verified vs baseline (Track B)
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
                "knowledge_verification": verified_metrics.get("knowledge_verification", 0.0)
            },
        )

        # pairwise vs author summary (optional)
        author_summary = paper.get("author_summary", "") or ""
        if author_summary.strip():
            author_metrics = self._score_summary(
                author_summary, 
                paper.get("abstract", ""), 
                author_summary,
                {}
            )
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
                meta={
                    "stage": "track_c",
                    "verified_score": verified_metrics["overall"],
                    "author_score": author_metrics["overall"],
                    "prefer_verified": prefer_verified,
                },
            )
    
    def _maybe_pipeline_run_id(self) -> Optional[str]:
        """Get pipeline run ID if available."""
        try:
            return getattr(self, "pipeline_run_id", None) or None
        except Exception:
            return None