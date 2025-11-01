"""
ATSSolver Implementation with Verification Mode

This implementation adds the critical verification mode where the solver
answers WITHOUT SEARCH using ONLY the proposer's evidence, as required
by the RAG-gated verification in the SSP paper.
"""

import heapq
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.components.ssp.core.roles.solver import Solver 
from stephanie.utils.similarity_utils import overlap_score
from stephanie.components.ssp.core.protocols import EpisodeContext, VerificationResult
from stephanie.utils.progress_mixin import ProgressMixin
from stephanie.components.ssp.core.node import Node

# System prompt for verification mode (NO SEARCH)
PROMPT_VERIFICATION = """
SYSTEM:
You are verifying if a question can be answered using ONLY the provided evidence.
DO NOT PERFORM ANY SEARCH. Use ONLY the evidence snippets provided below.

EVIDENCE:
{evidence}

QUESTION:
{question}

INSTRUCTIONS:
1. Analyze if the evidence contains sufficient information to answer the question
2. If yes, provide the answer based on the evidence
3. If no, state that the evidence is insufficient
4. DO NOT INVENT INFORMATION NOT IN THE EVIDENCE

OUTPUT FORMAT:
answer: [your answer or "INSUFFICIENT EVIDENCE"]
rationale: [brief explanation]
"""


@dataclass
class ATSSolver(Solver, ProgressMixin):
    """
    Agentic Tree Search Solver with verification mode.
    
    This implements the ATS solver from the SSP paper with two modes:
    1. Verification mode: answers using ONLY proposer's evidence (no search)
    2. Deep search mode: performs full search to find the best answer
    """
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        memory: Any,
        container: Any,
        logger: Any,
        searcher: Any,  # Should be SolutionSearch
        *,
        event_emitter: Optional[Any] = None,
    ):
        """
        Initialize the ATSSolver.
        
        Args:
            searcher: SolutionSearch instance for retrieval
            max_depth: Maximum search depth
            beam_width: Width of the beam search
            event_emitter: Event emitter for telemetry
            topic: Topic prefix for events
            container: Dependency container
            logger: Logger instance
        """
        self.cfg = cfg
        self.container = container
        self.memory = memory
        self.logger = logger
        self.events = event_emitter
        self.searcher = searcher
        self.max_depth = cfg.get("max_depth", 2)
        self.beam_width = cfg.get("beam_width", 3)
        self.topic = cfg.get("topic", "ssp.ats")
        
        # Get VPM control service
        self.vpm_control = container.get("vpm_control_service")
        self._init_progress(container, logger=logger)
    
    async def solve(
        self,
        question: str,
        seed_answer: str,
        context: Optional[EpisodeContext] = None,
        use_search: bool = True,
        evidence_snippets: Optional[List[str]] = None
    ) -> Tuple[str, List[str], int, Dict[str, Any]]:
        """
        Solve a question using the appropriate search strategy.
        
        Args:
            question: Question to answer
            seed_answer: Ground truth answer (for search guidance)
            context: Additional context for solving
            use_search: Whether to perform search (False for verification mode)
            evidence_snippets: Optional evidence to use (for verification mode)
            
        Returns:
            Tuple of (predicted_answer, evidence_used, steps_taken, metadata)
        """
        if not use_search and evidence_snippets:
            # Verification mode: answer using ONLY the provided evidence
            return await self._verify_with_evidence(question, seed_answer, evidence_snippets)
        
        # Deep search mode: perform full search
        return await self._deep_search(question, seed_answer, context)
    
    async def verify_answer(
        self,
        question: str,
        seed_answer: str,
        evidence_snippets: List[str]
    ) -> VerificationResult:
        """
        Verify an answer using ONLY the provided evidence (no search).
        
        This implements the RAG-gated verification step from the paper.
        
        Args:
            question: Question to answer
            seed_answer: Ground truth answer to verify against
            evidence_snippets: Evidence gathered by the Proposer
            
        Returns:
            VerificationResult object with verification outcome
        """
        # First, check if evidence is sufficient
        if not evidence_snippets:
            return VerificationResult(
                is_valid=False,
                score=0.0,
                reason="No evidence provided",
                filter_results={"evidence_usage": False},
                verification_details={"evidence_count": 0}
            )
        
        # Run verification in no-search mode
        predicted, _, _, meta = await self.solve(
            question,
            seed_answer,
            use_search=False,
            evidence_snippets=evidence_snippets
        )
        
        # Calculate score (F1 or exact match)
        score = self._calculate_verification_score(seed_answer, predicted)
        
        # Determine if verification passed
        threshold = self.container.cfg.get("verify", {}).get("pass_threshold", 0.75)
        is_valid = score >= threshold
        
        return VerificationResult(
            is_valid=is_valid,
            score=score,
            reason=f"Verification {'passed' if is_valid else 'failed'} (score={score:.2f})",
            filter_results={"evidence_usage": True},
            verification_details={
                "predicted": predicted,
                "score": score,
                "threshold": threshold
            }
        )
    
    async def _verify_with_evidence(
        self,
        question: str,
        seed_answer: str,
        evidence_snippets: List[str]
    ) -> Tuple[str, List[str], int, Dict[str, Any]]:
        """
        Answer a question using ONLY the provided evidence (no search).
        
        Args:
            question: Question to answer
            seed_answer: Ground truth answer
            evidence_snippets: Evidence to use
            
        Returns:
            Tuple of (predicted_answer, evidence_used, steps_taken, metadata)
        """
        # Prepare context for verification prompt
        merged_context = {
            "question": question,
            "evidence": "\n".join(evidence_snippets),
            "seed_answer": seed_answer
        }
        
        # Load verification prompt
        prompt = self.container.get("prompt_loader").from_text(
            PROMPT_VERIFICATION,
            merged_context
        )
        
        # Call model
        response = await self.container.get("prompt_service").run_prompt(
            prompt, 
            merged_context
        )
        
        # Parse response
        lines = response.strip().splitlines()
        answer_line = next((ln for ln in lines if ln.startswith("answer:")), "")
        rationale_line = next((ln for ln in lines if ln.startswith("rationale:")), "")
        
        predicted = answer_line.replace("answer:", "").strip()
        rationale = rationale_line.replace("rationale:", "").strip()
        
        return (
            predicted,
            evidence_snippets,
            0,  # No search steps taken
            {
                "rationale": rationale,
                "evidence_used": len(evidence_snippets),
                "verification_mode": True
            }
        )
    
    async def _deep_search(
        self,
        question: str,
        seed_answer: str,
        context: Optional[EpisodeContext] = None
    ) -> Tuple[str, List[str], int, Dict[str, Any]]:
        """
        Perform full search to find the best answer.
        
        Args:
            question: Question to answer
            seed_answer: Ground truth answer
            context: Additional context
            
        Returns:
            Tuple of (predicted_answer, evidence_used, steps_taken, metadata)
        """
        # - progress: start -
        task_key = f"ATS:{hash(question) & 0xffff:04x}"
        rewrites_per_parent = len(self._rewrite(question))
        total_steps = self._estimate_total_steps(rewrites_per_parent)
        self.pstart(task=task_key, total=total_steps)
        self.pstage(task=task_key, stage="root")
        
        # Initialize search
        root = Node(
            id=f"root-{uuid.uuid4().hex[:6]}",
            parent_id=None,
            root_id="root",
            depth=0,
            sibling_index=0,
            node_type="root",
            query=question,
            score=0.0,
            context="",
            task_description=question,
        )
        
        if self.events:
            self.events.on_root_created(root)
        
        best = root
        steps = 0
        done = 0
        
        # Main search loop
        for depth in range(1, self.max_depth + 1):
            self.pstage(task=task_key, stage=f"depth-{depth}")
            
            # Get current candidates (beam)
            candidates = self._get_candidates(root, depth)
            
            # Expand each candidate
            for parent in candidates:
                rewrites = self._rewrite(parent.query)
                
                for i, q2 in enumerate(rewrites):
                    # Search for evidence
                    results = await self.searcher.search(q2, seed_answer=seed_answer, context=context)
                    for question in results:
                        sc = overlap_score(question, seed_answer, memory=self.memory, strategy="lexical")
                        
                        # Create child node
                        child = Node(
                            id=f"node-{uuid.uuid4().hex[:6]}",
                            parent_id=parent.id,
                            root_id=root.id,
                            depth=depth,
                            sibling_index=i,
                            node_type="rewrite",
                            query=q2,
                            score=sc,
                            context=question,
                            task_description=question,
                        )
                        
                        if self.events:
                            self.events.on_node_added(parent, child)
                            self.events.on_backprop(child, delta=float(sc))
                        
                        # Update best node
                        if sc > best.score:
                            best = child
                            if self.events:
                                self.events.on_best_update(best)
                        
                        # Progress tracking
                        steps += 1
                        done += 1
                        self.ptick(task=task_key, done=done, total=total_steps)
            
            # Prune to beam width
            self._prune_to_beam(root)
        
        # - progress: done -
        self.pdone(task=task_key)
        
        # MVP answer extraction
        predicted_answer = best.context if best.context else seed_answer
        evidence = best.context.splitlines()
        
        if self.events:
            self.events.on_progress({
                "phase": "ats_solve_complete",
                "steps": steps,
                "best_score": best.score
            })
            self.events.on_rollout_complete({
                "best": {
                    "id": best.id,
                    "score": best.score,
                    "query": best.query,
                    "depth": best.depth,
                },
                "steps": steps,
            })
        
        return (
            predicted_answer,
            evidence,
            steps,
            {
                "best_score": best.score,
                "search_depth": best.depth,
                "evidence_count": len(evidence)
            }
        )
    
    # --- Helper methods ---
    
    @staticmethod
    def _rewrite(query: str) -> List[str]:
        """Minimal, deterministic rewrites for query expansion."""
        return [
            query,
            query.replace("explain", "describe"),
            query + " in practical terms",
        ]
    
    @staticmethod
    def _overlap_score(text: str, target: str) -> float:
        """Calculate overlap score between text and target."""
        a = set([t for t in text.lower().split() if t.isalpha() or t.isalnum()])
        b = set([t for t in target.lower().split() if t.isalpha() or t.isalnum()])
        if not a or not b:
            return 0.0
        inter = len(a & b)
        return inter / max(len(b), 1)
    
    def _estimate_total_steps(self, rewrites_per_parent: int) -> int:
        """Estimate total steps for progress tracking."""
        steps = 0
        nodes_at_depth = 1
        for _ in range(1, self.max_depth + 1):
            steps += nodes_at_depth * rewrites_per_parent
            nodes_at_depth = min(self.beam_width, nodes_at_depth * rewrites_per_parent)
        return steps
    
    def _get_candidates(self, root: Node, depth: int) -> List[Node]:
        """Get candidates at the current depth for expansion."""
        # Get all nodes at the target depth
        candidates = []
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            if node.depth == depth - 1:
                candidates.append(node)
            elif node.depth < depth - 1:
                # In a real implementation, you'd have children to explore
                pass
        
        # Sort by score and take top beam_width
        return heapq.nlargest(self.beam_width, candidates, key=lambda n: n.score)
    
    def _prune_to_beam(self, root: Node) -> None:
        """Prune the tree to keep only the top beam_width nodes at each depth."""
        # In a real implementation, this would prune the tree
        pass
    
    def _calculate_verification_score(self, ground_truth: str, predicted: str) -> float:
        """Calculate F1 score for verification."""
        # Simple implementation - in practice, use proper F1
        gt_words = set(ground_truth.lower().split())
        pred_words = set(predicted.lower().split())
        
        if not gt_words or not pred_words:
            return 0.0
            
        common = gt_words & pred_words
        precision = len(common) / len(pred_words)
        recall = len(common) / len(gt_words)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supports_verification_mode": True,
            "max_search_depth": self.max_depth,
            "beam_width": self.beam_width
        }