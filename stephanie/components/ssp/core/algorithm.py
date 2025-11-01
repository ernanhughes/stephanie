# stephanie/components/ssp/core/algorithm.py
"""
Main SSP (Self-Play System) Algorithm

This module implements the complete SSP algorithm as described in the paper,
coordinating the interaction between Proposer, Solver, and Verifier components.
"""
from __future__ import annotations

import asyncio
import time
import datetime
import uuid
from typing import Any, Dict, List, Optional

from stephanie.components.ssp.core.roles.proposer import Proposer 
from stephanie.components.ssp.core.roles.solver import Solver 
from stephanie.components.ssp.core.roles.verifier import Verifier

from stephanie.components.ssp.core.protocols import EpisodeContext, SSPMetrics
from stephanie.components.ssp.utils.trace import EpisodeTrace
from stephanie.components.ssp.training.rewards import (
    calculate_self_play_rewards,
    update_episode_with_rewards
)
from stephanie.components.ssp.services.vpm_visualization_service import VPMVisualizationService


class SSPAlgorithm:
    """
    Implementation of the SSP algorithm that coordinates the interaction
    between Proposer, Solver, and Verifier components.
    
    This implements the full SSP loop:
    1. Proposer generates a question from a seed answer WITH EVIDENCE GATHERING
    2. Verifier applies rule-based filters and RAG-gated verification
    3. If verified, Solver performs deep search to answer the question
    4. Rewards are calculated for both Proposer and Solver
    5. Components may be updated based on the rewards
    """
    
    def __init__(
        self,
        proposer: Proposer,
        solver: Solver,
        verifier: Verifier,
        vpm_visualization: Optional[VPMVisualizationService] = None,
        **kwargs
    ):
        """
        Initialize the SSP algorithm with concrete component implementations.
        
        Args:
            proposer: Proposer implementation
            solver: Solver implementation
            verifier: Verifier implementation
            vpm_visualization: VPM visualization service
            **kwargs: Additional configuration parameters
        """
        self.proposer = proposer
        self.solver = solver
        self.verifier = verifier
        self.vpm_visualization = vpm_visualization
        self.metrics = SSPMetrics()
        self.episode_history = []
    
    async def run_episode(
        self,
        seed_answer: str,
        context: Optional[EpisodeContext] = None
    ) -> EpisodeTrace:
        """
        Run a single SSP episode starting from a seed answer.
        
        Args:
            seed_answer: Ground truth answer to build a question around
            context: Additional context for the episode
            
        Returns:
            EpisodeTrace containing the full episode data
        """
        episode_id = f"ssp-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
        start_time = time.time()
        
        try:
            # 1. Proposer generates question WITH EVIDENCE GATHERING
            question, proposer_evidence, proposer_meta = await self.proposer.propose(
                seed_answer, 
                context=context
            )
            
            # 2. Verifier applies rule-based filters and RAG-gated verification
            verification_result = await self.verifier.verify(
                question,
                seed_answer,
                proposer_evidence,
                context=context
            )
            
            # 3. If verified, Solver performs deep search
            predicted_answer = ""
            evidence_docs = []
            solver_steps = 0
            solver_meta = {}
            
            if verification_result.is_valid:
                predicted_answer, evidence_docs, solver_steps, solver_meta = await self.solver.solve(
                    question,
                    seed_answer,
                    context=context,
                    use_search=True
                )
            
            # 4. Create episode trace
            episode = EpisodeTrace(
                episode_id=episode_id,
                seed_answer=seed_answer,
                question=question,
                proposer_evidence=proposer_evidence,
                predicted_answer=predicted_answer,
                evidence_docs=evidence_docs,
                verified=verification_result.is_valid,
                verifier_score=verification_result.score,
                solver_steps=solver_steps,
                difficulty=proposer_meta.get("difficulty", 0.5),
                proposer_meta=proposer_meta,
                verifier_meta={
                    "reason": verification_result.reason,
                    "filter_results": verification_result.filter_results,
                    **verification_result.verification_details
                },
                solver_meta=solver_meta
            )
            
            # 5. Calculate rewards
            self._calculate_and_apply_rewards([episode], 0)
            
            # 6. Update metrics
            self._update_metrics(episode)
            
            # 7. Generate VPM visualization if service is available
            if self.vpm_visualization:
                self.vpm_visualization.generate_episode_visualization(
                    unit=episode_id,
                    episode=episode
                )
            
            # 8. Record episode duration
            episode.episode_duration = time.time() - start_time
            
            return episode
            
        except Exception as e:
            self.metrics.total_episodes += 1
            self.metrics.verification_pass_rate = self.metrics.verified_episodes / max(self.metrics.total_episodes, 1)
            
            # Return error episode
            return EpisodeTrace(
                episode_id=episode_id,
                seed_answer=seed_answer,
                question="",
                proposer_evidence=[],
                predicted_answer="",
                evidence_docs=[],
                verified=False,
                verifier_score=0.0,
                solver_steps=0,
                difficulty=0.0,
                proposer_meta={"error": str(e)},
                verifier_meta={"error": str(e)},
                solver_meta={"error": str(e)}
            )
    
    async def train_step(
        self,
        seed_answers: List[str],
        context: Optional[EpisodeContext] = None
    ) -> Dict[str, Any]:
        """
        Run a training step with multiple seed answers.
        
        Args:
            seed_answers: List of ground truth answers to process
            context: Additional context for the training step
            
        Returns:
            Dictionary of training metrics and results
        """
        verified_episodes = []
        unverified_count = 0
        
        # Run episodes in parallel
        tasks = [self.run_episode(seed, context) for seed in seed_answers]
        episodes = await asyncio.gather(*tasks)
        
        # Process results
        for episode in episodes:
            if episode.verified:
                verified_episodes.append(episode)
            else:
                unverified_count += 1
        
        # Calculate rewards
        rewards = calculate_self_play_rewards(verified_episodes, unverified_count)
        
        # Store episodes for potential training updates
        self.episode_history.extend(episodes)
        
        # Return metrics
        return {
            **rewards,
            "total_episodes": len(episodes),
            "verified_count": len(verified_episodes),
            "unverified_count": unverified_count,
            "metrics": self.get_metrics().__dict__
        }
    
    def _calculate_and_apply_rewards(
        self,
        verified_episodes: List[EpisodeTrace],
        unverified_count: int
    ) -> None:
        """Calculate and apply rewards to episodes."""
        rewards = calculate_self_play_rewards(verified_episodes, unverified_count)
        
        for episode in verified_episodes:
            update_episode_with_rewards(
                episode,
                rewards["solver_reward"],
                rewards["proposer_reward"]
            )
    
    def _update_metrics(self, episode: EpisodeTrace) -> None:
        """Update system metrics based on episode outcome."""
        self.metrics.total_episodes += 1
        
        if episode.verified:
            self.metrics.verified_episodes += 1
            
            # Update proposer metrics
            self.metrics.proposer_success_rate = self.metrics.verified_episodes / self.metrics.total_episodes
            self.metrics.avg_question_difficulty = (
                (self.metrics.avg_question_difficulty * (self.metrics.verified_episodes - 1) + episode.difficulty) 
                / self.metrics.verified_episodes
            )
            
            # Update solver metrics
            self.metrics.solver_accuracy = (
                (self.metrics.solver_accuracy * (self.metrics.verified_episodes - 1) + (1.0 if episode.predicted_answer == episode.seed_answer else 0.0)) 
                / self.metrics.verified_episodes
            )
            self.metrics.avg_solver_steps = (
                (self.metrics.avg_solver_steps * (self.metrics.verified_episodes - 1) + episode.solver_steps) 
                / self.metrics.verified_episodes
            )
            
            # Update verification metrics
            self.metrics.verification_pass_rate = self.metrics.verified_episodes / self.metrics.total_episodes
            self.metrics.avg_verification_score = (
                (self.metrics.avg_verification_score * (self.metrics.verified_episodes - 1) + episode.verifier_score) 
                / self.metrics.verified_episodes
            )
        
        # Update self-play metrics
        if episode.proposer_reward is not None and episode.solver_reward is not None:
            self.metrics.proposer_adversarial_reward = (
                (self.metrics.proposer_adversarial_reward * (self.metrics.total_episodes - 1) + episode.proposer_reward) 
                / self.metrics.total_episodes
            )
            self.metrics.solver_cooperative_reward = (
                (self.metrics.solver_cooperative_reward * (self.metrics.total_episodes - 1) + episode.solver_reward) 
                / self.metrics.total_episodes
            )
        
        # Update curriculum difficulty
        self.metrics.curriculum_difficulty = (
            self.metrics.curriculum_difficulty * 0.9 + 
            episode.difficulty * 0.1
        )
        
        self.metrics.last_updated = datetime.now()
    
    def get_metrics(self) -> SSPMetrics:
        """
        Get current performance metrics for the SSP system.
        
        Returns:
            SSPMetrics object with current system metrics
        """
        return self.metrics
    
    def reset(self) -> None:
        """
        Reset the algorithm state (for fresh training runs).
        """
        self.metrics = SSPMetrics()
        self.episode_history = []
    
    def is_initialized(self) -> bool:
        """
        Check if the algorithm has been properly initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        return all([
            self.proposer is not None,
            self.solver is not None,
            self.verifier is not None
        ])