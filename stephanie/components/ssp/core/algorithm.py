# stephanie/components/ssp/core/algorithm.py
"""
Main SSP (Self-Play System) Algorithm Interface

This module defines the core SSP algorithm that coordinates the interaction
between Proposer, Solver, and Verifier components. It encapsulates the SSP loop:

1. Proposer generates a question from a seed answer
2. Verifier checks if the question meets quality criteria (RAG-gated verification)
3. If verified, Solver attempts to answer the question with deep search
4. Rewards are calculated for both Proposer and Solver
5. Components may be updated based on the rewards (training)

The interface is designed to be stateless for easy integration with pipeline systems,
with state management handled by external services.
""" 
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from stephanie.components.ssp.core.protocols import EpisodeContext, EpisodeResult, SSPMetrics
from stephanie.components.ssp.core.roles.proposer import Proposer
from stephanie.components.ssp.core.roles.solver import Solver
from stephanie.components.ssp.core.roles.verifier import Verifier


class SSPAlgorithm(ABC):
    """
    Abstract base class for the SSP algorithm implementation.
    
    This class coordinates the interaction between Proposer, Solver, and Verifier
    components according to the SSP methodology. It handles the full episode lifecycle
    from seed answer to verified question to deep search solution.
    
    Implementations should manage:
    - Episode state tracking
    - Component coordination
    - Reward calculation
    - Training updates (if applicable)
    - Visualization data collection
    """
    
    @abstractmethod
    def __init__(
        self,
        proposer: Proposer,
        solver: Solver,
        verifier: Verifier,
        **kwargs
    ):
        """
        Initialize the SSP algorithm with concrete component implementations.
        
        Args:
            proposer: Proposer implementation
            solver: Solver implementation
            verifier: Verifier implementation
            **kwargs: Additional configuration parameters
        """
        pass
    
    @abstractmethod
    async def run_episode(
        self,
        seed_answer: str,
        context: Optional[EpisodeContext] = None
    ) -> EpisodeResult:
        """
        Run a single SSP episode starting from a seed answer.
        
        Args:
            seed_answer: Ground truth answer to build a question around
            context: Additional context for the episode
            
        Returns:
            EpisodeResult containing the full episode data
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_metrics(self) -> SSPMetrics:
        """
        Get current performance metrics for the SSP system.
        
        Returns:
            SSPMetrics object with current system metrics
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the algorithm state (for fresh training runs).
        """
        pass
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Check if the algorithm has been properly initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        pass