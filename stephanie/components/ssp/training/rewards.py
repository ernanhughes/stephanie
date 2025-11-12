"""
Self-Play Rewards Calculation

This module implements the reward calculation logic for the
SSP self-play dynamics as described in the paper:

- Solver reward: accuracy on verified questions (cooperative)
- Proposer reward: 1 - solver accuracy (adversarial)
"""

from typing import Dict, List

from stephanie.components.ssp.utils.trace import EpisodeTrace


def calculate_self_play_rewards(
    verified_episodes: List[EpisodeTrace],
    unverified_count: int
) -> Dict[str, float]:
    """
    Calculate rewards for both proposer and solver based on episode outcomes.
    
    Args:
        verified_episodes: List of successfully verified episodes
        unverified_count: Number of episodes that failed verification
        
    Returns:
        Dictionary with proposer_reward and solver_reward
    """
    total = len(verified_episodes) + unverified_count
    
    if total == 0:
        return {"proposer_reward": 0.0, "solver_reward": 0.0}
    
    # Solver reward: accuracy on verified questions
    solver_success = sum(1 for ep in verified_episodes if ep.verified) / len(verified_episodes) if verified_episodes else 0.0
    
    # Proposer reward: adversarial (1 - solver accuracy) with novelty bonus
    proposer_reward = (1.0 - solver_success) + (unverified_count / total * 0.1)
    
    return {
        "solver_reward": solver_success,
        "proposer_reward": min(proposer_reward, 1.5),  # Cap at 1.5
        "solver_success_rate": solver_success,
        "acceptance_rate": len(verified_episodes) / total,
        "unverified_rate": unverified_count / total
    }


def update_episode_with_rewards(
    episode: EpisodeTrace,
    solver_reward: float,
    proposer_reward: float
) -> None:
    """
    Update an episode trace with reward information.
    
    Args:
        episode: Episode to update
        solver_reward: Solver's reward for this episode
        proposer_reward: Proposer's reward for this episode
    """
    episode.solver_reward = solver_reward
    episode.proposer_reward = proposer_reward
    episode.verifier_meta["solver_reward"] = solver_reward
    episode.proposer_meta["proposer_reward"] = proposer_reward