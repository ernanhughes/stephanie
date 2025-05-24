from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class PatternStat:
    goal_id: int
    hypothesis_id: int
    model_name: str
    agent_name: str
    dimension: str            # e.g., "Evidence Use"
    label: str                # e.g., "Data-Driven"
    confidence_score: Optional[float] = None
    created_at: datetime = datetime.utcnow()


def generate_pattern_stats(goal_text, hypothesis_text, pattern_dict, memory, cfg, agent_name, confidence_score=None):
    """Create PatternStat entries for a classified CoT using DB lookup for IDs."""

    # Look up goal and hypothesis IDs
    goal_id = memory.hypotheses.get_or_create_goal(goal_text)
    hypothesis_id = memory.hypotheses.get_id_by_text(hypothesis_text)

    if goal_id is None or hypothesis_id is None:
        raise ValueError("Could not find goal or hypothesis ID for pattern stats insertion.")

    model_name = cfg.get("model", {}).get("name", "unknown")

    stats = []
    for dimension, label in pattern_dict.items():
        stat = PatternStat(
            goal_id=goal_id,
            hypothesis_id=hypothesis_id,
            model_name=model_name,
            agent_name=agent_name,
            dimension=dimension,
            label=label,
            confidence_score=confidence_score,
        )
        stats.append(stat)

    return goal_id, hypothesis_id, stats
