# stephanie/config/schema.py
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, PositiveInt, conint, conlist


class AIRWeights(BaseModel):
    E1_viable_children: float = 0.35
    E2_diversity: float = 0.20
    E3_novelty: float = 0.15
    E4_budget: float = 0.10
    V1_delta_score: float = 0.35
    V2_uncert_reduct: float = 0.25
    V3_consistency: float = 0.10
    V4_grounding: float = 0.10
    V5_efficiency: float = 0.10
    V6_dead_end: float = -0.25


class AIR(BaseModel):
    uncert_mode: Literal["disagreement", "ebt"] = "disagreement"
    weights: AIRWeights = AIRWeights()


class MCTS(BaseModel):
    strategy_name: str = "mcts_v1"
    domain: str = "general"
    max_depth: conint(ge=1, le=32) = 4
    branching_factor: conint(ge=1, le=32) = 2
    num_simulations: conint(ge=1, le=4096) = 20
    ucb_weight: float = 1.41
    expand_mode: Literal["single_step", "rollout"] = "single_step"
    samples_per_expand: conint(ge=1, le=64) = 2
    max_lm_calls: PositiveInt = 64
    eval_at: Literal["leaf", "every_k"] = "leaf"
    eval_stride: PositiveInt = 2
    top_k_leaves: PositiveInt = 3
    debug_prompts: bool = False


class ModelCfg(BaseModel):
    name: str = "ollama_chat/qwen3"
    api_base: str = "http://localhost:11434"
    api_key: str = ""


class Scoring(BaseModel):
    scorer_name: str = "sicql"
    dimensions: conlist(str, min_length=1) = [
        "alignment",
        "clarity",
        "implementability",
        "novelty",
        "relevance",
    ]


class AppConfig(BaseModel):
    version: str = "v1"
    mcts: MCTS = MCTS()
    air: AIR = AIR()
    model: ModelCfg = ModelCfg()
    scoring: Scoring = Scoring()

    class Config:
        extra = "forbid"  # no silent typos
