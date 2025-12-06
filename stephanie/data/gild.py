# stephanie/data/gild.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import torch


# -------------------------
#  Core training signal
# -------------------------

@dataclass
class GILDSignal:
    """
    Canonical unit of data for GILD policy updates.

    This wraps what you're currently building in GILDTrainerAgent:
    - SICQL diagnostics (q_value, state_value, advantage)
    - Encoded latent state (state_z)
    - Target metadata (goal, target, dimension)
    - Optional extra diagnostics (energy, uncertainty, deltas, etc.)
    """

    # Identity / provenance
    evaluation_id: int
    goal_id: Optional[UUID]
    target_id: UUID
    target_type: str
    dimension: str

    # SICQL diagnostics
    q_value: float
    state_value: float
    advantage: torch.Tensor  # shape: () or (1,)

    # Latent state used by the policy head
    state_z: torch.Tensor    # shape: (z_dim,)

    # Optional diagnostics / meta-signals
    source: str = "sicql"
    pi_value: Optional[float] = None
    energy: Optional[float] = None
    uncertainty: Optional[float] = None
    entropy: Optional[float] = None
    delta_llm: Optional[float] = None
    delta_hrm: Optional[float] = None

    # Bag for anything else you later care about
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- Convenience helpers ----

    def to_device(self, device: torch.device) -> "GILDSignal":
        """Move all tensor fields onto the given device (in-place)."""
        if isinstance(self.state_z, torch.Tensor):
            self.state_z = self.state_z.to(device)
        if isinstance(self.advantage, torch.Tensor):
            self.advantage = self.advantage.to(device)
        return self

    def detach(self) -> "GILDSignal":
        """Detach tensor fields from the computational graph (in-place)."""
        if isinstance(self.state_z, torch.Tensor):
            self.state_z = self.state_z.detach()
        if isinstance(self.advantage, torch.Tensor):
            self.advantage = self.advantage.detach()
        return self

    @staticmethod
    def batch_to_tensors(
        signals: List["GILDSignal"],
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Turn a list of signals into the (state_z, advantage) batch tensors
        that your current _run_training_epoch expects.
        """
        if not signals:
            raise ValueError("batch_to_tensors called with empty signals list")

        states = torch.stack([s.state_z for s in signals])
        advantages = torch.stack([s.advantage for s in signals])

        if device is not None:
            states = states.to(device)
            advantages = advantages.to(device)

        return states, advantages


# -------------------------
#  Meta examples for selector / meta-objective
# -------------------------

@dataclass
class GILDMetaExample:
    """
    Flattened comparison example used by GILDSelectorAgent and any
    meta-objective logic.

    This is the structured version of the rows you already use in
    generate_comparison_report / load_comparison_examples:
    - multiple scorers vs LLM/HRM
    - per-goal, per-target, per-dimension
    """

    goal_id: UUID
    target_id: UUID
    dimension: str

    # Per-scorer raw scores, e.g.
    # {"llm": 82.5, "hnet_mrq": 78.0, "huggingface_ebt": 75.0, ...}
    scores: Dict[str, Optional[float]]

    # Optional extra diagnostics
    delta_llm: Optional[float] = None
    delta_hrm: Optional[float] = None
    cost_ms: Optional[float] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    meta: Dict[str, Any] = field(default_factory=dict)

    # Convenience accessors

    @property
    def llm_score(self) -> Optional[float]:
        return self.scores.get("llm")

    def scorer_names(self) -> List[str]:
        return [k for k in self.scores.keys() if k != "llm"]


# -------------------------
#  GILD configuration & result
# -------------------------

@dataclass
class GILDConfig:
    """
    Hyperparameters for AWR-style GILD training.
    This mirrors the knobs you already use in GILDTrainerAgent.
    """

    beta: float = 1.0                 # temperature for advantage weighting
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 5
    max_examples: Optional[int] = None
    min_abs_advantage: float = 0.0
    entropy_coef: float = 0.0
    gradient_clip_norm: Optional[float] = 1.0

    # Warm-start semantics
    warm_start_only: bool = False
    warm_start_fraction: float = 0.1  # fraction of total rounds to run GILD in


@dataclass
class GILDTrainingResult:
    """
    Summary of a single dimension's GILD training run.

    Intended to be serialized into PlanTrace.meta["gild_results"][dimension]
    and/or a dedicated ORM table.
    """

    dimension: str
    status: str                      # "completed", "model_not_found", etc.
    final_loss: float = 0.0
    num_examples: int = 0
    num_epochs: int = 0

    # Optional before/after metrics for meta-objective
    before_metrics: Dict[str, float] = field(default_factory=dict)
    after_metrics: Dict[str, float] = field(default_factory=dict)

    # To link back to the trace
    trace_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
