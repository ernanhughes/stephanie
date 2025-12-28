"""
Hierarchical Reasoning Model (HRM) - Advanced Neural Architecture for Quality Assessment

This module implements a sophisticated hierarchical recurrent neural network for
evaluating AI model responses. The architecture features two coupled recurrent
networks operating at different temporal scales, enabling complex reasoning
patterns and comprehensive quality assessment.

Architecture Overview:
- Dual recurrent hierarchy: Low-level (L) and High-level (H) modules
- Cyclic processing with fine-grained (L) and abstract (H) updates
- Multi-head prediction for comprehensive quality diagnostics
- Robustness through consistency regularization and uncertainty estimation

Key Features:
- Hierarchical temporal processing (T steps per cycle × N cycles)
- Multi-dimensional quality assessment (score, uncertainty, agreement, etc.)
- Input reconstruction for comprehension verification
- Finite-difference sensitivity analysis
- Aleatoric uncertainty estimation

Author: Stephanie AI Team
Version: 2.1
Date: 2024
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from stephanie.scoring.analysis.trace_tap import TraceTap

log = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) - Efficient Alternative to LayerNorm.
    
    Normalizes across the feature dimension while maintaining representational
    capacity through a learnable scaling parameter. More computationally efficient
    than LayerNorm as it doesn't maintain running statistics.
    
    Reference: "Root Mean Square Layer Normalization" by Zhang & Sennrich (2019)
    
    Args:
        dim: Feature dimension to normalize
        eps: Small constant for numerical stability
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scale parameter

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization: x / sqrt(mean(x^2) + eps)
        
        Args:
            x: Input tensor of shape [..., dim]
            
        Returns:
            Normalized tensor with same shape
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with type preservation.
        
        Args:
            x: Input tensor of any dtype
            
        Returns:
            Normalized tensor with original dtype
        """
        # Convert to float for stable computation, then back to original type
        output = self._norm(x.float()).type_as(x) * self.weight
        return output


class RecurrentBlock(nn.Module):
    """
    Gated Recurrent Unit (GRU) Block with RMSNorm for Stable State Updates.
    
    Implements a single recurrent step with gating mechanisms and normalization
    for stable long-term gradient flow. Used by both L and H modules in HRM.
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden state
        name: Identifier for debugging and logging
    """
    def __init__(self, input_dim: int, hidden_dim: int, name: str = "RecurrentBlock"):
        super().__init__()
        self.name = name
        self.rnn_cell = nn.GRUCell(input_dim, hidden_dim)  # Gated recurrent update
        self.norm = RMSNorm(hidden_dim)  # State normalization

    def forward(self, z_prev: torch.Tensor, input_combined: torch.Tensor) -> torch.Tensor:
        """
        Single recurrent step: GRU update + normalization.
        
        Args:
            z_prev: Previous hidden state [batch, hidden_dim]
            input_combined: Current input features [batch, input_dim]
            
        Returns:
            Updated hidden state [batch, hidden_dim]
        """
        z_next = self.rnn_cell(input_combined, z_prev)  # GRU state update
        z_next = self.norm(z_next)  # Stabilize hidden state
        return z_next

    @staticmethod
    def init_state(batch_size: int, hidden_dim: int, device: torch.device) -> torch.Tensor:
        """
        Initialize hidden state with zeros.
        
        Args:
            batch_size: Number of sequences in batch
            hidden_dim: Hidden state dimension
            device: Target device for tensor allocation
            
        Returns:
            Zero-initialized hidden state [batch_size, hidden_dim]
        """
        return torch.zeros(batch_size, hidden_dim, device=device)


class InputProjector(nn.Module):
    """
    Input Embedding Projection with Normalization and Dropout.
    
    Projects high-dimensional input embeddings into the HRM's hidden space
    with regularization for improved generalization.
    
    Args:
        input_dim: Original input embedding dimension
        hidden_dim: Target HRM hidden dimension
        dropout: Dropout probability for regularization
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim)  # Linear projection
        self.norm = RMSNorm(hidden_dim)  # Output normalization
        self.drop = nn.Dropout(dropout)  # Regularization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input to HRM hidden space: Dropout → Linear → RMSNorm.
        
        Args:
            x: Input embeddings [batch, input_dim]
            
        Returns:
            Projected features [batch, hidden_dim]
        """
        return self.norm(self.drop(self.project(x)))


class HRMModel(nn.Module):
    """
    Hierarchical Reasoning Model (HRM) - Dual-Recurrent Architecture.
    
    Implements a hierarchical reasoning process through two coupled RNNs:
    - Low-level (L) module: Fine-grained processing (T steps per cycle)
    - High-level (H) module: Abstract reasoning (1 step per cycle)
    
    The architecture enables multi-scale temporal processing where the L module
    performs detailed analysis and the H module integrates information across
    longer time horizons.
    
    Multi-Head Diagnostic Surface:
      - score_head: Quality score ∈ [0,1] with temperature calibration
      - logvar_head: Aleatoric uncertainty estimation
      - aux3_head: 3-way classification (bad/medium/good)
      - disagree_head: Prediction of model disagreement
      - consistency_head: Robustness to input perturbations
      - ood_head: Out-of-distribution detection
      - temp_head: Adaptive temperature for score calibration
      - recon_head: Input reconstruction for comprehension verification
      
    Args:
        cfg: Configuration dictionary containing model hyperparameters
        logger: Optional logger instance for training diagnostics
    """

    def __init__(self, cfg: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__()
        self.logger = logger or log

        # Model hyperparameters with type conversion and defaults
        self.input_dim = int(cfg.get("input_dim", 2048))  # Input embedding dimension
        self.h_dim = int(cfg.get("h_dim", 256))           # High-level hidden dimension
        self.l_dim = int(cfg.get("l_dim", 128))           # Low-level hidden dimension
        self.n_cycles = int(cfg.get("n_cycles", 4))       # Number of H cycles
        self.t_steps = int(cfg.get("t_steps", 4))         # L steps per H cycle
        self.dropout = float(cfg.get("dropout", 0.1))     # Dropout probability
        self.consistency_mask_p = float(cfg.get("consistency_mask_p", 0.10))  # Mask probability
        self.fd_eps = float(cfg.get("fd_eps", 1e-3))      # Finite-difference epsilon

        # Device management (updated during .to() calls)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Input projection module
        self.input_projector = InputProjector(
            self.input_dim, self.h_dim, dropout=self.dropout
        )

        # Hierarchical recurrent modules
        # L-module: Fine-grained processing with access to input and H-state
        self.l_module = RecurrentBlock(2 * self.h_dim, self.l_dim, name="LModule")
        # H-module: Abstract reasoning integrating L-states and previous H-state
        self.h_module = RecurrentBlock(self.l_dim + self.h_dim, self.h_dim, name="HModule")

        # Multi-head prediction layer with dropout
        self.head_drop = nn.Dropout(self.dropout)
        
        # Diagnostic prediction heads
        self.score_head = nn.Linear(self.h_dim, 1)        # Quality score logits
        self.logvar_head = nn.Linear(self.h_dim, 1)       # Aleatoric uncertainty
        self.aux3_head = nn.Linear(self.h_dim, 3)         # 3-way classification
        self.disagree_head = nn.Linear(self.h_dim, 1)     # Disagreement prediction
        self.consistency_head = nn.Linear(self.h_dim, 1)  # Robustness prediction
        self.ood_head = nn.Linear(self.h_dim, 1)          # OOD detection
        self.temp_head = nn.Linear(self.h_dim, 1)         # Temperature calibration
        self.recon_head = nn.Linear(self.h_dim, self.h_dim)  # Input reconstruction

        # Final normalization for head inputs
        self.final_norm = RMSNorm(self.h_dim)

    # ---------------------------
    # Core Hierarchical Rollout
    # ---------------------------

    def _rollout(
        self,
        x_tilde: torch.Tensor,
        *,
        max_cycles: Optional[int] = None,
        tap: Optional[TraceTap] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute hierarchical recurrent processing across cycles and steps.
        
        Processing Flow:
          1. For each of n_cycles:
             a. L-module runs t_steps with access to (x_tilde, current zH)
             b. H-module updates once using (final zL, previous zH)
          2. Track maximum H-state for evidence accumulation
        
        Args:
            x_tilde: Projected input features [batch, h_dim]
            
        Returns:
            zL_final: Final low-level state [batch, l_dim]
            zH_final: Final high-level state [batch, h_dim] 
            zH_traj_max: Maximum H-state across trajectory [batch, h_dim]
        """
        batch_size = x_tilde.size(0)
        
        # Initialize hidden states
        zL = RecurrentBlock.init_state(batch_size, self.l_dim, self.device)  # Low-level
        zH = RecurrentBlock.init_state(batch_size, self.h_dim, self.device)  # High-level
        zH_max = torch.zeros_like(zH)  # Track maximum activation

        # Hierarchical recurrent processing
        cycles = self.n_cycles if max_cycles is None else int(max_cycles)
        cycles = max(1, min(cycles, self.n_cycles))
        for cycle in range(cycles):            # Low-level fine-grained processing (T steps)
            for step in range(self.t_steps):
                # L-module input: projected input + current H-state
                l_input = torch.cat([x_tilde, zH], dim=-1)  # [batch, 2 * h_dim]
                zL = self.l_module(zL, l_input)
                if tap is not None:
                    tap.add("hrm/zL", zL)

            # High-level abstract update (1 step per cycle)
            # H-module input: final L-state + previous H-state
            h_input = torch.cat([zL, zH], dim=-1)  # [batch, l_dim + h_dim]
            zH = self.h_module(zH, h_input)
            if tap is not None:
                tap.add("hrm/zH", zH)            

            # Track maximum activation for evidence accumulation
            zH_max = torch.maximum(zH_max, zH)

        # Final normalization for prediction heads
        zH = self.final_norm(zH)
        zH_max = self.final_norm(zH_max)
        
        if tap is not None:
            tap.add("hrm/zL_final", zL)
            tap.add("hrm/zH_final", zH)
        return zL, zH, zH_max

    # ---------------------------
    # Main Forward Pass
    # ---------------------------

    def forward(
        self,
        x: torch.Tensor,                      # Input embeddings [batch, input_dim]
        *,
        return_aux: bool = True,              # Return auxiliary diagnostics
        n_steps: Optional[int] = None,
        tap: Optional["TraceTap"] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Complete forward pass with hierarchical reasoning and multi-head prediction.
        
        Args:
            x: Input embeddings (typically goal ⊕ response or plan)
            return_aux: Whether to compute and return auxiliary outputs
            
        Returns:
            score01: Primary quality score ∈ [0,1] [batch, 1]
            aux: Dictionary of raw and derived diagnostic outputs
        """


        # Input projection and hierarchical processing
        x_tilde = self.input_projector(x)     # [batch, h_dim]
        zL, zH, zH_max = self._rollout(x_tilde, max_cycles=self.n_cycles, tap=tap)
        
        # Prepare state for prediction heads
        zH_head = self.head_drop(zH)  # Regularization

        # Temperature-calibrated scoring
        tau_raw = self.temp_head(zH_head)
        tau = 0.5 + 0.5 * F.softplus(tau_raw)  # τ ∈ (0.5, ∞) with softplus
        temp01  = torch.sigmoid(tau_raw)              # nice bounded proxy for telemetry
        score_logit = self.score_head(zH_head)
        score01 = torch.sigmoid(score_logit / tau)  # Calibrated score ∈ [0,1]

        # Core diagnostic heads
        log_var = self.logvar_head(zH_head)           # Aleatoric uncertainty
        aux3_logits = self.aux3_head(zH_head)         # 3-way classification
        aux3_probs = F.softmax(aux3_logits, dim=-1)   # Probability distribution
        disagree_hat = torch.sigmoid(self.disagree_head(zH_head))  # Disagreement
        ood_hat = torch.sigmoid(self.ood_head(zH_head))           # OOD probability

        # Consistency regularization target
        mask = (torch.rand_like(zH_head) < self.consistency_mask_p).float()
        zH_masked = zH_head * (1.0 - mask)  # Randomly masked state
        consistency_hat = torch.sigmoid(self.consistency_head(zH_head))
        consistency_target = self._cos01(zH_head, zH_masked).unsqueeze(-1)

        # Input reconstruction (comprehension proxy)
        x_recon = self.recon_head(zH_head)  # Reconstruct projected input
        recon_sim = self._cos01(x_recon, x_tilde).unsqueeze(-1)  # Reconstruction quality

        # Finite-difference sensitivity analysis
        x_eps = x + self.fd_eps * F.normalize(torch.randn_like(x), dim=-1)
        with torch.no_grad():
            x_tilde_eps = self.input_projector(x_eps)
            _, zH_eps, _ = self._rollout(x_tilde_eps)
            zH_eps = self.head_drop(zH_eps)
            tau_eps = 0.5 + 0.5 * F.softplus(self.temp_head(zH_eps))
            score_eps = torch.sigmoid(self.score_head(zH_eps) / tau_eps)
        
        # Jacobian approximation via finite differences
        jacobian_fd = ((score_eps - score01).abs() / self.fd_eps).clamp(0, 10.0) / 10.0

        # Pseudo-halting signal (evidence accumulation)
        halt_logit = (zH_max * zH_head).mean(-1, keepdim=True) / max(self.h_dim, 1)
        halt_prob = torch.sigmoid(halt_logit)

        # Early return if only primary score needed
        if not return_aux:
            return score01, {}

        # Comprehensive auxiliary outputs dictionary
        aux: Dict[str, Any] = {
            # Raw head outputs (for loss computation)
            "score_logit": score_logit,                 # [batch, 1]
            "log_var": log_var,                         # [batch, 1]  
            "aux3_logits": aux3_logits,                 # [batch, 3]
            "disagree_logit": self.disagree_head(zH_head),  # [batch, 1]
            "consistency_logit": self.consistency_head(zH_head),  # [batch, 1]
            "x_recon": x_recon,                         # [batch, h_dim]

            # Derived metrics (normalized for visualization)
            "score": score01,                           # [batch, 1] ∈ [0,1]
            "certainty01": torch.sigmoid(-log_var),     # [batch, 1] certainty measure
            "uncertainty": 1.0 - torch.sigmoid(-log_var),     # [batch, 1] alias for back-compat
            "aux3_probs": aux3_probs,                   # [batch, 3] probability distribution
            "entropy_aux": (-(aux3_probs * F.log_softmax(aux3_logits, -1)).sum(-1)
                             / torch.log(torch.tensor(3.0, device=x.device))).unsqueeze(-1),  # [batch, 1]
            "disagree_hat": disagree_hat,               # [batch, 1] predicted disagreement
            "consistency_hat": consistency_hat,         # [batch, 1] robustness prediction
            "consistency_target": consistency_target,   # [batch, 1] regularization target
            "recon_sim": recon_sim,                     # [batch, 1] reconstruction quality
            "ood_hat": ood_hat,                         # [batch, 1] OOD probability
            "temp01": temp01,                           # [batch, 1] temperature proxy
            "jacobian_fd": jacobian_fd,                 # [batch, 1] input sensitivity
            "halt_prob": halt_prob,                     # [batch, 1] evidence accumulation

            # Internal states (for introspection/debugging)
            "zL_final": zL,                            # [batch, l_dim] final L-state
            "zH_final": zH,                            # [batch, h_dim] final H-state
        }
        
        return score01, aux

    # ---------------------------
    # Utility Methods
    # ---------------------------

    @staticmethod
    def _cos01(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        """
        Compute cosine similarity mapped from [-1, 1] to [0, 1].
        
        Args:
            a, b: Input tensors to compare
            dim: Dimension for cosine computation
            eps: Numerical stability constant
            
        Returns:
            Cosine similarity normalized to [0, 1] range
        """
        sim = F.cosine_similarity(a, b, dim=dim, eps=eps)
        return (sim + 1.0) * 0.5

    def to(self, device):
        """
        Move model to specified device and update internal device reference.
        
        Args:
            device: Target device (cuda/cpu)
            
        Returns:
            self: Model instance on target device
        """
        super().to(device)
        self.device = device
        return self

    def self_test(self, *, device: str = "cpu", n_trials: int = 16) -> Dict[str, Any]:
        """
        Quick sanity check to detect:
        - constant outputs
        - always-near-zero outputs
        - exploding temperature
        - collapsed latents (zH/zL near 0)
        """
        from stephanie.scoring.model.model_selftest import ModelSelfTest, summarize_selftest

        input_dim = int(getattr(self, "input_dim", 0) or getattr(self, "cfg", {}).get("input_dim", 0) or 0)
        if input_dim <= 0:
            # fall back: infer from projector weight
            for name, p in self.named_parameters():
                if name.endswith("input_projector.project.weight") and p.ndim == 2:
                    input_dim = int(p.shape[1])
                    break

        def build_inputs():
            B = 8
            x = torch.randn(B, input_dim)
            return {"x": x, "return_aux": True}

        def extract_debug(aux: Any):
            # aux is dict in your HRM
            if not isinstance(aux, dict):
                return {}
            out = {}
            for k in ["score_logit", "temp01", "zH_mag", "zL_mag", "entropy", "energy"]:
                if k in aux:
                    out[k] = aux[k]
            # also include any final latent if present
            for k in ["zH_final", "zL_final"]:
                if k in aux:
                    out[k] = aux[k]
            return out

        tester = ModelSelfTest(
            name="HRMModel",
            build_inputs=build_inputs,
            extract_debug=extract_debug,
            device=device,
            n_trials=n_trials,
        )
        res = tester.run(self)
        return {"ok": res.ok, "summary": summarize_selftest(res), "details": res.details}
