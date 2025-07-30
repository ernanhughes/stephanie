from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn


class EpistemicTraceEncoder(nn.Module):
    """
    A hybrid encoder that transforms a full PlanTrace (goal + steps + scores + final output)
    into a single latent vector for downstream HRM-style scoring.

    The final representation is used as input to models like the Hierarchical Reasoning Model (HRM).
    It fuses multiple modalities:
      - goal and output embeddings (from LLM or embedding model)
      - encoded step-wise reasoning traces
      - aggregate scoring statistics (Q/V/energy/etc.)
    """

    def __init__(self, cfg: Dict[str, any]):
        """
        Initialize the encoder architecture based on configurable hyperparameters.

        Args:
            cfg (dict): Config dictionary with keys:
                - embedding_dim: size of input text embeddings (default: 1024)
                - step_hidden_dim: output dim for encoded step traces
                - stats_input_dim: number of scalar stats per trace (e.g., Q/V/E)
                - stats_hidden_dim: MLP hidden dim for stats vector
                - final_dim: final encoded vector size
        """
        super().__init__()

        # Configuration with sensible defaults
        self.embedding_dim = cfg.get("embedding_dim", 1024)
        self.step_hidden_dim = cfg.get("step_hidden_dim", 64)
        self.stats_input_dim = cfg.get("stats_input_dim", 32)
        self.stats_hidden_dim = cfg.get("stats_hidden_dim", 128)
        self.final_dim = cfg.get("final_dim", 256)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("[EpistemicTraceEncoder] Config:")
        print(f"  - embedding_dim: {self.embedding_dim}")
        print(f"  - step_hidden_dim: {self.step_hidden_dim}")
        print(f"  - stats_input_dim: {self.stats_input_dim}")
        print(f"  - stats_hidden_dim: {self.stats_hidden_dim}")
        print(f"  - final_dim: {self.final_dim}")

        # 1. Step encoder: compress individual step embeddings into a latent vector
        self.step_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.step_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.step_hidden_dim, self.step_hidden_dim),
        ).to(self.device)

        # 2. Scoring statistics encoder: MLP for Q/V/Energy stats etc.
        self.stats_encoder = nn.Sequential(
            nn.Linear(self.stats_input_dim, self.stats_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.stats_hidden_dim, self.stats_hidden_dim),
        ).to(self.device)

        # 3. Final combiner: concatenate goal, final output, steps, stats
        combined_input_dim = 2 * self.embedding_dim + self.step_hidden_dim + self.stats_hidden_dim
        self.combiner = nn.Sequential(
            nn.Linear(combined_input_dim, self.final_dim),
            nn.ReLU(),
            nn.Linear(self.final_dim, self.final_dim)
        ).to(self.device)

    def forward(
        self,
        trace,
        embedding_lookup_fn: Callable[[str], torch.Tensor],
        score_stats_fn: Callable[[object, list], torch.Tensor],
        dimensions: list[str]
    ) -> torch.Tensor:
        """
        Encode a reasoning trace into a latent vector.

        Args:
            trace: PlanTrace object (or dict-like) with fields:
                - goal_text
                - final_output_text
                - execution_steps: list of ExecutionStep
            embedding_lookup_fn: callable that maps text → embedding tensor
            score_stats_fn: callable that returns numeric feature vector for scores
            dimensions: list of scoring dimensions (for stat extraction)

        Returns:
            torch.Tensor of shape [final_dim]
        """

        # -- Embed goal and final output text
        goal_emb = embedding_lookup_fn(trace.goal_text)
        final_emb = embedding_lookup_fn(trace.final_output_text)

        goal_emb = torch.as_tensor(goal_emb, dtype=torch.float32, device=self.device)
        final_emb = torch.as_tensor(final_emb, dtype=torch.float32, device=self.device)

        # -- Encode each step in the trace
        step_embeddings = []
        for step in trace.execution_steps:
            z_np = embedding_lookup_fn(step.output_text)
            z = torch.tensor(z_np, dtype=torch.float32, device=self.device) \
                if isinstance(z_np, np.ndarray) else z_np.to(self.device)

            step_encoded = self.step_encoder(z)  # shape: [step_hidden_dim]
            step_embeddings.append(step_encoded)

        # -- Aggregate step representations (mean pool)
        if step_embeddings:
            step_pooled = torch.mean(torch.stack(step_embeddings, dim=0), dim=0)
        else:
            step_pooled = torch.zeros(self.step_hidden_dim, device=self.device)

        # -- Get score stats (e.g., mean Q, max energy, etc.)
        stats_vector = score_stats_fn(trace, dimensions)  # shape: [stats_input_dim]
        stats_encoded = self.stats_encoder(stats_vector.to(self.device))

        # -- Concatenate all latent components
        combined = torch.cat([
            goal_emb,         # [embedding_dim]
            final_emb,        # [embedding_dim]
            step_pooled,      # [step_hidden_dim]
            stats_encoded     # [stats_hidden_dim]
        ], dim=-1)

        # -- Final projection to fixed-size trace representation
        z_trace = self.combiner(combined)  # shape: [final_dim]
        print(f"[EpistemicTraceEncoder] Encoded trace to shape: {z_trace.shape}")   
        return z_trace