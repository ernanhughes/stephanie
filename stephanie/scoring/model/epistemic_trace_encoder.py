import torch
import torch.nn as nn
import numpy as np
from typing import Callable

class EpistemicTraceEncoder(nn.Module):
    """
    A hybrid trace encoder combining:
    - goal & final output embeddings,
    - pooled step embeddings,
    - aggregated score statistics (e.g., Q, V, Energy, Uncertainty).
    """

    def __init__(self,
                 embedding_dim=1024,
                 step_hidden_dim=64,
                 stats_input_dim=12,
                 stats_hidden_dim=128,
                 final_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.step_hidden_dim = step_hidden_dim
        self.stats_input_dim = stats_input_dim
        self.stats_hidden_dim = stats_hidden_dim
        self.final_dim = final_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encode each step's textual embedding
        self.step_encoder = nn.Sequential(
            nn.Linear(embedding_dim, step_hidden_dim),
            nn.ReLU(),
            nn.Linear(step_hidden_dim, step_hidden_dim),
        ).to(self.device)

        # Encode statistical features across steps
        self.stats_encoder = nn.Sequential(
            nn.Linear(stats_input_dim, stats_hidden_dim),
            nn.ReLU(),
            nn.Linear(stats_hidden_dim, stats_hidden_dim),
        ).to(self.device)

        # Final fusion layer: combines everything
        combined_input_dim = 2 * embedding_dim + step_hidden_dim + stats_hidden_dim
        self.combiner = nn.Sequential(
            nn.Linear(combined_input_dim, final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, final_dim)
        ).to(self.device)

    def forward(self,
                trace,
                embedding_lookup_fn: Callable[[str], torch.Tensor],
                score_stats_fn: Callable[[object], torch.Tensor]) -> torch.Tensor:
        """
        Args:
            trace: PlanTrace instance or dict with:
                - goal_text
                - final_output_text
                - execution_steps: list of ExecutionStep
            embedding_lookup_fn: function(str) -> torch.Tensor[embedding_dim]
            score_stats_fn: function(trace) -> torch.Tensor[stats_input_dim]

        Returns:
            torch.Tensor of shape [final_dim]
        """

        # Step 1: Goal and final output embeddings
        goal_emb = embedding_lookup_fn(trace.goal_text)
        final_emb = embedding_lookup_fn(trace.final_output_text)

        goal_emb = torch.as_tensor(goal_emb, dtype=torch.float32).to(self.device)
        final_emb = torch.as_tensor(final_emb, dtype=torch.float32).to(self.device)

        # Step 2: Mean-pooled encoded step embeddings
        step_embeddings = []
        for step in trace.execution_steps:
            z_np = embedding_lookup_fn(step.output_text)
            z = torch.tensor(z_np, dtype=torch.float32) if isinstance(z_np, np.ndarray) else z_np
            step_encoded = self.step_encoder(z.to(self.device))  
            step_embeddings.append(step_encoded)

        if step_embeddings:
            step_pooled = torch.mean(torch.stack(step_embeddings, dim=0), dim=0)
        else:
            step_pooled = torch.zeros(self.step_hidden_dim)

        # Step 3: Score statistics
        stats_vector = score_stats_fn(trace)  # shape: [stats_input_dim]
        stats_encoded = self.stats_encoder(stats_vector.to(self.device))  # shape: [stats_hidden_dim]

        step_pooled = torch.as_tensor(step_pooled, dtype=torch.float32).to(self.device)
        stats_encoded = torch.as_tensor(stats_encoded, dtype=torch.float32).to(self.device)

        # Step 4: Combine all parts
        combined = torch.cat([
            goal_emb,
            final_emb,
            step_pooled,
            stats_encoded
        ], dim=-1)

        z_trace = self.combiner(combined)  # shape: [final_dim]
        return z_trace
