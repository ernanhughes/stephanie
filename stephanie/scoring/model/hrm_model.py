# stephanie/models/hrm/hrm_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """RMSNorm layer, as used in the HRM paper."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RecurrentBlock(nn.Module):
    """
    A recurrent block (GRU-based) for L and H modules.
    Follows Post-Norm architecture with RMSNorm.
    """
    def __init__(self, input_dim, hidden_dim, name="RecurrentBlock"):
        super().__init__()
        self.name = name
        # Using GRU cell for simplicity and efficiency
        # Can be made configurable (e.g., LSTMCell)
        self.rnn_cell = nn.GRUCell(input_dim, hidden_dim)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, z_prev, input_combined):
        """
        Args:
            z_prev: Previous hidden state (B, hidden_dim)
            input_combined: Combined input for this step (B, input_dim)
                            This could be x_tilde, zH, zL depending on the module.
        Returns:
            z_next: Next hidden state (B, hidden_dim)
        """
        z_next = self.rnn_cell(input_combined, z_prev)
        z_next = self.norm(z_next)
        return z_next

    def init_state(self, batch_size, hidden_dim, device):
        """Initialize hidden state."""
        return torch.zeros(batch_size, hidden_dim, device=device)


class InputProjector(nn.Module):
    """Projects input embeddings to the HRM's internal state dimension."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, input_dim) - e.g., concatenated ctx_emb and doc_emb
        Returns:
            x_tilde: Projected and normalized input (B, hidden_dim)
        """
        x_proj = self.project(x)
        x_tilde = self.norm(x_proj)
        return x_tilde

class OutputProjector(nn.Module):
    """Projects the final high-level state to the output dimension."""
    def __init__(self, h_dim, output_dim):
        super().__init__()
        self.project = nn.Linear(h_dim, output_dim)

    def forward(self, zH_final):
        """
        Args:
            zH_final: Final high-level state (B, h_dim)
        Returns:
            y_hat: Output prediction (B, output_dim)
        """
        return self.project(zH_final)


class HRMModel(nn.Module):
    """
    Hierarchical Reasoning Model.
    Implements the core loop with L and H modules.
    Follows the fixed-depth approach for now (no ACT).
    """
    def __init__(self, cfg, logger=None):
        super().__init__()
        self.logger = logger
        self.input_dim = cfg.get("hrm.input_dim", 1536) # e.g., 768 (ctx) + 768 (doc)
        self.h_dim = cfg.get("hrm.h_dim", 256)
        self.l_dim = cfg.get("hrm.l_dim", 128)
        self.output_dim = cfg.get("hrm.output_dim", 1) # e.g., 1 for score prediction
        self.n_cycles = cfg.get("hrm.n_cycles", 4) # N
        self.t_steps = cfg.get("hrm.t_steps", 4)   # T
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Input Network (fI)
        self.input_projector = InputProjector(self.input_dim, self.h_dim) # Project to H dim
        
        # 2. Low-Level Module (fL)
        # Input to L: x_tilde (h_dim) + zH (h_dim) -> concatenated to 2*h_dim
        # (Paper suggests interactions, we simplify to concat for now)
        self.l_module = RecurrentBlock(2 * self.h_dim, self.l_dim, name="LModule")
        
        # 3. High-Level Module (fH)
        # Input to H: zL_final (l_dim) + zH_prev (h_dim) -> concatenated to l_dim + h_dim
        self.h_module = RecurrentBlock(self.l_dim + self.h_dim, self.h_dim, name="HModule")
        
        # 4. Output Network (fO)
        self.output_projector = OutputProjector(self.h_dim, self.output_dim)

    def forward(self, x):
        """
        Forward pass of the HRM model.
        Args:
            x: Input tensor (B, input_dim)
        Returns:
            y_hat: Output prediction (B, output_dim)
            intermediate_states: Optional dict for logging/debugging (can be extended)
        """
        batch_size = x.size(0)
        
        # 1. Project input
        x_tilde = self.input_projector(x) # (B, h_dim)

        # 2. Initialize states
        zL = self.l_module.init_state(batch_size, self.l_dim, self.device) # (B, l_dim)
        zH = self.h_module.init_state(batch_size, self.h_dim, self.device) # (B, h_dim)

        # 3. Nested Loop: N cycles of T steps
        for n in range(self.n_cycles):
            # Run T low-level steps within this cycle
            for t in range(self.t_steps):
                # Prepare input for L: [x_tilde, zH_prev]
                l_input = torch.cat([x_tilde, zH], dim=-1) # (B, h_dim + h_dim)
                zL = self.l_module(zL, l_input) # (B, l_dim)
            
            # After T steps, update H module
            # Prepare input for H: [zL_final_of_cycle, zH_prev]
            h_input = torch.cat([zL, zH], dim=-1) # (B, l_dim + h_dim)
            zH = self.h_module(zH, h_input) # (B, h_dim)
            # Note: L state (zL) is NOT reset per cycle in this simple version,
            #       but could be if needed. Paper implies reset or re-init.

        # 4. Final Output
        y_hat = self.output_projector(zH) # (B, output_dim)
        
        # Optional: Return intermediate states for deep supervision or analysis
        intermediate_states = {'zL_final': zL, 'zH_final': zH}
        
        return y_hat, intermediate_states

    def to(self, device):
        """Override to move submodules."""
        super().to(device)
        self.device = device
        return self
