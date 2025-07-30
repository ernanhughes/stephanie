import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Normalizes across features while preserving scale via a learned weight.
    Used throughout HRM instead of LayerNorm.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        print(f"RMSNorm initialized with dim={dim}, eps={eps}")

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RecurrentBlock(nn.Module):
    """
    A recurrent update block used by both L and H modules.
    Internally uses a GRUCell + RMSNorm for stable updates.
    """
    def __init__(self, input_dim, hidden_dim, name="RecurrentBlock"):
        super().__init__()
        self.name = name
        self.rnn_cell = nn.GRUCell(input_dim, hidden_dim)
        self.norm = RMSNorm(hidden_dim)
        print(f"{self.name} initialized with input_dim={input_dim}, hidden_dim={hidden_dim}")

    def forward(self, z_prev, input_combined):
        """
        Forward step of the RNN.
        - z_prev: previous hidden state (B, hidden_dim)
        - input_combined: input at this step (B, input_dim)
        Returns: next hidden state (B, hidden_dim)
        """
        z_next = self.rnn_cell(input_combined, z_prev)
        z_next = self.norm(z_next)
        return z_next

    def init_state(self, batch_size, hidden_dim, device):
        """Returns a zero-initialized state."""
        return torch.zeros(batch_size, hidden_dim, device=device)

class InputProjector(nn.Module):
    """
    Projects the input embedding into the HRM hidden space.
    This is the 'x_tilde' used throughout reasoning.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        self.norm = RMSNorm(hidden_dim)
        print(f"InputProjector initialized with input_dim={input_dim}, hidden_dim={hidden_dim}")

    def forward(self, x):
        x_proj = self.project(x)
        x_tilde = self.norm(x_proj)
        return x_tilde

class OutputProjector(nn.Module):
    """
    Projects the final high-level hidden state (zH) to the output space.
    For HRM this is typically a scalar quality score.
    """
    def __init__(self, h_dim, output_dim):
        super().__init__()
        self.project = nn.Linear(h_dim, output_dim)
        print(f"OutputProjector initialized with h_dim={h_dim}, output_dim={output_dim}")

    def forward(self, zH_final):
        return self.project(zH_final)

class HRMModel(nn.Module):
    """
    Hierarchical Reasoning Model (HRM)

    Models layered reasoning using two coupled RNNs:
    - Low-level module (L): simulates fine-grained steps (e.g. CoT steps)
    - High-level module (H): aggregates abstract strategic updates

    The model processes reasoning traces through N nested cycles,
    each composed of T low-level updates and a single high-level update.
    """
    def __init__(self, cfg, logger=None):
        super().__init__()
        self.logger = logger

        # Model hyperparameters from config
        self.input_dim = cfg.get("input_dim", 2048)
        self.h_dim = cfg.get("h_dim", 256)
        self.l_dim = cfg.get("l_dim", 128)
        self.output_dim = cfg.get("output_dim", 1)
        self.n_cycles = cfg.get("n_cycles", 4)  # Outer loop depth
        self.t_steps = cfg.get("t_steps", 4)    # Inner loop steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Input projection network
        self.input_projector = InputProjector(self.input_dim, self.h_dim)

        # Low-level module (L): operates on [x_tilde, zH] → updates zL
        self.l_module = RecurrentBlock(2 * self.h_dim, self.l_dim, name="LModule")

        # High-level module (H): operates on [zL, zH] → updates zH
        self.h_module = RecurrentBlock(self.l_dim + self.h_dim, self.h_dim, name="HModule")

        # Output layer from final zH
        self.output_projector = OutputProjector(self.h_dim, self.output_dim)
        print(f"HRMModel initialized with input_dim={self.input_dim}, h_dim={self.h_dim}, l_dim={self.l_dim}, output_dim={self.output_dim}, n_cycles={self.n_cycles}, t_steps={self.t_steps}")

    def forward(self, x):
        """
        Executes the full HRM reasoning process.

        Args:
            x: Input tensor of shape (B, input_dim) — typically a plan embedding
        Returns:
            y_hat: Final prediction (B, output_dim)
            intermediate_states: Final zL and zH for optional introspection
        """
        batch_size = x.size(0)

        # Project input into hidden reasoning space
        x_tilde = self.input_projector(x)  # (B, h_dim)

        # Initialize low-level and high-level memory states
        zL = self.l_module.init_state(batch_size, self.l_dim, self.device)
        zH = self.h_module.init_state(batch_size, self.h_dim, self.device)

        # N outer cycles (high-level reasoning updates)
        for n in range(self.n_cycles):
            # T low-level reasoning steps per cycle
            for t in range(self.t_steps):
                l_input = torch.cat([x_tilde, zH], dim=-1)  # (B, 2*h_dim)
                zL = self.l_module(zL, l_input)             # update zL

            # After T low-level steps, update high-level memory
            h_input = torch.cat([zL, zH], dim=-1)          # (B, l_dim + h_dim)
            zH = self.h_module(zH, h_input)                # update zH

        # Final prediction from abstract reasoning memory
        y_hat = self.output_projector(zH)                  # (B, output_dim)

        # Return prediction and final latent states (optional for training/debug)
        intermediate_states = {'zL_final': zL, 'zH_final': zH}
        return y_hat, intermediate_states

    def to(self, device):
        """
        Custom `.to()` to move internal state tracking.
        """
        super().to(device)
        self.device = device
        return self
