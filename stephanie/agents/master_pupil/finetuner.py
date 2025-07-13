# stephanie/agents/master_pupil/finetuner.py
import logging

import torch
import torch.nn as nn
import torch.optim as optim

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PupilModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        assert x.shape[-1] == self.input_dim, (
            f"Expected input dim {self.input_dim}, got {x.shape[-1]}"
        )
        return self.model(x)


class PupilFineTuner:
    def __init__(self, input_dim=1024, output_dim=1024, lr=1e-4):
        logger.info(
            f"Initializing PupilModel with input_dim={input_dim}, output_dim={output_dim}"
        )
        self.model = PupilModel(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, student_input, teacher_output):
        # Assertions
        assert student_input.shape[-1] == self.model.input_dim, (
            f"Student input has wrong shape: {student_input.shape[-1]} vs expected {self.model.input_dim}"
        )
        assert teacher_output.shape[-1] == self.model.output_dim, (
            f"Teacher output has wrong shape: {teacher_output.shape[-1]} vs expected {self.model.output_dim}"
        )

        self.model.train()
        pred = self.model(student_input)
        loss = self.loss_fn(pred, teacher_output)

        logger.info(f"Training step loss: {loss.item():.6f}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
