# stephanie/scoring/mrq/value_predictor.py
import logging

from torch import nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValuePredictor(nn.Module):
    """Predicts a quality score for a document given its contextual embedding."""

    def __init__(self, zsa_dim=4096, hdim=2048):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(zsa_dim, hdim), nn.ReLU(), nn.Linear(hdim, 1)
        )

    def forward(self, zsa_embedding):
        assert len(zsa_embedding.shape) == 2, (
            f"Expected 2D input, got {zsa_embedding.shape}"
        )
        return self.value_net(zsa_embedding)
