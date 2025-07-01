from torch import nn


class HypothesisValuePredictor(nn.Module):
    """Predicts a quality score for a hypothesis given its embedding."""
    def __init__(self, zsa_dim=512, hdim=1024):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(zsa_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, 1)
        )

    def forward(self, zsa_embedding):
        assert len(zsa_embedding.shape) == 2, f"Expected 2D input, got {zsa_embedding.shape}"
        return self.value_net(zsa_embedding)
