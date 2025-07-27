import torch
import torch.nn.functional as F


def expectile_loss(pred, target, expectile=0.7):
    diff = pred - target.detach()
    return torch.where(
        diff > 0,
        expectile * diff ** 2,
        (1 - expectile) * diff ** 2
    ).mean()

def advantage_weighted_regression(logits, advantage, beta=1.0):
    advantage = advantage.detach()
    weights = torch.exp(beta * advantage).unsqueeze(-1)
    return -(F.log_softmax(logits, dim=-1) * weights).mean()