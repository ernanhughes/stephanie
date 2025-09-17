# stephanie/agents/ebt/thinker.py
import torch


class EBTThinker:
    def __init__(self, model, step_size=0.05, steps=10):
        self.model = model
        self.step_size = step_size
        self.steps = steps

    def optimize(self, context_emb, initial_candidate):
        y = initial_candidate.clone().detach().requires_grad_(True)
        for _ in range(self.steps):
            energy = self.model(context_emb, y)
            grad = torch.autograd.grad(energy.sum(), y, create_graph=False)[0]
            y = y - self.step_size * grad
        return y.detach(), self.model(context_emb, y).item()
