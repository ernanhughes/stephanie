# stephanie/scoring/model/energy_based_scorer.py
import torch
import torch.nn as nn


class EBTThinker:
    def __init__(self, model, step_size=0.05, steps=10):
        self.model = model
        self.step_size = step_size
        self.steps = steps

    def optimize(self, context_emb, candidate_emb):
        y = candidate_emb.clone().detach().requires_grad_(True)
        energies = []

        for _ in range(self.steps):
            energy = self.model(context_emb, y)
            grad = torch.autograd.grad(energy.sum(), y, create_graph=False)[0]
            y = y - self.step_size * grad
            energies.append(energy.item())

        final_energy = energies[-1]
        converged = abs(energies[-1] - energies[0]) < 0.01
        return {
            "refined_candidate": y.detach(),
            "energy": final_energy,
            "steps_used": len(energies),
            "converged": converged,
            "energy_trace": energies,
        }


class EnergyBasedScorer(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer  # Any pretrained or scratch Transformer

    def forward(self, context_emb, candidate_emb):
        combined = torch.cat([context_emb, candidate_emb], dim=-1)
        hidden = self.transformer(combined)
        energy = hidden.mean(dim=-1)  # scalar energy per example
        return energy
