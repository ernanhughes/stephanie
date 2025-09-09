# stephanie/scoring/model/pacs_optimizer.py

from typing import Optional

import torch
from torch.optim import Optimizer


class PACSOptimizer(Optimizer):
    """
    Preconditioned Adaptive Control of Stochasticity (PACS) Optimizer.

    Implements a variance-reduced, preconditioned gradient descent update 
    suitable for training models in the PACS framework.

    This optimizer maintains both:
    - A moving average of past gradients (variance reduction)
    - A diagonal preconditioner (scaling step size per-parameter)

    Args:
        params (iterable): model parameters
        lr (float): base learning rate
        beta (float): momentum for gradient averaging (default 0.9)
        eps (float): small value to avoid division by zero (default 1e-8)
        weight_decay (float): L2 penalty (default 0.0)
        preconditioner_decay (float): decay for preconditioner (default 0.999)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        preconditioner_decay: float = 0.999,
    ):
        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            weight_decay=weight_decay,
            preconditioner_decay=preconditioner_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            preconditioner_decay = group["preconditioner_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient (variance reduction)
                    state["grad_avg"] = torch.zeros_like(p.data)
                    # Preconditioner accumulator (like RMSprop/Adam second moment)
                    state["precond"] = torch.zeros_like(p.data)

                grad_avg = state["grad_avg"]
                precond = state["precond"]

                state["step"] += 1

                # Update moving average of gradient
                grad_avg.mul_(beta).add_(grad, alpha=1 - beta)

                # Update preconditioner (running avg of squared grads)
                precond.mul_(preconditioner_decay).addcmul_(
                    grad, grad, value=1 - preconditioner_decay
                )

                # Compute preconditioned gradient
                denom = precond.sqrt().add_(eps)
                step = grad_avg / denom

                # Update parameters
                p.data.add_(step, alpha=-lr)

        return loss
