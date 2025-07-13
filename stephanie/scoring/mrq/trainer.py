# stephanie\scoring\mrq\trainer.py
import torch
import torch.nn.functional as F


class MRQTrainer:
    def __init__(self, mrq_model, optimizer):
        self.model = mrq_model
        self.optimizer = optimizer
        self.model.train_mode()

    def update(self, goal: str, chunk: str, reward: float) -> float:
        self.model.train_mode()
        pred = self.model.predict(goal, chunk)
        pred_tensor = torch.tensor([[pred]], requires_grad=True)
        target_tensor = torch.tensor([[reward]], dtype=torch.float32)

        loss = F.mse_loss(pred_tensor, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
