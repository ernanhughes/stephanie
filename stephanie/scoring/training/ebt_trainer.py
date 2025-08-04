
import torch
import torch.nn.functional as F

from stephanie.scoring.training.base_trainer import BaseTrainer


class EBTTrainer(BaseTrainer):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_type = "ebt"
        self.num_actions = 3 #cfg.get("num_actions", 3)

    def train(self, samples, dimension):
        dl = self._create_dataloader(samples)
        if not dl:
            return {"error": f"Insufficient samples for {dimension}"}

        from torch import nn
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        from stephanie.scoring.model.ebt_model import EBTModel

        model = EBTModel(self.dim, self.hdim, self.num_actions, self.device).to(self.device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.get("lr", 2e-5))
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        mse = nn.MSELoss()

        def expectile_loss(diff, tau=0.7):
            return (torch.where(diff > 0, tau * diff.pow(2), (1 - tau) * diff.pow(2))).mean()

        q_losses, v_losses, pi_losses, entropies = [], [], [], []

        for epoch in range(self.cfg.get("epochs", 10)):
            total_q, total_v, total_pi = 0.0, 0.0, 0.0
            for ctx, doc, label in dl:
                ctx, doc, label = ctx.to(self.device), doc.to(self.device), label.to(self.device)
                outputs = model(ctx, doc)

                q_loss = mse(outputs["q_value"], label)
                v_loss = expectile_loss(outputs["q_value"].detach() - outputs["state_value"])
                adv = (outputs["q_value"] - outputs["state_value"]).detach()
                policy_probs = F.softmax(outputs["action_logits"], dim=-1)
                entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-8), dim=-1).mean()
                adv = adv.unsqueeze(1)  # Shape becomes [batch_size, 1]
                pi_loss = -(torch.log(policy_probs) * adv).mean() - 0.01 * entropy

                loss = q_loss * self.cfg.get("q_weight", 1.0) + \
                       v_loss * self.cfg.get("v_weight", 0.5) + \
                       pi_loss * self.cfg.get("pi_weight", 0.3)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_q += q_loss.item()
                total_v += v_loss.item()
                total_pi += pi_loss.item()
                entropies.append(entropy.item())

            avg_q = total_q / len(dl)
            avg_v = total_v / len(dl)
            avg_pi = total_pi / len(dl)
            scheduler.step(avg_q)

            self.log_event("EBTTrainerEpoch", {
                "dimension": dimension,
                "epoch": epoch + 1,
                "q_loss": avg_q,
                "v_loss": avg_v,
                "pi_loss": avg_pi,
                "policy_entropy": sum(entropies) / len(entropies)
            })

            q_losses.append(avg_q)
            v_losses.append(avg_v)
            pi_losses.append(avg_pi)

        self._save_model(model, dimension)

        return {
            "q_loss": q_losses[-1],
            "v_loss": v_losses[-1],
            "pi_loss": pi_losses[-1],
            "policy_entropy": sum(entropies) / len(entropies),
        }

    def _save_model(self, model, dimension):
        locator = self.get_locator(dimension)
        torch.save(model.encoder.state_dict(), locator.encoder_file())
        torch.save(model.q_head.state_dict(), locator.q_head_file())
        torch.save(model.v_head.state_dict(), locator.v_head_file())
        torch.save(model.pi_head.state_dict(), locator.pi_head_file())

        meta = {
            "dim": model.embedding_dim,
            "hdim": model.hidden_dim,
            "num_actions": model.num_actions,
            "version": self.version,
            "min_value": self.cfg.get("min_value", 0),
            "max_value": self.cfg.get("max_value", 100),
        }
        self._save_meta_file(meta, dimension)
