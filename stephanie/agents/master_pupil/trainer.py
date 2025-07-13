# stephanie/agents/master_pupil/trainer.py
import matplotlib.pyplot as plt
import numpy as np
import torch

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.master_pupil.finetuner import PupilFineTuner


class TrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None, master=None, pupil=None):
        super().__init__(cfg, memory, logger)
        self.master = master
        self.pupil = pupil
        self.embedding_store = memory.embedding
        self.finetuner = PupilFineTuner(
            input_dim=1024,  # embedding dim of pupil
            output_dim=1024,  # embedding dim of master
        )

    async def run(self, context: dict) -> dict:
        """Pipeline entrypoint (unused in this agent directly)."""
        return context

    def align_response(self, question, context=None, epochs=25, plot=True):
        master_answer = self.master.answer(question, context)
        self.logger.log("MasterAnswer", {"master_answer": master_answer})
        pupil_answer = self.pupil.answer(question, context)
        self.logger.log("PupilAnswer", {"pupil_answer": pupil_answer})

        master_emb = torch.tensor(
            self.embedding_store.get_or_create(master_answer), dtype=torch.float32
        )
        pupil_emb = torch.tensor(
            self.embedding_store.get_or_create(pupil_answer), dtype=torch.float32
        )

        losses = []
        print(f"Initial pupil answer:\n{pupil_answer}\n")

        for i in range(epochs):
            loss = self.finetuner.train_step(pupil_emb, master_emb)
            losses.append(loss)
            print(f"Epoch {i + 1} Loss: {loss:.4f}")

        if plot:
            self.plot_training_curve(losses)

        return pupil_answer

    def predict_embedding(self, text):
        emb = np.array(self.embedding_store.get_or_create(text))
        input_tensor = torch.tensor(emb, dtype=torch.float32)
        with torch.no_grad():
            aligned = self.finetuner.model(input_tensor).numpy()
        return aligned

    def _approximate_generation_from_embedding(self, emb: torch.Tensor) -> str:
        """
        Dummy method: in future, this could map embeddings back to text.
        """
        return " ".join(["generated"] * 10)

    def plot_training_curve(self, losses):
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(losses) + 1), losses, marker="o", linestyle="-")
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
