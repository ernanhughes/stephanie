import torch
from co_ai.evaluator.text_encoder import TextEncoder
from co_ai.evaluator.hypothesis_value_predictor import HypothesisValuePredictor


class MRQSelfEvaluator:
    def __init__(self, memory, device="cpu"):
        self.device = device
        self.memory = memory  # memory provides get_embedding
        self.encoder = TextEncoder().to(self.device)
        self.value_predictor = HypothesisValuePredictor(512, 1024).to(self.device)

    def evaluate(self, prompt, output_a, output_b):
        prompt_emb = torch.tensor(
            self.memory.get_embedding(prompt), device=self.device
        ).unsqueeze(0)
        output_a_emb = torch.tensor(
            self.memory.get_embedding(output_a), device=self.device
        ).unsqueeze(0)
        output_b_emb = torch.tensor(
            self.memory.get_embedding(output_b), device=self.device
        ).unsqueeze(0)

        zsa_a = self.encoder(prompt_emb, output_a_emb)
        zsa_b = self.encoder(prompt_emb, output_b_emb)

        value_a = self.value_predictor(zsa_a).item()
        value_b = self.value_predictor(zsa_b).item()

        preferred_output = output_a if value_a >= value_b else output_b
        scores = {"value_a": value_a, "value_b": value_b}

        return preferred_output, scores

    def train_from_database(
        self, goal: str, limit: int = 500, epochs: int = 5, lr: float = 1e-4
    ):
        if self.memory is None:
            raise ValueError("Database connection not provided.")

        samples = self.memory.prompt.get_mrq_training_pairs(goal=goal, limit=limit)

        inputs, labels = [], []
        for item in samples:
            prompt_emb = self.memory.get_embedding(item["prompt"])
            output_a_emb = self.memory.get_embedding(item["output_a"])
            output_b_emb = self.memory.get_embedding(item["output_b"])
            preferred = item["preferred"]

            zsa_a = self.encoder(
                torch.tensor(prompt_emb).unsqueeze(0).to(self.device),
                torch.tensor(output_a_emb).unsqueeze(0).to(self.device),
            )
            zsa_b = self.encoder(
                torch.tensor(prompt_emb).unsqueeze(0).to(self.device),
                torch.tensor(output_b_emb).unsqueeze(0).to(self.device),
            )

            diff = zsa_a - zsa_b if preferred == "a" else zsa_b - zsa_a
            inputs.append(diff.squeeze(0))
            labels.append(torch.tensor([1.0], device=self.device))

        dataset = torch.utils.data.TensorDataset(
            torch.stack(inputs), torch.stack(labels)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        opt = torch.optim.Adam(self.value_predictor.parameters(), lr=lr)

        self.value_predictor.train()
        for epoch in range(epochs):
            for x_batch, y_batch in dataloader:
                preds = self.value_predictor(x_batch)
                loss = -torch.log(torch.sigmoid(preds)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()


    def retrieve_similar(self, prompt, k=3):
        """Fetch top-k similar (prompt, output, reward) tuples from memory."""
        return self.memory.get_similar(prompt, k)