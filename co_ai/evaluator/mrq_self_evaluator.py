import torch
from co_ai.evaluator.text_encoder import TextEncoder
from co_ai.evaluator.hypothesis_value_predictor import HypothesisValuePredictor
from co_ai.models.sharpening_prediction import SharpeningPrediction
from dataclasses import asdict

class MRQSelfEvaluator:
    def __init__(self, memory, logger, device="cpu"):
        self.device = device
        self.memory = memory  # memory provides get_embedding
        self.logger = logger
        self.encoder = TextEncoder().to(self.device)
        self.value_predictor = HypothesisValuePredictor(512, 1024).to(self.device)

    def evaluate(self, goal, prompt, output_a, output_b):
        prompt_emb = torch.tensor(
            self.memory.embedding.get_or_create(prompt), device=self.device
        ).unsqueeze(0)
        output_a_emb = torch.tensor(
            self.memory.embedding.get_or_create(output_a), device=self.device
        ).unsqueeze(0)
        output_b_emb = torch.tensor(
            self.memory.embedding.get_or_create(output_b), device=self.device
        ).unsqueeze(0)

        zsa_a = self.encoder(prompt_emb, output_a_emb)
        zsa_b = self.encoder(prompt_emb, output_b_emb)

        value_a = self.value_predictor(zsa_a).item()
        value_b = self.value_predictor(zsa_b).item()

        preferred_output = output_a if value_a >= value_b else output_b
        scores = {"value_a": value_a, "value_b": value_b}

        if self.memory.mrq.log_evaluations():
            prediction = SharpeningPrediction(
                id=None,
                goal_id=-1,
                prompt_text=prompt,
                output_a=output_a,
                output_b=output_b,
                preferred="a" if value_a >= value_b else "b",
                predicted="a" if value_a >= value_b else "b",
                value_a=value_a,
                value_b=value_b,
            )

            self.memory.mrq.insert_sharpening_prediction(asdict(prediction), goal)

        return preferred_output, scores

    def train_from_database(self, goal:str, cfg:dict):
        if self.memory is None:
            raise ValueError("Database connection not provided.")

        limit = cfg.get("limit", 1000)
        epochs = cfg.get("epochs", 20)
        lr = cfg.get("lr", 1e-4)
        patience = cfg.get("patience", 3)
        min_delta = cfg.get("min_delta", 0.0001)

        self.logger.log("MRQTrainingStart", {
            "goal": goal,
            "limit": limit,
            "epochs": epochs,
            "learning_rate": lr,
            "patience": patience,
            "min_delta": min_delta
        })

        samples = self.memory.mrq.get_training_pairs(goal=goal, limit=limit)
        if not samples or len(samples) == 0:
            self.logger.log("MRQTrainingError", {
                "error": "No training samples found for the given goal.",
                "goal": goal
            })
            print("[ERROR] No training samples found. Cannot train MR.Q evaluator.")
            return  # Exit gracefully
        else:
            self.logger.log("MRQTraining", {
                "Sample count": len(samples),
                "goal": goal
            })


        inputs, labels = [], []
        for item in samples:
            prompt_emb = self.memory.embedding.get_or_create(item["prompt"])
            output_a_emb = self.memory.embedding.get_or_create(item["output_a"])
            output_b_emb = self.memory.embedding.get_or_create(item["output_b"])
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
            inputs.append(diff.squeeze(0).detach())
            labels.append(torch.tensor([1.0], device=self.device))

        dataset = torch.utils.data.TensorDataset(
            torch.stack(inputs), torch.stack(labels)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        opt = torch.optim.Adam(self.value_predictor.parameters(), lr=lr)

        self.value_predictor.train()

        best_loss = float("inf") # early stopping
        epochs_no_improve = 0
        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch, y_batch in dataloader:
                preds = self.value_predictor(x_batch)
                loss = -torch.log(torch.sigmoid(preds)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.logger.log(
                "MRQTrainingEpoch",
                {"epoch": epoch + 1, "avg_loss": round(avg_loss, 5), "goal": goal},
            )
            # 🛑 Early stopping logic
            if best_loss - avg_loss > min_delta:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    self.logger.log(
                        "MRQEarlyStopping",
                        {
                            "stopped_epoch": epoch + 1,
                            "best_loss": round(best_loss, 5),
                            "goal": goal,
                        },
                    )
                    break

        self.logger.log("MRQTrainingComplete", {
            "goal": goal,
            "epochs": epochs
        })

    def retrieve_similar(self, prompt, k=3):
        """Fetch top-k similar (prompt, output, reward) tuples from memory."""
        return self.memory.prompt.get_similar(prompt, k)
