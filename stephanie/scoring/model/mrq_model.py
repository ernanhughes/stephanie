# stephanie/models/mrq_model.py

import torch

class MRQModel:
    def __init__(self, encoder, predictor, device="cpu"):
        self.encoder = encoder.to(device)
        self.predictor = predictor.to(device)
        self.device = device

    def predict(self, prompt_text: str, response_text: str, embedding_store) -> float:
        prompt_emb = torch.tensor(
            embedding_store.get_or_create(prompt_text), device=self.device
        ).unsqueeze(0)
        response_emb = torch.tensor(
            embedding_store.get_or_create(response_text), device=self.device
        ).unsqueeze(0)

        zsa = self.encoder(prompt_emb, response_emb)
        value = self.predictor(zsa).item()
        return value

    def load_weights(self, encoder_path: str, predictor_path: str):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        self.predictor.load_state_dict(torch.load(predictor_path, map_location=self.device))
        self.encoder.eval()
        self.predictor.eval()
