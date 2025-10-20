# stephanie/scoring/model/mrq_model.py
from __future__ import annotations

import torch


class MRQModel:
    def __init__(self, encoder, predictor, embedding_store, device="cpu"):
        self.encoder = encoder.to(device)
        self.predictor = predictor.to(device)
        self.embedding_store = embedding_store
        self.device = device

    # --- NEW: make the model callable like a torch Module ---
    def __call__(self, ctx, doc, *, apply_sigmoid: bool = False, return_dict: bool = True):
        """
        Accepts ctx/doc embeddings as:
          - torch.Tensor [B, D] (preferred), or
          - numpy array / list (will be coerced).
        Returns:
          {"q_value": logits} by default (tensor [B]),
          or raw tensor if return_dict=False.
        If apply_sigmoid=True, q_value contains probabilities in [0,1].
        """
        ctx_t = torch.as_tensor(ctx, dtype=torch.float32, device=self.device)
        doc_t = torch.as_tensor(doc, dtype=torch.float32, device=self.device)
        if ctx_t.dim() == 1: ctx_t = ctx_t.unsqueeze(0)
        if doc_t.dim() == 1: doc_t = doc_t.unsqueeze(0)

        z = self.encoder(ctx_t, doc_t)              # [B, D']
        logits = self.predictor(z).view(-1)         # [B]
        out = torch.sigmoid(logits) if apply_sigmoid else logits
        return {"q_value": out} if return_dict else out

    # Optional: keep a PyTorch-style alias
    forward = __call__

    def predict(self, prompt_text: str, response_text: str, *, return_prob: bool = False) -> float:
        prompt_emb = torch.tensor(self.embedding_store.get_or_create(prompt_text),
                                  dtype=torch.float32, device=self.device).unsqueeze(0)
        response_emb = torch.tensor(self.embedding_store.get_or_create(response_text),
                                    dtype=torch.float32, device=self.device).unsqueeze(0)
        z = self.encoder(prompt_emb, response_emb)
        logit = self.predictor(z).view(-1)[0]
        return float(torch.sigmoid(logit) if return_prob else logit)

    def load_weights(self, encoder_path: str, predictor_path: str):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        self.predictor.load_state_dict(torch.load(predictor_path, map_location=self.device))
        self.encoder.eval(); self.predictor.eval()

    def train(self):
        self.encoder.train(); self.predictor.train()

    def eval(self):
        self.encoder.eval(); self.predictor.eval()
