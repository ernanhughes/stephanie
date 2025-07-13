import torch
from typing import List


def srft_collate_fn(batch: List[dict], embedding_store, device):
    """
    Collate function for SRFT refinement training.

    Embeds:
        - context
        - original
        - refined

    Returns:
        - context_tensor: (B, D)
        - original_tensor: (B, D)
        - refined_tensor: (B, D)
        - original_scores: (B,)
        - refined_scores: (B,)
        - original_energy: (B,)
        - refined_energy: (B,)
        - llm_scores: (B,) or None
        - uncertainty: (B,)
    """
    contexts = []
    originals = []
    refineds = []
    orig_scores, ref_scores = [], []
    orig_energy, ref_energy = [], []
    llm_scores = []
    uncertainties = []

    for item in batch:
        contexts.append(torch.tensor(embedding_store.get_or_create(item["context"])))
        originals.append(torch.tensor(embedding_store.get_or_create(item["original"])))
        refineds.append(torch.tensor(embedding_store.get_or_create(item["refined"])))

        orig_scores.append(item["original_score"])
        ref_scores.append(item["refined_score"])
        orig_energy.append(item["original_energy"])
        ref_energy.append(item["refined_energy"])
        uncertainties.append(item.get("uncertainty", 0.5))
        llm_scores.append(item.get("llm_score", -1))  # use -1 for missing supervision

    return (
        torch.stack(contexts).to(device),
        torch.stack(originals).to(device),
        torch.stack(refineds).to(device),
        torch.tensor(orig_scores, dtype=torch.float32).to(device),
        torch.tensor(ref_scores, dtype=torch.float32).to(device),
        torch.tensor(orig_energy, dtype=torch.float32).to(device),
        torch.tensor(ref_energy, dtype=torch.float32).to(device),
        torch.tensor(llm_scores, dtype=torch.float32).to(device),
        torch.tensor(uncertainties, dtype=torch.float32).to(device),
    )
