# stephanie/scoring/ebt/srft_refinement_dataset.py

from torch.utils.data import Dataset
from typing import List, Dict


class SRFTRefinementDataset(Dataset):
    """
    Dataset for SRFT-style training:
    Includes original/refined embeddings + score/energy/uncertainty data
    """

    def __init__(self, examples: List[Dict]):
        """
        Args:
            examples: list of refinement examples with fields like:
                - context
                - original
                - refined
                - dimension
                - original_score
                - refined_score
                - original_energy
                - refined_energy
                - llm_score
                - uncertainty
        """
        self.data = []

        for ex in examples:
            self.data.append({
                "context": ex["context"],
                "original": ex["original"],
                "refined": ex["refined"],
                "dimension": ex["dimension"],
                "original_score": ex.get("original_score", 0.5),
                "refined_score": ex.get("refined_score", 0.6),
                "original_energy": ex.get("original_energy", 0.5),
                "refined_energy": ex.get("refined_energy", 0.4),
                "llm_score": ex.get("llm_score"),
                "uncertainty": ex.get("uncertainty", 0.5)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
