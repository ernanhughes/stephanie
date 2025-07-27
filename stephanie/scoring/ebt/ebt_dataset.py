# stephanie/scoring/ebt/ebt_dataset.py
from torch.utils.data import Dataset


class EBTDataset(Dataset):
    def __init__(self, data, device="cpu"):
        self.device = device
        self.data = [self._to_device(item) for item in data if self._is_valid(item)]
    
    def _is_valid(self, item):
        """Ensure item has valid data"""
        return (
            item.get("context_emb") is not None and 
            item.get("doc_emb") is not None and
            item.get("label") is not None
        )
    
    def _to_device(self, item):
        """Move item to device"""
        item["context_emb"] = item["context_emb"].to(self.device)
        item["doc_emb"] = item["doc_emb"].to(self.device)
        item["label"] = item["label"].to(self.device)
        if "expert_policy" in item:
            item["expert_policy"] = item["expert_policy"].to(self.device)
        return item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "context_emb": item["context_emb"],
            "doc_emb": item["doc_emb"],
            "label": item["label"],
            "expert_policy": item.get("expert_policy", [0.3, 0.7, 0.0])
        }