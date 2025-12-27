from __future__ import annotations

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq  # may vary per model

class HFVLMMixin:
    def _select_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_vlm(self, model_name: str):
        """
        Generic HF VLM loader.
        Note: some models might require AutoModelForCausalLM or another class,
        but the flow (processor + generate) is stable.
        """
        device = self._select_device()
        proc = AutoProcessor.from_pretrained(model_name)

        if device == "cuda":
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            ).to(device)
        else:
            model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)

        model.eval()
        return proc, model, device
