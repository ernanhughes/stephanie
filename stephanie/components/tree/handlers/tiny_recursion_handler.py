# stephanie/components/tree/handlers/tiny_recursion_handler.py
"""
TinyRecursionHandler
--------------------
Executes recursive reasoning cycles using TinyRecursionModel.

Features:
• Loads model from config or memory.
• Runs multiple recursive passes with attention toggle.
• Returns logits, halting probability, and intermediate stability diagnostics.
• Integrates cleanly with TaskHandler via register_handler().
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from stephanie.model.tiny import TinyModel


class TinyRecursionHandler:
    def __init__(self, cfg, memory=None, logger=None):
        """
        Args:
            cfg: dict-like configuration (supports hydra-style keys)
            memory: optional Memory or EmbeddingStore reference
            logger: optional structured logger
        """
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize model
        self.model = TinyModel(
            d_model=cfg.get("d_model", 256),
            n_layers=cfg.get("n_layers", 2),
            n_recursions=cfg.get("n_recursions", 6),
            vocab_size=cfg.get("vocab_size", 1024),
            use_attention=cfg.get("use_attention", False),
            dropout=cfg.get("dropout", 0.1),
        ).to(self.device)

        self.model.eval()

    # ----------------------------------------------------------- #
    async def __call__(self, plan: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entry point for TaskHandler.
        Expects embeddings or latent vectors in context: x, y, z.
        """
        try:
            x, y, z = await self._prepare_inputs(context)
            logits, halt_p, z_new = self.model(x, y, z)

            # diagnostics
            halt_mean = float(halt_p.mean().item())
            z_shift = float(torch.norm(z_new - z, dim=-1).mean().item())
            conf = float(F.softmax(logits, dim=-1).max(dim=-1).values.mean().item())

            summary = (
                f"TinyRecursion step complete: halt_p={halt_mean:.3f}, "
                f"Δz={z_shift:.3f}, conf={conf:.3f}"
            )

            if self.logger:
                self.logger.log("TinyRecursionHandlerResult", {
                    "halt_p": halt_mean,
                    "z_shift": z_shift,
                    "confidence": conf,
                })

            return {
                "metric": conf,
                "summary": summary,
                "halt_p": halt_mean,
                "z_shift": z_shift,
                "merged_output": z_new.detach().cpu().tolist(),
                "is_bug": False,
            }

        except Exception as e:
            if self.logger:
                self.logger.log("TinyRecursionHandlerError", {"error": str(e)})
            return {"error": str(e), "metric": 0.0, "summary": "Recursion failed", "is_bug": True}

    # ----------------------------------------------------------- #
    async def _prepare_inputs(self, context: Dict[str, Any]):
        """
        Retrieve or embed x, y, z from context.
        Accepts:
            - raw text keys ("x_text", "y_text", "z_text")
            - precomputed tensors ("x", "y", "z")
        """
        def to_tensor(value):
            if isinstance(value, torch.Tensor):
                return value.to(self.device)
            if isinstance(value, (list, tuple)):
                return torch.tensor(value, dtype=torch.float, device=self.device)
            if isinstance(value, str) and self.memory:
                return torch.tensor(self.memory.embedding.get_or_create(value), dtype=torch.float, device=self.device)
            raise ValueError(f"Unsupported input type for TinyRecursion: {type(value)}")

        x = to_tensor(context.get("x") or context.get("x_text"))
        y = to_tensor(context.get("y") or context.get("y_text"))
        z = to_tensor(context.get("z") or context.get("z_text"))
        return x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)
