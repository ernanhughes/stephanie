# stephanie/agents/trainers/vpm_vit_trainer.py
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.model.vpm_vit import VPMViT


# Optional viz service (safe to no-op if missing)
def _get_viz(container):
    try:
        return container.get("vpm_visualization")
    except Exception:
        return None

# ---------------- Dataset ----------------
class VPMFrameDataset(Dataset):
    """
    Expects files under data_dir:
      frames/*.png or *.npy
      labels/*.json with {"reg":[...], "cls": int}
    If no labels found, will synthesize zeros.
    """
    def __init__(self, root: Path, in_ch: int = 1, patch: int = 8, mask_ratio: float = 0.15):
        self.root = Path(root)
        self.in_ch = in_ch
        self.patch = patch
        self.mask_ratio = float(mask_ratio)

        frame_dir = self.root / "frames"
        self.paths = sorted([p for p in frame_dir.glob("*.png")] + [p for p in frame_dir.glob("*.npy")])
        self.label_dir = self.root / "labels"

        if not self.paths:
            raise RuntimeError(f"No frames found under {frame_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def _load_frame(self, p: Path) -> np.ndarray:
        if p.suffix == ".npy":
            arr = np.load(p)
            # Expect (H,W) or (C,H,W)
            if arr.ndim == 2:
                arr = arr[None, ...]   # (1,H,W)
            elif arr.ndim == 3 and arr.shape[0] not in (1,3):
                # assume HWC
                arr = np.transpose(arr, (2,0,1))
            return arr.astype(np.float32)
        else:
            img = Image.open(p)
            if self.in_ch == 1:
                img = img.convert("L")
                arr = np.array(img, dtype=np.float32)[None, ...]  # (1,H,W)
            else:
                img = img.convert("RGB")
                arr = np.transpose(np.array(img, dtype=np.float32), (2,0,1))  # (3,H,W)
            return arr

    def _load_label(self, p: Path) -> Tuple[np.ndarray, int]:
        j = (self.label_dir / (p.stem + ".json"))
        if j.exists():
            try:
                obj = json.loads(j.read_text(encoding="utf-8"))
                reg = np.array(obj.get("reg", []), dtype=np.float32)
                cls = int(obj.get("cls", 0))
                return reg, cls
            except Exception:
                pass
        return np.zeros((5,), dtype=np.float32), 0

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        x = self._load_frame(p) / (255.0 if x_max(x:=None) else 1.0)  # see helper below

        C, H, W = x.shape
        # ensure channel count
        if C == 1 and self.in_ch == 3:
            x = np.repeat(x, 3, axis=0)
        elif C != self.in_ch:
            if self.in_ch == 1:
                x = x.mean(axis=0, keepdims=True)
            elif C == 3 and self.in_ch > 3:
                x = np.concatenate([x, np.zeros((self.in_ch-3, H, W), dtype=np.float32)], axis=0)
            else:
                x = np.repeat(x, self.in_ch // C + 1, axis=0)[:self.in_ch]

        reg, cls = self._load_label(p)
        # mask length N = (H/patch)*(W/patch)
        h, w = H // self.patch, W // self.patch
        N = h * w
        m = np.zeros((N,), dtype=bool)
        k = max(1, int(self.mask_ratio * N))
        # random masked indices
        idxs = np.random.choice(N, size=k, replace=False)
        m[idxs] = True

        return torch.from_numpy(x), torch.from_numpy(reg), torch.tensor(cls, dtype=torch.long), torch.from_numpy(m)

def x_max(x):  # tiny helper to branch 255-scale vs 0-1 inputs
    return None

# ---------------- Trainer ----------------
class VPMViTTrainer(BaseAgent):
    """
    Trains VPM-ViT on VPM frames with multi-task loss:
      - Regression on 5 dims
      - Risk class (3-way)
      - MPM (masked token reconstruction)
    """
    def __init__(self, cfg: DictConfig, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.model_cfg = cfg.model
        self.opt_cfg = cfg.optimizer
        self.sched_cfg = cfg.scheduler
        self.train_cfg = cfg.training

        # Build model
        self.model: VPMViT = VPMViT(**self.model_cfg.params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer
        opt_cls = self._resolve(self.opt_cfg.class)
        self.optimizer = opt_cls(self.model.parameters(), **self.opt_cfg.params)

        # Scheduler
        sched_cls = self._resolve(self.sched_cfg.class)
        self.scheduler = sched_cls(self.optimizer, **self.sched_cfg.params)

        # Losses
        self.reg_loss_fn = nn.SmoothL1Loss(beta=1.0)
        self.cls_loss_fn = nn.CrossEntropyLoss()

        # Optional viz
        self.vpm_viz = _get_viz(container)

    def _resolve(self, dotted: str):
        mod, name = dotted.rsplit(".", 1)
        return getattr(importlib.import_module(mod), name)

    def _dataloader(self, data_dir: Path) -> DataLoader:
        ds = VPMFrameDataset(
            data_dir,
            in_ch=self.model_cfg.params.in_ch,
            patch=self.model_cfg.params.patch,
            mask_ratio=self.train_cfg.get("mask_ratio", 0.15),
        )
        return DataLoader(ds, batch_size=self.train_cfg.batch_size, shuffle=True, num_workers=self.train_cfg.get("num_workers", 0))

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        data_dir = Path(context.get("data_dir", "data/vpm"))
        out_path = Path(context.get("model_save_path", "models/vpm_vit_final.pth"))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        loader = self._dataloader(data_dir)
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.model.train()
        for epoch in range(int(self.train_cfg.epochs)):
            ep_loss = 0.0
            reg_mae, cls_correct, cls_total = 0.0, 0, 0

            for vpm, reg_t, cls_t, mask in loader:
                vpm = vpm.to(self.device).float()
                reg_t = reg_t.to(self.device).float()
                cls_t = cls_t.to(self.device).long()
                mask = mask.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=True):
                    out = self.model(vpm, mask=mask)
                    loss = 0.0
                    if "reg" in out:
                        loss_reg = self.reg_loss_fn(out["reg"], reg_t)
                        loss += self.train_cfg.loss_weights.reg * loss_reg
                        reg_mae += torch.abs(out["reg"] - reg_t).mean().item()
                    if "cls" in out:
                        loss_cls = self.cls_loss_fn(out["cls"], cls_t)
                        loss += self.train_cfg.loss_weights.cls * loss_cls
                        cls_pred = out["cls"].argmax(dim=-1)
                        cls_correct += (cls_pred == cls_t).sum().item()
                        cls_total += cls_t.numel()
                    if "mpm_rec" in out:
                        # target tokens: (B,N,D) -> masked -> (M,D)
                        with torch.no_grad():
                            target_tok = self.model.patch_embed(vpm)  # (B,N,D)
                            target_tok = target_tok[mask]
                        loss_mpm = self.reg_loss_fn(out["mpm_rec"], target_tok)
                        loss += self.train_cfg.loss_weights.mpm * loss_mpm

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
                scaler.step(self.optimizer)
                scaler.update()

                ep_loss += float(loss.item())

            self.scheduler.step()
            n_batches = len(loader)
            avg_loss = ep_loss / max(1, n_batches)
            avg_mae = reg_mae / max(1, n_batches)
            acc = (cls_correct / max(1, cls_total)) if cls_total else 0.0

            # Log
            self.logger.log("VPMViTTrainingEpoch", {
                "epoch": epoch,
                "avg_loss": avg_loss,
                "reg_mae": avg_mae,
                "cls_acc": acc,
                "lr": float(self.scheduler.get_last_lr()[0]),
            })
            if self.vpm_viz:
                self.vpm_viz.track_metrics("vpm-vit-training", {
                    "epoch": epoch, "loss": avg_loss, "reg_mae": avg_mae, "cls_acc": acc,
                    "lr": float(self.scheduler.get_last_lr()[0]),
                })

        torch.save({
            "state_dict": self.model.state_dict(),
            "config": dict(self.model_cfg.params),
            "dims": context.get("dims", ["reasoning","knowledge","clarity","faithfulness","coverage"]),
            "risk_labels": context.get("risk_labels", ["OK","WATCH","RISK"]),
        }, out_path)

        return {"status": "completed", "final_model": str(out_path)}
