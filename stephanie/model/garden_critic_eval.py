# stephanie/model/garden_critic_eval.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import resnet18

# ----------------------------
#  Dataset
# ----------------------------

@dataclass
class GardenSample:
    run_id: str
    label: int            # 0 = baseline, 1 = improved
    vpm_img: np.ndarray   # HxW or HxWxC
    metrics: np.ndarray   # HxW


class GardenHealthDataset(Dataset):
    """
    Dataset for training a critic to distinguish improved vs baseline gardens.

    Expects:
        baseline_path: JSON mapping run_id -> { "vpm_img": ..., "metrics": ... }
        improved_path: same schema

    For each run_id present in both files, we create:
        - one sample with label=0 (baseline)
        - one sample with label=1 (improved)
    """

    def __init__(
        self,
        baseline_path: Path,
        improved_path: Path,
    ) -> None:
        baseline = json.loads(Path(baseline_path).read_text())
        improved = json.loads(Path(improved_path).read_text())

        run_ids = sorted(set(baseline.keys()) & set(improved.keys()))
        if not run_ids:
            raise ValueError("No overlapping run_ids between baseline and improved JSON files")

        self.samples: List[GardenSample] = []
        for run_id in run_ids:
            b = baseline[run_id]
            i = improved[run_id]

            b_img = np.asarray(b["vpm_img"], dtype=np.float32)
            b_metrics = np.asarray(b["metrics"], dtype=np.float32)
            i_img = np.asarray(i["vpm_img"], dtype=np.float32)
            i_metrics = np.asarray(i["metrics"], dtype=np.float32)

            self.samples.append(
                GardenSample(
                    run_id=run_id,
                    label=0,
                    vpm_img=b_img,
                    metrics=b_metrics,
                )
            )
            self.samples.append(
                GardenSample(
                    run_id=run_id,
                    label=1,
                    vpm_img=i_img,
                    metrics=i_metrics,
                )
            )

        # Precompute run_id -> global indices mapping for pairwise eval
        self.run_to_indices: Dict[str, List[int]] = {}
        for idx, s in enumerate(self.samples):
            self.run_to_indices.setdefault(s.run_id, []).append(idx)

        # Basic sanity check: exactly 2 samples per run
        for run_id, idxs in self.run_to_indices.items():
            if len(idxs) != 2:
                raise ValueError(f"Run {run_id} has {len(idxs)} samples (expected 2)")

        # Metric names are implicit in column order; we treat them as indices for now
        # If you want real names, you can pass them separately.
        example_metrics = self.samples[0].metrics
        if example_metrics.ndim != 2:
            raise ValueError(f"metrics must be HxW, got shape {example_metrics.shape}")
        self.num_metrics = example_metrics.shape[1]
        self.metric_names = [f"m_{j}" for j in range(self.num_metrics)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        # VPM image -> tensor (C, H, W)
        img = torch.from_numpy(s.vpm_img)  # HxW or HxWxC
        if img.ndim == 2:
            img = img.unsqueeze(0)  # 1, H, W
        elif img.ndim == 3:
            # Assume HxWxC
            img = img.permute(2, 0, 1)  # C, H, W
        else:
            raise ValueError(f"Unexpected vpm_img shape {s.vpm_img.shape}")

        # Ensure 3 channels (for ResNet)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {img.shape[0]}")

        # Metrics -> tensor (H, W)
        metrics = torch.from_numpy(s.metrics)  # HxW

        # Label -> float (for BCEWithLogitsLoss)
        label = torch.tensor(float(s.label), dtype=torch.float32)

        return {
            "run_id": s.run_id,
            "vpm_img": img,
            "metrics": metrics,
            "label": label,
        }


def train_test_split_by_run(
    dataset: GardenHealthDataset,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[Subset, Subset, List[str]]:
    """
    Split by run_id so that each baseline/improved pair stays in the same split.
    Returns:
        train_subset, test_subset, test_run_ids
    """
    rng = np.random.RandomState(seed)
    all_run_ids = sorted(dataset.run_to_indices.keys())
    rng.shuffle(all_run_ids)

    n_test = max(1, int(len(all_run_ids) * test_size))
    test_run_ids = sorted(all_run_ids[:n_test])
    train_run_ids = sorted(all_run_ids[n_test:])

    train_indices: List[int] = []
    test_indices: List[int] = []

    for rid in train_run_ids:
        train_indices.extend(dataset.run_to_indices[rid])
    for rid in test_run_ids:
        test_indices.extend(dataset.run_to_indices[rid])

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    return train_subset, test_subset, test_run_ids


# ----------------------------
#  Model
# ----------------------------

class GardenHealthCritic(nn.Module):
    """
    Fused critic with toggles for image-only / metrics-only / both.

    - If use_image=True, use img_encoder branch.
    - If use_metrics=True, use metrics branch.
    - If both, fuse by concatenation.

    Inputs:
        vpm_img:    (B, 3, H, W)
        metrics:    (B, H, W)
    Output:
        logits:     (B,)
    """

    def __init__(
        self,
        metric_names: List[str],
        img_encoder: nn.Module | None = None,
        use_image: bool = True,
        use_metrics: bool = True,
        d_struct: int = 64,
        d_fuse: int = 128,
    ) -> None:
        super().__init__()

        if not use_image and not use_metrics:
            raise ValueError("At least one of use_image or use_metrics must be True")

        self.metric_names = metric_names
        self.num_metrics = len(metric_names)
        self.use_image = use_image
        self.use_metrics = use_metrics

        # Visual branch
        self.img_encoder = img_encoder if use_image else None
        if self.img_encoder is not None:
            # For ResNet, we replace fc with Identity so we get feature vectors
            if hasattr(self.img_encoder, "fc") and isinstance(self.img_encoder.fc, nn.Linear):
                out_dim = self.img_encoder.fc.in_features
                self.img_encoder.fc = nn.Identity()
                self.img_out_dim = out_dim
            else:
                raise ValueError("img_encoder must be a ResNet-like model with .fc attribute")

            self.img_proj = nn.Linear(self.img_out_dim, d_fuse)

        # Structured branch
        if use_metrics:
            self.name_embed = nn.Embedding(self.num_metrics, d_struct // 2)
            self.value_proj = nn.Linear(1, d_struct // 2)
            self.struct_proj = nn.Linear(d_struct, d_fuse)

        # Fusion output dim
        fuse_in_dim = 0
        if use_image:
            fuse_in_dim += d_fuse
        if use_metrics:
            fuse_in_dim += d_fuse

        self.fuse = nn.Sequential(
            nn.Linear(fuse_in_dim, d_fuse),
            nn.ReLU(),
            nn.Linear(d_fuse, 1),
        )

    def forward(self, vpm_img: torch.Tensor, metrics: torch.Tensor) -> torch.Tensor:
        """
        vpm_img: (B, 3, H, W)
        metrics: (B, H, W)
        """
        feats: List[torch.Tensor] = []

        if self.use_image:
            img_feat = self.img_encoder(vpm_img)          # (B, img_out_dim)
            img_feat = self.img_proj(img_feat)           # (B, d_fuse)
            feats.append(img_feat)

        if self.use_metrics:
            B, H, W = metrics.shape
            if W != self.num_metrics:
                raise ValueError(
                    f"Metric grid width {W} != num_metrics {self.num_metrics}; "
                    "ensure metrics are HxW with W == len(metric_names)."
                )

            # Metric name embeddings: (W, d_struct/2)
            device = metrics.device
            name_ids = torch.arange(self.num_metrics, device=device, dtype=torch.long)
            name_emb = self.name_embed(name_ids)         # (W, d_struct/2)
            name_emb = name_emb.unsqueeze(0).unsqueeze(0).expand(B, H, W, -1)

            # Metric values: (B, H, W, 1) -> (B, H, W, d_struct/2)
            values = metrics.unsqueeze(-1)               # (B, H, W, 1)
            value_emb = self.value_proj(values)          # (B, H, W, d_struct/2)

            struct_grid = torch.cat([value_emb, name_emb], dim=-1)  # (B, H, W, d_struct)
            struct_feat = struct_grid.mean(dim=1).mean(dim=1)       # (B, d_struct)
            struct_feat = self.struct_proj(struct_feat)             # (B, d_fuse)
            feats.append(struct_feat)

        if len(feats) == 1:
            fused = feats[0]
        else:
            fused = torch.cat(feats, dim=-1)

        logits = self.fuse(fused).view(-1)  # (B,)
        return logits


# ----------------------------
#  Evaluation Metrics
# ----------------------------

def expected_calibration_error(probs, labels, n_bins=10):
    """Compute Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(probs >= bin_lower, probs < bin_upper)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(labels[in_bin])
            bin_conf = np.mean(probs[in_bin])
        else:
            bin_acc = 0
            bin_conf = 0
        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)
        bin_sizes.append(bin_size)
    
    ece = 0
    for i in range(len(bin_sizes)):
        ece += bin_sizes[i] * np.abs(bin_accs[i] - bin_confs[i])
    ece /= len(probs)
    
    return ece


def compute_pairwise_accuracy(probs, labels, run_ids):
    """Compute pairwise accuracy for baseline vs improved pairs"""
    # Build run -> list of (label, prob)
    run_pairs: Dict[str, List[Tuple[int, float]]] = {}
    for run_id, label, prob in zip(run_ids, labels, probs):
        run_pairs.setdefault(run_id, []).append((label, prob))
    
    correct_pairs = 0
    total_pairs = 0
    for run_id, items in run_pairs.items():
        if len(items) != 2:
            continue
        total_pairs += 1
        
        # Find which item is the improved sample (label=1)
        improved_item = None
        baseline_item = None
        for item in items:
            if item[0] == 1:
                improved_item = item
            else:
                baseline_item = item
        
        if improved_item is not None and baseline_item is not None:
            if improved_item[1] > baseline_item[1]:
                correct_pairs += 1
    
    return correct_pairs / max(1, total_pairs)


def compute_metrics(probs, labels, run_ids):
    """Compute all evaluation metrics"""
    # AUC
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    
    # Pairwise accuracy
    pairwise_acc = compute_pairwise_accuracy(probs, labels, run_ids)
    
    # ECE
    ece = expected_calibration_error(probs, labels)
    
    # Brier Score
    brier = brier_score_loss(labels, probs)
    
    # Accuracy
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    
    return {
        "auc": auc,
        "pairwise_acc": pairwise_acc,
        "ece": ece,
        "brier": brier,
        "accuracy": acc
    }


def plot_calibration_curve(probs, labels, n_bins=10, save_path=None):
    """Plot calibration curve and save if save_path is provided"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(probs >= bin_lower, probs < bin_upper)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(labels[in_bin])
            bin_conf = np.mean(probs[in_bin])
        else:
            bin_acc = 0
            bin_conf = 0
        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)
        bin_sizes.append(bin_size)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    plt.plot(bin_confs, bin_accs, "s-", label="Model")
    
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# ----------------------------
#  Training / Evaluation
# ----------------------------

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 3e-4,
    save_model: Path | None = None,
) -> Dict[str, float]:
    """
    Train a single critic model and return metrics.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # ---- Train ----
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0

        for batch in train_loader:
            vpm = batch["vpm_img"].to(device)           # (B, 3, H, W)
            metrics = batch["metrics"].to(device)       # (B, H, W)
            labels = batch["label"].to(device)          # (B,)

            logits = model(vpm, metrics)                # (B,)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * vpm.size(0)
            n += vpm.size(0)

        avg_loss = running_loss / max(1, n)
        print(f"[{name}] Epoch {epoch}/{epochs} - loss={avg_loss:.4f}")

    # ---- Evaluate ----
    model.eval()
    all_logits: List[float] = []
    all_labels: List[int] = []
    all_run_ids: List[str] = []
    all_global_indices: List[int] = []  # reference back to base dataset

    # test_loader.dataset is a Subset
    subset: Subset = test_loader.dataset
    base_dataset: GardenHealthDataset = subset.dataset  # type: ignore[assignment]
    subset_indices: List[int] = subset.indices          # type: ignore[assignment]

    with torch.no_grad():
        offset = 0
        for batch_idx, batch in enumerate(test_loader):
            vpm = batch["vpm_img"].to(device)
            metrics = batch["metrics"].to(device)
            labels = batch["label"].cpu().numpy()
            run_ids = batch["run_id"]

            logits = model(vpm, metrics).cpu().numpy()

            B = len(labels)
            all_logits.extend(logits.tolist())
            all_labels.extend(labels.astype(int).tolist())
            all_run_ids.extend(run_ids)

            # Map to global indices
            all_global_indices.extend(subset_indices[offset : offset + B])
            offset += B

    # Convert logits to probabilities
    probs = 1.0 / (1.0 + np.exp(-np.array(all_logits)))
    labels_arr = np.array(all_labels)
    
    # Compute metrics
    metrics = compute_metrics(probs, labels_arr, all_run_ids)
    
    # Save model if requested
    if save_model:
        save_path = save_model / f"{name}_model.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved {name} model to {save_path}")
        
        # Save calibration plot
        calib_plot_path = save_model / f"{name}_calibration.png"
        plot_calibration_curve(probs, labels_arr, save_path=calib_plot_path)
        print(f"Saved calibration curve to {calib_plot_path}")
    
    # Print metrics
    print(f"[{name}] AUC={metrics['auc']:.4f}, PairwiseAcc={metrics['pairwise_acc']:.4f}, "
          f"ECE={metrics['ece']:.4f}, Brier={metrics['brier']:.4f}, "
          f"Accuracy={metrics['accuracy']:.4f}")
    
    return metrics


def build_dataloaders(
    dataset: GardenHealthDataset,
    batch_size: int = 16,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    train_subset, test_subset, _ = train_test_split_by_run(dataset, test_size=test_size, seed=seed)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Garden Health Critic Evaluation")
    parser.add_argument(
        "--baseline_json",
        type=Path,
        default=Path("data/baseline_graph.json"),
        help="Path to baseline JSON",
    )
    parser.add_argument(
        "--improved_json",
        type=Path,
        default=Path("data/graph_improved.json"),
        help="Path to improved JSON",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs per model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--save_model",
        type=Path,
        default=None,
        help="Directory to save trained models and calibration plots",
    )
    args = parser.parse_args()

    set_seed(42)

    print("Loading dataset...")
    dataset = GardenHealthDataset(
        baseline_path=args.baseline_json,
        improved_path=args.improved_json,
    )
    
    print(f"Dataset loaded with {len(dataset)} samples ({len(dataset.run_to_indices)} runs)")
    print(f"Train/Test split: {int((1-0.2)*len(dataset.run_to_indices))} training runs, "
          f"{int(0.2*len(dataset.run_to_indices))} test runs")
    
    train_loader, test_loader = build_dataloaders(
        dataset,
        batch_size=args.batch_size,
        test_size=0.2,
        seed=42,
    )

    device = torch.device(args.device)
    metric_names = dataset.metric_names

    # --- Build three variants: metrics-only, image-only, fused ---
    results: Dict[str, Dict[str, float]] = {}

    # Metrics-only
    metrics_only_model = GardenHealthCritic(
        metric_names=metric_names,
        img_encoder=None,
        use_image=False,
        use_metrics=True,
    )
    print("\n=== Training metrics-only critic ===")
    results["metrics_only"] = train_one_model(
        name="metrics_only",
        model=metrics_only_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        save_model=args.save_model,
    )

    # Image-only
    img_encoder = resnet18(pretrained=False)
    image_only_model = GardenHealthCritic(
        metric_names=metric_names,
        img_encoder=img_encoder,
        use_image=True,
        use_metrics=False,
    )
    print("\n=== Training image-only critic ===")
    results["image_only"] = train_one_model(
        name="image_only",
        model=image_only_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        save_model=args.save_model,
    )

    # Fused
    img_encoder_fused = resnet18(pretrained=False)
    fused_model = GardenHealthCritic(
        metric_names=metric_names,
        img_encoder=img_encoder_fused,
        use_image=True,
        use_metrics=True,
    )
    print("\n=== Training fused critic (image + metrics) ===")
    results["fused"] = train_one_model(
        name="fused",
        model=fused_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        save_model=args.save_model,
    )

    # --- Print summary table ---
    print("\n================ Experiment Summary ================")
    print(f"{'Model':<15} {'AUC':>8} {'PairwiseAcc':>14} {'ECE':>8} {'Brier':>8} {'Accuracy':>10}")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['auc']:>8.4f} {metrics['pairwise_acc']:>14.4f} "
              f"{metrics['ece']:>8.4f} {metrics['brier']:>8.4f} {metrics['accuracy']:>10.4f}")
    print("===================================================")
    
    # Print conclusion
    if results["fused"]["auc"] > results["metrics_only"]["auc"] and \
       results["fused"]["auc"] > results["image_only"]["auc"]:
        print("\nConclusion: The fused model outperforms both single-modality models, "
              "demonstrating the value of combining visual and metric information.")
    else:
        print("\nConclusion: One single-modality model outperforms the fused model, "
              "suggesting potential issues with the fusion strategy.")


if __name__ == "__main__":
    main()