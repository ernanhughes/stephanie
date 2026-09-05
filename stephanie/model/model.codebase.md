# Codebase Pack: model

```text
ROOT: C:\Project\stephanie\stephanie\model
GENERATED_AT_UTC: 2026-07-04T11:51:40.018059+00:00
PART: 1/1
FILES_IN_PART: 18
TOTAL_LINES_IN_PART: 3803
TOTAL_BYTES_UTF8_IN_PART: 134196
MODE: configured include extensions
LINE_NUMBERS: True
MAX_FILE_KB: 400
```

## How to cite this pack in review

Use the stable file ID plus line numbers, for example:

```text
F0007 `services/replay_service.py` L0042-L0068
```

## File Index

| ID | Path | Lang | Lines | KB | SHA256 |
|---|---|---:|---:|---:|---|
| F0001 | `ebt.py` | python | 80 | 2.4 | `438fdb08e781` |
| F0002 | `epistemic_trace_encoder.py` | python | 150 | 6.1 | `78d4b32b97de` |
| F0003 | `garden_critic_eval.py` | python | 680 | 21.2 | `8a66c522bccc` |
| F0004 | `hrm.py` | python | 508 | 20.5 | `efacc1d28c91` |
| F0005 | `knowledge.py` | python | 317 | 10.7 | `c21cca5d1b7e` |
| F0006 | `model_locator_mixin.py` | python | 74 | 2.4 | `850b51b53e3f` |
| F0007 | `model_selftest.py` | python | 162 | 5.1 | `56da08e30b90` |
| F0008 | `mrq.py` | python | 60 | 2.4 | `723ef1568047` |
| F0009 | `pacs_optimizer.py` | python | 105 | 3.2 | `0aa450618ed6` |
| F0010 | `preference_ranker.py` | python | 30 | 0.8 | `872fb2f26fd8` |
| F0011 | `risk_predictor.py` | python | 332 | 11.2 | `e8e972ec3741` |
| F0012 | `sicql.py` | python | 302 | 8.3 | `3642d44f4d23` |
| F0013 | `text_encoder.py` | python | 19 | 0.5 | `e19d76cd7cac` |
| F0014 | `tiny.py` | python | 507 | 19.7 | `03b7d4e3ed09` |
| F0015 | `value_predictor.py` | python | 20 | 0.6 | `1944fd2c5985` |
| F0016 | `vpm.py` | python | 259 | 8.5 | `e919df285562` |
| F0017 | `vpm_thought_policy.py` | python | 60 | 2.1 | `247d192fd9d2` |
| F0018 | `vpm_vit.py` | python | 138 | 5.3 | `d6e3d02d2595` |

## Directory Tree

```text
└─ ebt.py
└─ epistemic_trace_encoder.py
└─ garden_critic_eval.py
└─ hrm.py
└─ knowledge.py
└─ model_locator_mixin.py
└─ model_selftest.py
└─ mrq.py
└─ pacs_optimizer.py
└─ preference_ranker.py
└─ risk_predictor.py
└─ sicql.py
└─ text_encoder.py
└─ tiny.py
└─ value_predictor.py
└─ vpm.py
└─ vpm_thought_policy.py
└─ vpm_vit.py
```

## Files


---

## F0001 — `ebt.py`

```text
FILE_ID: F0001
PATH: ebt.py
LANGUAGE: python
LINES: 80
BYTES_UTF8: 2423
SHA256: 438fdb08e781f07c4618c9f83da25fcde8af2b5799b079f8074f5e87b346a93e
```

```python
0001 | # stephanie/model/ebt.py
0002 | from __future__ import annotations
0003 | 
0004 | import torch
0005 | from torch import nn
0006 | from torch.nn import functional as F
0007 | 
0008 | 
0009 | class EBTModel(nn.Module):
0010 |     def __init__(self, embedding_dim=1024, hidden_dim=256, num_actions=3, device="cpu"):
0011 |         super().__init__()
0012 |         self.embedding_dim = embedding_dim
0013 |         self.hidden_dim = hidden_dim
0014 |         self.num_actions = num_actions
0015 |         self.device = device
0016 | 
0017 |         # Encoder with attention
0018 |         self.encoder = nn.Sequential(
0019 |             nn.Linear(embedding_dim * 2, hidden_dim),
0020 |             nn.ReLU(),
0021 |             nn.LayerNorm(hidden_dim),
0022 |             nn.Linear(hidden_dim, hidden_dim)
0023 |         )
0024 | 
0025 |         # Q head with learnable scaling
0026 |         self.q_head = nn.Sequential(
0027 |             nn.Linear(hidden_dim, hidden_dim // 2),
0028 |             nn.ReLU(),
0029 |             nn.Linear(hidden_dim // 2, 1)
0030 |         )
0031 | 
0032 |         # V head with expectile regression
0033 |         self.v_head = nn.Sequential(
0034 |             nn.Linear(hidden_dim, hidden_dim // 2),
0035 |             nn.ReLU(),
0036 |             nn.Linear(hidden_dim // 2, 1)
0037 |         )
0038 | 
0039 |         # Policy head with policy entropy
0040 |         self.pi_head = nn.Sequential(
0041 |             nn.Linear(hidden_dim, hidden_dim // 2),
0042 |             nn.ReLU(),
0043 |             nn.Linear(hidden_dim // 2, num_actions)
0044 |         )
0045 | 
0046 |         # Learnable scaling factor
0047 |         self.scale_factor = nn.Parameter(torch.tensor(10.0))
0048 | 
0049 |     def forward(self, context_emb, output_emb):
0050 |         # Ensure device alignment
0051 |         context_emb = context_emb.to(self.device)
0052 |         output_emb = output_emb.to(self.device)
0053 |         
0054 |         # Combine embeddings
0055 |         combined = torch.cat([context_emb, output_emb], dim=-1)
0056 |         zsa = self.encoder(combined)
0057 |         
0058 |         # Q/V heads
0059 |         q_value = self.q_head(zsa).squeeze()
0060 |         state_value = self.v_head(zsa).squeeze()
0061 |         
0062 |         # Policy head
0063 |         action_logits = self.pi_head(zsa)
0064 |         action_probs = F.softmax(action_logits, dim=-1)
0065 |         
0066 |         # Compute advantage
0067 |         advantage = q_value - state_value
0068 |         
0069 |         # Scale final score
0070 |         final_score = q_value * torch.sigmoid(self.scale_factor).item()
0071 |         
0072 |         return {
0073 |             "q_value": q_value,
0074 |             "state_value": state_value,
0075 |             "action_logits": action_logits,
0076 |             "action_probs": action_probs,
0077 |             "advantage": advantage,
0078 |             "score": final_score
0079 |         }
0080 |     
```


---

## F0002 — `epistemic_trace_encoder.py`

```text
FILE_ID: F0002
PATH: epistemic_trace_encoder.py
LANGUAGE: python
LINES: 150
BYTES_UTF8: 6290
SHA256: 78d4b32b97dee537c92769cbcd3ff7a6a0cd6294a28e9153223b515c0fbcdb4c
```

```python
0001 | # stephanie/model/epistemic_trace_encoder.py
0002 | from __future__ import annotations
0003 | 
0004 | from typing import Callable, Dict
0005 | 
0006 | import numpy as np
0007 | import torch
0008 | import torch.nn as nn
0009 | 
0010 | 
0011 | class EpistemicTraceEncoder(nn.Module):
0012 |     """
0013 |     A hybrid encoder that transforms a full PlanTrace (goal + steps + scores + final output)
0014 |     into a single latent vector for downstream HRM-style scoring.
0015 | 
0016 |     The final representation is used as input to models like the Hierarchical Reasoning Model (HRM).
0017 |     It fuses multiple modalities:
0018 |       - goal and output embeddings (from LLM or embedding model)
0019 |       - encoded step-wise reasoning traces
0020 |       - aggregate scoring statistics (Q/V/energy/etc.)
0021 |     """
0022 | 
0023 |     def __init__(self, cfg: Dict[str, any]):
0024 |         """
0025 |         Initialize the encoder architecture based on configurable hyperparameters.
0026 | 
0027 |         Args:
0028 |             cfg (dict): Config dictionary with keys:
0029 |                 - embedding_dim: size of input text embeddings (default: 1024)
0030 |                 - step_hidden_dim: output dim for encoded step traces
0031 |                 - stats_input_dim: number of scalar stats per trace (e.g., Q/V/E)
0032 |                 - stats_hidden_dim: MLP hidden dim for stats vector
0033 |                 - final_dim: final encoded vector size
0034 |         """
0035 |         super().__init__()
0036 | 
0037 |         # Configuration with sensible defaults
0038 |         self.embedding_dim = cfg.get("embedding_dim", 1024)
0039 |         self.step_hidden_dim = cfg.get("step_hidden_dim", 64)
0040 |         self.stats_input_dim = cfg.get("stats_input_dim", 32)
0041 |         self.stats_hidden_dim = cfg.get("stats_hidden_dim", 128)
0042 |         self.final_dim = cfg.get("final_dim", 256)
0043 | 
0044 |         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
0045 | 
0046 |         print("[EpistemicTraceEncoder] Config:")
0047 |         print(f"  - embedding_dim: {self.embedding_dim}")
0048 |         print(f"  - step_hidden_dim: {self.step_hidden_dim}")
0049 |         print(f"  - stats_input_dim: {self.stats_input_dim}")
0050 |         print(f"  - stats_hidden_dim: {self.stats_hidden_dim}")
0051 |         print(f"  - final_dim: {self.final_dim}")
0052 | 
0053 |         # 1. Step encoder: compress individual step embeddings into a latent vector
0054 |         self.step_encoder = nn.Sequential(
0055 |             nn.Linear(self.embedding_dim, self.step_hidden_dim),
0056 |             nn.ReLU(),
0057 |             nn.Linear(self.step_hidden_dim, self.step_hidden_dim),
0058 |         ).to(self.device)
0059 | 
0060 |         # 2. Scoring statistics encoder: MLP for Q/V/Energy stats etc.
0061 |         self.stats_encoder = nn.Sequential(
0062 |             nn.Linear(self.stats_input_dim, self.stats_hidden_dim),
0063 |             nn.ReLU(),
0064 |             nn.Linear(self.stats_hidden_dim, self.stats_hidden_dim),
0065 |         ).to(self.device)
0066 | 
0067 |         # 3. Final combiner: concatenate goal, final output, steps, stats
0068 |         combined_input_dim = 2 * self.embedding_dim + self.step_hidden_dim + self.stats_hidden_dim
0069 |         self.combiner = nn.Sequential(
0070 |             nn.Linear(combined_input_dim, self.final_dim),
0071 |             nn.ReLU(),
0072 |             nn.Linear(self.final_dim, self.final_dim)
0073 |         ).to(self.device)
0074 | 
0075 |     def forward(
0076 |         self,
0077 |         trace,
0078 |         embedding_lookup_fn: Callable[[str], torch.Tensor],
0079 |         score_stats_fn: Callable[[object, list], torch.Tensor],
0080 |         dimensions: list[str]
0081 |     ) -> torch.Tensor:
0082 |         """
0083 |         Encode a reasoning trace into a latent vector.
0084 | 
0085 |         Args:
0086 |             trace: PlanTrace object (or dict-like) with fields:
0087 |                 - goal_text
0088 |                 - final_output_text
0089 |                 - execution_steps: list of ExecutionStep
0090 |             embedding_lookup_fn: callable that maps text → embedding tensor
0091 |             score_stats_fn: callable that returns numeric feature vector for scores
0092 |             dimensions: list of scoring dimensions (for stat extraction)
0093 | 
0094 |         Returns:
0095 |             torch.Tensor of shape [final_dim]
0096 |         """
0097 | 
0098 |         # -- Embed goal and final output text
0099 |         goal_emb = embedding_lookup_fn(trace.goal_text)
0100 |         final_emb = embedding_lookup_fn(trace.final_output_text)
0101 | 
0102 |         goal_emb = torch.as_tensor(goal_emb, dtype=torch.float32, device=self.device)
0103 |         final_emb = torch.as_tensor(final_emb, dtype=torch.float32, device=self.device)
0104 | 
0105 |         # -- Encode each step in the trace
0106 |         step_embeddings = []
0107 |         for step in trace.execution_steps:
0108 |             z_np = embedding_lookup_fn(step.output_text)
0109 |             z = torch.tensor(z_np, dtype=torch.float32, device=self.device) \
0110 |                 if isinstance(z_np, np.ndarray) else z_np.to(self.device)
0111 | 
0112 |             step_encoded = self.step_encoder(z)  # shape: [step_hidden_dim]
0113 |             step_embeddings.append(step_encoded)
0114 | 
0115 |         # -- Aggregate step representations (mean pool)
0116 |         if step_embeddings:
0117 |             step_pooled = torch.mean(torch.stack(step_embeddings, dim=0), dim=0)
0118 |         else:
0119 |             step_pooled = torch.zeros(self.step_hidden_dim, device=self.device)
0120 | 
0121 |         # -- Get score stats (e.g., mean Q, max energy, etc.)
0122 |         stats_vector = score_stats_fn(trace, dimensions)  # shape: [stats_input_dim]
0123 | 
0124 |         stats_vector = torch.as_tensor(stats_vector, dtype=torch.float32, device=self.device)
0125 | 
0126 |         # Normalize shapes to [N]
0127 |         stats_vector = stats_vector.view(-1)
0128 | 
0129 |         if stats_vector.numel() != self.stats_input_dim:
0130 |             raise RuntimeError(
0131 |                 f"[EpistemicTraceEncoder] stats_vector has {stats_vector.numel()} features "
0132 |                 f"but stats_encoder expects stats_input_dim={self.stats_input_dim}. "
0133 |                 f"Fix by setting cfg['stats_input_dim']={stats_vector.numel()} "
0134 |                 f"or updating score_stats_fn to return {self.stats_input_dim} features."
0135 |             )
0136 | 
0137 |         stats_encoded = self.stats_encoder(stats_vector.to(self.device))
0138 | 
0139 |         # -- Concatenate all latent components
0140 |         combined = torch.cat([
0141 |             goal_emb,         # [embedding_dim]
0142 |             final_emb,        # [embedding_dim]
0143 |             step_pooled,      # [step_hidden_dim]
0144 |             stats_encoded     # [stats_hidden_dim]
0145 |         ], dim=-1)
0146 | 
0147 |         # -- Final projection to fixed-size trace representation
0148 |         z_trace = self.combiner(combined)  # shape: [final_dim]
0149 |         print(f"[EpistemicTraceEncoder] Encoded trace to shape: {z_trace.shape}")   
0150 |         return z_trace
```


---

## F0003 — `garden_critic_eval.py`

```text
FILE_ID: F0003
PATH: garden_critic_eval.py
LANGUAGE: python
LINES: 680
BYTES_UTF8: 21710
SHA256: 8a66c522bccc59883e877c30beaa1db72d816a7e46ec4ed747181b2bcdc2bf87
```

```python
0001 | # stephanie/model/garden_critic_eval.py
0002 | from __future__ import annotations
0003 | 
0004 | import argparse
0005 | import json
0006 | from dataclasses import dataclass
0007 | from pathlib import Path
0008 | from typing import Any, Dict, List, Tuple
0009 | 
0010 | import matplotlib.pyplot as plt
0011 | import numpy as np
0012 | import torch
0013 | import torch.nn as nn
0014 | from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
0015 | from torch.utils.data import DataLoader, Dataset, Subset
0016 | from torchvision.models import resnet18
0017 | 
0018 | # ----------------------------
0019 | #  Dataset
0020 | # ----------------------------
0021 | 
0022 | @dataclass
0023 | class GardenSample:
0024 |     run_id: str
0025 |     label: int            # 0 = baseline, 1 = improved
0026 |     vpm_img: np.ndarray   # HxW or HxWxC
0027 |     metrics: np.ndarray   # HxW
0028 | 
0029 | 
0030 | class GardenHealthDataset(Dataset):
0031 |     """
0032 |     Dataset for training a critic to distinguish improved vs baseline gardens.
0033 | 
0034 |     Expects:
0035 |         baseline_path: JSON mapping run_id -> { "vpm_img": ..., "metrics": ... }
0036 |         improved_path: same schema
0037 | 
0038 |     For each run_id present in both files, we create:
0039 |         - one sample with label=0 (baseline)
0040 |         - one sample with label=1 (improved)
0041 |     """
0042 | 
0043 |     def __init__(
0044 |         self,
0045 |         baseline_path: Path,
0046 |         improved_path: Path,
0047 |     ) -> None:
0048 |         baseline = json.loads(Path(baseline_path).read_text())
0049 |         improved = json.loads(Path(improved_path).read_text())
0050 | 
0051 |         run_ids = sorted(set(baseline.keys()) & set(improved.keys()))
0052 |         if not run_ids:
0053 |             raise ValueError("No overlapping run_ids between baseline and improved JSON files")
0054 | 
0055 |         self.samples: List[GardenSample] = []
0056 |         for run_id in run_ids:
0057 |             b = baseline[run_id]
0058 |             i = improved[run_id]
0059 | 
0060 |             b_img = np.asarray(b["vpm_img"], dtype=np.float32)
0061 |             b_metrics = np.asarray(b["metrics"], dtype=np.float32)
0062 |             i_img = np.asarray(i["vpm_img"], dtype=np.float32)
0063 |             i_metrics = np.asarray(i["metrics"], dtype=np.float32)
0064 | 
0065 |             self.samples.append(
0066 |                 GardenSample(
0067 |                     run_id=run_id,
0068 |                     label=0,
0069 |                     vpm_img=b_img,
0070 |                     metrics=b_metrics,
0071 |                 )
0072 |             )
0073 |             self.samples.append(
0074 |                 GardenSample(
0075 |                     run_id=run_id,
0076 |                     label=1,
0077 |                     vpm_img=i_img,
0078 |                     metrics=i_metrics,
0079 |                 )
0080 |             )
0081 | 
0082 |         # Precompute run_id -> global indices mapping for pairwise eval
0083 |         self.run_to_indices: Dict[str, List[int]] = {}
0084 |         for idx, s in enumerate(self.samples):
0085 |             self.run_to_indices.setdefault(s.run_id, []).append(idx)
0086 | 
0087 |         # Basic sanity check: exactly 2 samples per run
0088 |         for run_id, idxs in self.run_to_indices.items():
0089 |             if len(idxs) != 2:
0090 |                 raise ValueError(f"Run {run_id} has {len(idxs)} samples (expected 2)")
0091 | 
0092 |         # Metric names are implicit in column order; we treat them as indices for now
0093 |         # If you want real names, you can pass them separately.
0094 |         example_metrics = self.samples[0].metrics
0095 |         if example_metrics.ndim != 2:
0096 |             raise ValueError(f"metrics must be HxW, got shape {example_metrics.shape}")
0097 |         self.num_metrics = example_metrics.shape[1]
0098 |         self.metric_names = [f"m_{j}" for j in range(self.num_metrics)]
0099 | 
0100 |     def __len__(self) -> int:
0101 |         return len(self.samples)
0102 | 
0103 |     def __getitem__(self, idx: int) -> Dict[str, Any]:
0104 |         s = self.samples[idx]
0105 | 
0106 |         # VPM image -> tensor (C, H, W)
0107 |         img = torch.from_numpy(s.vpm_img)  # HxW or HxWxC
0108 |         if img.ndim == 2:
0109 |             img = img.unsqueeze(0)  # 1, H, W
0110 |         elif img.ndim == 3:
0111 |             # Assume HxWxC
0112 |             img = img.permute(2, 0, 1)  # C, H, W
0113 |         else:
0114 |             raise ValueError(f"Unexpected vpm_img shape {s.vpm_img.shape}")
0115 | 
0116 |         # Ensure 3 channels (for ResNet)
0117 |         if img.shape[0] == 1:
0118 |             img = img.repeat(3, 1, 1)
0119 |         elif img.shape[0] != 3:
0120 |             raise ValueError(f"Expected 1 or 3 channels, got {img.shape[0]}")
0121 | 
0122 |         # Metrics -> tensor (H, W)
0123 |         metrics = torch.from_numpy(s.metrics)  # HxW
0124 | 
0125 |         # Label -> float (for BCEWithLogitsLoss)
0126 |         label = torch.tensor(float(s.label), dtype=torch.float32)
0127 | 
0128 |         return {
0129 |             "run_id": s.run_id,
0130 |             "vpm_img": img,
0131 |             "metrics": metrics,
0132 |             "label": label,
0133 |         }
0134 | 
0135 | 
0136 | def train_test_split_by_run(
0137 |     dataset: GardenHealthDataset,
0138 |     test_size: float = 0.2,
0139 |     seed: int = 42,
0140 | ) -> Tuple[Subset, Subset, List[str]]:
0141 |     """
0142 |     Split by run_id so that each baseline/improved pair stays in the same split.
0143 |     Returns:
0144 |         train_subset, test_subset, test_run_ids
0145 |     """
0146 |     rng = np.random.RandomState(seed)
0147 |     all_run_ids = sorted(dataset.run_to_indices.keys())
0148 |     rng.shuffle(all_run_ids)
0149 | 
0150 |     n_test = max(1, int(len(all_run_ids) * test_size))
0151 |     test_run_ids = sorted(all_run_ids[:n_test])
0152 |     train_run_ids = sorted(all_run_ids[n_test:])
0153 | 
0154 |     train_indices: List[int] = []
0155 |     test_indices: List[int] = []
0156 | 
0157 |     for rid in train_run_ids:
0158 |         train_indices.extend(dataset.run_to_indices[rid])
0159 |     for rid in test_run_ids:
0160 |         test_indices.extend(dataset.run_to_indices[rid])
0161 | 
0162 |     train_subset = Subset(dataset, train_indices)
0163 |     test_subset = Subset(dataset, test_indices)
0164 |     return train_subset, test_subset, test_run_ids
0165 | 
0166 | 
0167 | # ----------------------------
0168 | #  Model
0169 | # ----------------------------
0170 | 
0171 | class GardenHealthCritic(nn.Module):
0172 |     """
0173 |     Fused critic with toggles for image-only / metrics-only / both.
0174 | 
0175 |     - If use_image=True, use img_encoder branch.
0176 |     - If use_metrics=True, use metrics branch.
0177 |     - If both, fuse by concatenation.
0178 | 
0179 |     Inputs:
0180 |         vpm_img:    (B, 3, H, W)
0181 |         metrics:    (B, H, W)
0182 |     Output:
0183 |         logits:     (B,)
0184 |     """
0185 | 
0186 |     def __init__(
0187 |         self,
0188 |         metric_names: List[str],
0189 |         img_encoder: nn.Module | None = None,
0190 |         use_image: bool = True,
0191 |         use_metrics: bool = True,
0192 |         d_struct: int = 64,
0193 |         d_fuse: int = 128,
0194 |     ) -> None:
0195 |         super().__init__()
0196 | 
0197 |         if not use_image and not use_metrics:
0198 |             raise ValueError("At least one of use_image or use_metrics must be True")
0199 | 
0200 |         self.metric_names = metric_names
0201 |         self.num_metrics = len(metric_names)
0202 |         self.use_image = use_image
0203 |         self.use_metrics = use_metrics
0204 | 
0205 |         # Visual branch
0206 |         self.img_encoder = img_encoder if use_image else None
0207 |         if self.img_encoder is not None:
0208 |             # For ResNet, we replace fc with Identity so we get feature vectors
0209 |             if hasattr(self.img_encoder, "fc") and isinstance(self.img_encoder.fc, nn.Linear):
0210 |                 out_dim = self.img_encoder.fc.in_features
0211 |                 self.img_encoder.fc = nn.Identity()
0212 |                 self.img_out_dim = out_dim
0213 |             else:
0214 |                 raise ValueError("img_encoder must be a ResNet-like model with .fc attribute")
0215 | 
0216 |             self.img_proj = nn.Linear(self.img_out_dim, d_fuse)
0217 | 
0218 |         # Structured branch
0219 |         if use_metrics:
0220 |             self.name_embed = nn.Embedding(self.num_metrics, d_struct // 2)
0221 |             self.value_proj = nn.Linear(1, d_struct // 2)
0222 |             self.struct_proj = nn.Linear(d_struct, d_fuse)
0223 | 
0224 |         # Fusion output dim
0225 |         fuse_in_dim = 0
0226 |         if use_image:
0227 |             fuse_in_dim += d_fuse
0228 |         if use_metrics:
0229 |             fuse_in_dim += d_fuse
0230 | 
0231 |         self.fuse = nn.Sequential(
0232 |             nn.Linear(fuse_in_dim, d_fuse),
0233 |             nn.ReLU(),
0234 |             nn.Linear(d_fuse, 1),
0235 |         )
0236 | 
0237 |     def forward(self, vpm_img: torch.Tensor, metrics: torch.Tensor) -> torch.Tensor:
0238 |         """
0239 |         vpm_img: (B, 3, H, W)
0240 |         metrics: (B, H, W)
0241 |         """
0242 |         feats: List[torch.Tensor] = []
0243 | 
0244 |         if self.use_image:
0245 |             img_feat = self.img_encoder(vpm_img)          # (B, img_out_dim)
0246 |             img_feat = self.img_proj(img_feat)           # (B, d_fuse)
0247 |             feats.append(img_feat)
0248 | 
0249 |         if self.use_metrics:
0250 |             B, H, W = metrics.shape
0251 |             if W != self.num_metrics:
0252 |                 raise ValueError(
0253 |                     f"Metric grid width {W} != num_metrics {self.num_metrics}; "
0254 |                     "ensure metrics are HxW with W == len(metric_names)."
0255 |                 )
0256 | 
0257 |             # Metric name embeddings: (W, d_struct/2)
0258 |             device = metrics.device
0259 |             name_ids = torch.arange(self.num_metrics, device=device, dtype=torch.long)
0260 |             name_emb = self.name_embed(name_ids)         # (W, d_struct/2)
0261 |             name_emb = name_emb.unsqueeze(0).unsqueeze(0).expand(B, H, W, -1)
0262 | 
0263 |             # Metric values: (B, H, W, 1) -> (B, H, W, d_struct/2)
0264 |             values = metrics.unsqueeze(-1)               # (B, H, W, 1)
0265 |             value_emb = self.value_proj(values)          # (B, H, W, d_struct/2)
0266 | 
0267 |             struct_grid = torch.cat([value_emb, name_emb], dim=-1)  # (B, H, W, d_struct)
0268 |             struct_feat = struct_grid.mean(dim=1).mean(dim=1)       # (B, d_struct)
0269 |             struct_feat = self.struct_proj(struct_feat)             # (B, d_fuse)
0270 |             feats.append(struct_feat)
0271 | 
0272 |         if len(feats) == 1:
0273 |             fused = feats[0]
0274 |         else:
0275 |             fused = torch.cat(feats, dim=-1)
0276 | 
0277 |         logits = self.fuse(fused).view(-1)  # (B,)
0278 |         return logits
0279 | 
0280 | 
0281 | # ----------------------------
0282 | #  Evaluation Metrics
0283 | # ----------------------------
0284 | 
0285 | def expected_calibration_error(probs, labels, n_bins=10):
0286 |     """Compute Expected Calibration Error"""
0287 |     bin_boundaries = np.linspace(0, 1, n_bins + 1)
0288 |     bin_lowers = bin_boundaries[:-1]
0289 |     bin_uppers = bin_boundaries[1:]
0290 |     
0291 |     bin_accs = []
0292 |     bin_confs = []
0293 |     bin_sizes = []
0294 |     
0295 |     for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
0296 |         in_bin = np.logical_and(probs >= bin_lower, probs < bin_upper)
0297 |         bin_size = np.sum(in_bin)
0298 |         if bin_size > 0:
0299 |             bin_acc = np.mean(labels[in_bin])
0300 |             bin_conf = np.mean(probs[in_bin])
0301 |         else:
0302 |             bin_acc = 0
0303 |             bin_conf = 0
0304 |         bin_accs.append(bin_acc)
0305 |         bin_confs.append(bin_conf)
0306 |         bin_sizes.append(bin_size)
0307 |     
0308 |     ece = 0
0309 |     for i in range(len(bin_sizes)):
0310 |         ece += bin_sizes[i] * np.abs(bin_accs[i] - bin_confs[i])
0311 |     ece /= len(probs)
0312 |     
0313 |     return ece
0314 | 
0315 | 
0316 | def compute_pairwise_accuracy(probs, labels, run_ids):
0317 |     """Compute pairwise accuracy for baseline vs improved pairs"""
0318 |     # Build run -> list of (label, prob)
0319 |     run_pairs: Dict[str, List[Tuple[int, float]]] = {}
0320 |     for run_id, label, prob in zip(run_ids, labels, probs):
0321 |         run_pairs.setdefault(run_id, []).append((label, prob))
0322 |     
0323 |     correct_pairs = 0
0324 |     total_pairs = 0
0325 |     for run_id, items in run_pairs.items():
0326 |         if len(items) != 2:
0327 |             continue
0328 |         total_pairs += 1
0329 |         
0330 |         # Find which item is the improved sample (label=1)
0331 |         improved_item = None
0332 |         baseline_item = None
0333 |         for item in items:
0334 |             if item[0] == 1:
0335 |                 improved_item = item
0336 |             else:
0337 |                 baseline_item = item
0338 |         
0339 |         if improved_item is not None and baseline_item is not None:
0340 |             if improved_item[1] > baseline_item[1]:
0341 |                 correct_pairs += 1
0342 |     
0343 |     return correct_pairs / max(1, total_pairs)
0344 | 
0345 | 
0346 | def compute_metrics(probs, labels, run_ids):
0347 |     """Compute all evaluation metrics"""
0348 |     # AUC
0349 |     try:
0350 |         auc = roc_auc_score(labels, probs)
0351 |     except ValueError:
0352 |         auc = float("nan")
0353 |     
0354 |     # Pairwise accuracy
0355 |     pairwise_acc = compute_pairwise_accuracy(probs, labels, run_ids)
0356 |     
0357 |     # ECE
0358 |     ece = expected_calibration_error(probs, labels)
0359 |     
0360 |     # Brier Score
0361 |     brier = brier_score_loss(labels, probs)
0362 |     
0363 |     # Accuracy
0364 |     preds = (probs >= 0.5).astype(int)
0365 |     acc = accuracy_score(labels, preds)
0366 |     
0367 |     return {
0368 |         "auc": auc,
0369 |         "pairwise_acc": pairwise_acc,
0370 |         "ece": ece,
0371 |         "brier": brier,
0372 |         "accuracy": acc
0373 |     }
0374 | 
0375 | 
0376 | def plot_calibration_curve(probs, labels, n_bins=10, save_path=None):
0377 |     """Plot calibration curve and save if save_path is provided"""
0378 |     bin_boundaries = np.linspace(0, 1, n_bins + 1)
0379 |     bin_lowers = bin_boundaries[:-1]
0380 |     bin_uppers = bin_boundaries[1:]
0381 |     
0382 |     bin_accs = []
0383 |     bin_confs = []
0384 |     bin_sizes = []
0385 |     
0386 |     for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
0387 |         in_bin = np.logical_and(probs >= bin_lower, probs < bin_upper)
0388 |         bin_size = np.sum(in_bin)
0389 |         if bin_size > 0:
0390 |             bin_acc = np.mean(labels[in_bin])
0391 |             bin_conf = np.mean(probs[in_bin])
0392 |         else:
0393 |             bin_acc = 0
0394 |             bin_conf = 0
0395 |         bin_accs.append(bin_acc)
0396 |         bin_confs.append(bin_conf)
0397 |         bin_sizes.append(bin_size)
0398 |     
0399 |     # Plot
0400 |     plt.figure(figsize=(8, 6))
0401 |     plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
0402 |     
0403 |     plt.plot(bin_confs, bin_accs, "s-", label="Model")
0404 |     
0405 |     plt.xlabel("Confidence")
0406 |     plt.ylabel("Accuracy")
0407 |     plt.title("Calibration Curve")
0408 |     plt.legend()
0409 |     plt.grid(True)
0410 |     
0411 |     if save_path:
0412 |         plt.savefig(save_path)
0413 |         plt.close()
0414 |     else:
0415 |         plt.show()
0416 | 
0417 | 
0418 | # ----------------------------
0419 | #  Training / Evaluation
0420 | # ----------------------------
0421 | 
0422 | def set_seed(seed: int = 42) -> None:
0423 |     np.random.seed(seed)
0424 |     torch.manual_seed(seed)
0425 |     torch.cuda.manual_seed_all(seed)
0426 | 
0427 | 
0428 | def train_one_model(
0429 |     name: str,
0430 |     model: nn.Module,
0431 |     train_loader: DataLoader,
0432 |     test_loader: DataLoader,
0433 |     device: torch.device,
0434 |     epochs: int = 5,
0435 |     lr: float = 3e-4,
0436 |     save_model: Path | None = None,
0437 | ) -> Dict[str, float]:
0438 |     """
0439 |     Train a single critic model and return metrics.
0440 |     """
0441 |     model.to(device)
0442 |     optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
0443 |     criterion = nn.BCEWithLogitsLoss()
0444 | 
0445 |     # ---- Train ----
0446 |     for epoch in range(1, epochs + 1):
0447 |         model.train()
0448 |         running_loss = 0.0
0449 |         n = 0
0450 | 
0451 |         for batch in train_loader:
0452 |             vpm = batch["vpm_img"].to(device)           # (B, 3, H, W)
0453 |             metrics = batch["metrics"].to(device)       # (B, H, W)
0454 |             labels = batch["label"].to(device)          # (B,)
0455 | 
0456 |             logits = model(vpm, metrics)                # (B,)
0457 |             loss = criterion(logits, labels)
0458 | 
0459 |             optimizer.zero_grad()
0460 |             loss.backward()
0461 |             optimizer.step()
0462 | 
0463 |             running_loss += loss.item() * vpm.size(0)
0464 |             n += vpm.size(0)
0465 | 
0466 |         avg_loss = running_loss / max(1, n)
0467 |         print(f"[{name}] Epoch {epoch}/{epochs} - loss={avg_loss:.4f}")
0468 | 
0469 |     # ---- Evaluate ----
0470 |     model.eval()
0471 |     all_logits: List[float] = []
0472 |     all_labels: List[int] = []
0473 |     all_run_ids: List[str] = []
0474 |     all_global_indices: List[int] = []  # reference back to base dataset
0475 | 
0476 |     # test_loader.dataset is a Subset
0477 |     subset: Subset = test_loader.dataset
0478 |     base_dataset: GardenHealthDataset = subset.dataset  # type: ignore[assignment]
0479 |     subset_indices: List[int] = subset.indices          # type: ignore[assignment]
0480 | 
0481 |     with torch.no_grad():
0482 |         offset = 0
0483 |         for batch_idx, batch in enumerate(test_loader):
0484 |             vpm = batch["vpm_img"].to(device)
0485 |             metrics = batch["metrics"].to(device)
0486 |             labels = batch["label"].cpu().numpy()
0487 |             run_ids = batch["run_id"]
0488 | 
0489 |             logits = model(vpm, metrics).cpu().numpy()
0490 | 
0491 |             B = len(labels)
0492 |             all_logits.extend(logits.tolist())
0493 |             all_labels.extend(labels.astype(int).tolist())
0494 |             all_run_ids.extend(run_ids)
0495 | 
0496 |             # Map to global indices
0497 |             all_global_indices.extend(subset_indices[offset : offset + B])
0498 |             offset += B
0499 | 
0500 |     # Convert logits to probabilities
0501 |     probs = 1.0 / (1.0 + np.exp(-np.array(all_logits)))
0502 |     labels_arr = np.array(all_labels)
0503 |     
0504 |     # Compute metrics
0505 |     metrics = compute_metrics(probs, labels_arr, all_run_ids)
0506 |     
0507 |     # Save model if requested
0508 |     if save_model:
0509 |         save_path = save_model / f"{name}_model.pt"
0510 |         save_path.parent.mkdir(parents=True, exist_ok=True)
0511 |         torch.save(model.state_dict(), save_path)
0512 |         print(f"Saved {name} model to {save_path}")
0513 |         
0514 |         # Save calibration plot
0515 |         calib_plot_path = save_model / f"{name}_calibration.png"
0516 |         plot_calibration_curve(probs, labels_arr, save_path=calib_plot_path)
0517 |         print(f"Saved calibration curve to {calib_plot_path}")
0518 |     
0519 |     # Print metrics
0520 |     print(f"[{name}] AUC={metrics['auc']:.4f}, PairwiseAcc={metrics['pairwise_acc']:.4f}, "
0521 |           f"ECE={metrics['ece']:.4f}, Brier={metrics['brier']:.4f}, "
0522 |           f"Accuracy={metrics['accuracy']:.4f}")
0523 |     
0524 |     return metrics
0525 | 
0526 | 
0527 | def build_dataloaders(
0528 |     dataset: GardenHealthDataset,
0529 |     batch_size: int = 16,
0530 |     test_size: float = 0.2,
0531 |     seed: int = 42,
0532 | ) -> Tuple[DataLoader, DataLoader]:
0533 |     train_subset, test_subset, _ = train_test_split_by_run(dataset, test_size=test_size, seed=seed)
0534 |     train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
0535 |     test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
0536 |     return train_loader, test_loader
0537 | 
0538 | 
0539 | def main() -> None:
0540 |     parser = argparse.ArgumentParser(description="Garden Health Critic Evaluation")
0541 |     parser.add_argument(
0542 |         "--baseline_json",
0543 |         type=Path,
0544 |         default=Path("data/baseline_graph.json"),
0545 |         help="Path to baseline JSON",
0546 |     )
0547 |     parser.add_argument(
0548 |         "--improved_json",
0549 |         type=Path,
0550 |         default=Path("data/graph_improved.json"),
0551 |         help="Path to improved JSON",
0552 |     )
0553 |     parser.add_argument(
0554 |         "--epochs",
0555 |         type=int,
0556 |         default=5,
0557 |         help="Training epochs per model",
0558 |     )
0559 |     parser.add_argument(
0560 |         "--batch_size",
0561 |         type=int,
0562 |         default=16,
0563 |         help="Batch size",
0564 |     )
0565 |     parser.add_argument(
0566 |         "--device",
0567 |         type=str,
0568 |         default="cuda" if torch.cuda.is_available() else "cpu",
0569 |         help="Device (cuda or cpu)",
0570 |     )
0571 |     parser.add_argument(
0572 |         "--save_model",
0573 |         type=Path,
0574 |         default=None,
0575 |         help="Directory to save trained models and calibration plots",
0576 |     )
0577 |     args = parser.parse_args()
0578 | 
0579 |     set_seed(42)
0580 | 
0581 |     print("Loading dataset...")
0582 |     dataset = GardenHealthDataset(
0583 |         baseline_path=args.baseline_json,
0584 |         improved_path=args.improved_json,
0585 |     )
0586 |     
0587 |     print(f"Dataset loaded with {len(dataset)} samples ({len(dataset.run_to_indices)} runs)")
0588 |     print(f"Train/Test split: {int((1-0.2)*len(dataset.run_to_indices))} training runs, "
0589 |           f"{int(0.2*len(dataset.run_to_indices))} test runs")
0590 |     
0591 |     train_loader, test_loader = build_dataloaders(
0592 |         dataset,
0593 |         batch_size=args.batch_size,
0594 |         test_size=0.2,
0595 |         seed=42,
0596 |     )
0597 | 
0598 |     device = torch.device(args.device)
0599 |     metric_names = dataset.metric_names
0600 | 
0601 |     # --- Build three variants: metrics-only, image-only, fused ---
0602 |     results: Dict[str, Dict[str, float]] = {}
0603 | 
0604 |     # Metrics-only
0605 |     metrics_only_model = GardenHealthCritic(
0606 |         metric_names=metric_names,
0607 |         img_encoder=None,
0608 |         use_image=False,
0609 |         use_metrics=True,
0610 |     )
0611 |     print("\n=== Training metrics-only critic ===")
0612 |     results["metrics_only"] = train_one_model(
0613 |         name="metrics_only",
0614 |         model=metrics_only_model,
0615 |         train_loader=train_loader,
0616 |         test_loader=test_loader,
0617 |         device=device,
0618 |         epochs=args.epochs,
0619 |         save_model=args.save_model,
0620 |     )
0621 | 
0622 |     # Image-only
0623 |     img_encoder = resnet18(pretrained=False)
0624 |     image_only_model = GardenHealthCritic(
0625 |         metric_names=metric_names,
0626 |         img_encoder=img_encoder,
0627 |         use_image=True,
0628 |         use_metrics=False,
0629 |     )
0630 |     print("\n=== Training image-only critic ===")
0631 |     results["image_only"] = train_one_model(
0632 |         name="image_only",
0633 |         model=image_only_model,
0634 |         train_loader=train_loader,
0635 |         test_loader=test_loader,
0636 |         device=device,
0637 |         epochs=args.epochs,
0638 |         save_model=args.save_model,
0639 |     )
0640 | 
0641 |     # Fused
0642 |     img_encoder_fused = resnet18(pretrained=False)
0643 |     fused_model = GardenHealthCritic(
0644 |         metric_names=metric_names,
0645 |         img_encoder=img_encoder_fused,
0646 |         use_image=True,
0647 |         use_metrics=True,
0648 |     )
0649 |     print("\n=== Training fused critic (image + metrics) ===")
0650 |     results["fused"] = train_one_model(
0651 |         name="fused",
0652 |         model=fused_model,
0653 |         train_loader=train_loader,
0654 |         test_loader=test_loader,
0655 |         device=device,
0656 |         epochs=args.epochs,
0657 |         save_model=args.save_model,
0658 |     )
0659 | 
0660 |     # --- Print summary table ---
0661 |     print("\n================ Experiment Summary ================")
0662 |     print(f"{'Model':<15} {'AUC':>8} {'PairwiseAcc':>14} {'ECE':>8} {'Brier':>8} {'Accuracy':>10}")
0663 |     print("-" * 60)
0664 |     for name, metrics in results.items():
0665 |         print(f"{name:<15} {metrics['auc']:>8.4f} {metrics['pairwise_acc']:>14.4f} "
0666 |               f"{metrics['ece']:>8.4f} {metrics['brier']:>8.4f} {metrics['accuracy']:>10.4f}")
0667 |     print("===================================================")
0668 |     
0669 |     # Print conclusion
0670 |     if results["fused"]["auc"] > results["metrics_only"]["auc"] and \
0671 |        results["fused"]["auc"] > results["image_only"]["auc"]:
0672 |         print("\nConclusion: The fused model outperforms both single-modality models, "
0673 |               "demonstrating the value of combining visual and metric information.")
0674 |     else:
0675 |         print("\nConclusion: One single-modality model outperforms the fused model, "
0676 |               "suggesting potential issues with the fusion strategy.")
0677 | 
0678 | 
0679 | if __name__ == "__main__":
0680 |     main()
```


---

## F0004 — `hrm.py`

```text
FILE_ID: F0004
PATH: hrm.py
LANGUAGE: python
LINES: 508
BYTES_UTF8: 20956
SHA256: efacc1d28c913ce7c3b896014dafd7dd0e37642b14f688639b07469983ff8c86
```

```python
0001 | # stephanie/model/hrm.py
0002 | """
0003 | Hierarchical Reasoning Model (HRM) - Advanced Neural Architecture for Quality Assessment
0004 | 
0005 | This module implements a sophisticated hierarchical recurrent neural network for
0006 | evaluating AI model responses. The architecture features two coupled recurrent
0007 | networks operating at different temporal scales, enabling complex reasoning
0008 | patterns and comprehensive quality assessment.
0009 | 
0010 | Architecture Overview:
0011 | - Dual recurrent hierarchy: Low-level (L) and High-level (H) modules
0012 | - Cyclic processing with fine-grained (L) and abstract (H) updates
0013 | - Multi-head prediction for comprehensive quality diagnostics
0014 | - Robustness through consistency regularization and uncertainty estimation
0015 | 
0016 | Key Features:
0017 | - Hierarchical temporal processing (T steps per cycle × N cycles)
0018 | - Multi-dimensional quality assessment (score, uncertainty, agreement, etc.)
0019 | - Input reconstruction for comprehension verification
0020 | - Finite-difference sensitivity analysis
0021 | - Aleatoric uncertainty estimation
0022 | 
0023 | Author: Stephanie AI Team
0024 | Version: 2.1
0025 | Date: 2024
0026 | """
0027 | 
0028 | from __future__ import annotations
0029 | 
0030 | import logging
0031 | from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
0032 | 
0033 | import torch
0034 | import torch.nn as nn
0035 | import torch.nn.functional as F
0036 | 
0037 | from stephanie.scoring.analysis.trace_tap import TraceTap
0038 | 
0039 | log = logging.getLogger(__name__)
0040 | 
0041 | 
0042 | class RMSNorm(nn.Module):
0043 |     """
0044 |     Root Mean Square Normalization (RMSNorm) - Efficient Alternative to LayerNorm.
0045 |     
0046 |     Normalizes across the feature dimension while maintaining representational
0047 |     capacity through a learnable scaling parameter. More computationally efficient
0048 |     than LayerNorm as it doesn't maintain running statistics.
0049 |     
0050 |     Reference: "Root Mean Square Layer Normalization" by Zhang & Sennrich (2019)
0051 |     
0052 |     Args:
0053 |         dim: Feature dimension to normalize
0054 |         eps: Small constant for numerical stability
0055 |     """
0056 |     def __init__(self, dim: int, eps: float = 1e-6):
0057 |         super().__init__()
0058 |         self.eps = eps
0059 |         self.weight = nn.Parameter(torch.ones(dim))  # Learnable scale parameter
0060 | 
0061 |     def _norm(self, x: torch.Tensor) -> torch.Tensor:
0062 |         """
0063 |         Apply RMS normalization: x / sqrt(mean(x^2) + eps)
0064 |         
0065 |         Args:
0066 |             x: Input tensor of shape [..., dim]
0067 |             
0068 |         Returns:
0069 |             Normalized tensor with same shape
0070 |         """
0071 |         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
0072 | 
0073 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
0074 |         """
0075 |         Forward pass with type preservation.
0076 |         
0077 |         Args:
0078 |             x: Input tensor of any dtype
0079 |             
0080 |         Returns:
0081 |             Normalized tensor with original dtype
0082 |         """
0083 |         # Convert to float for stable computation, then back to original type
0084 |         output = self._norm(x.float()).type_as(x) * self.weight
0085 |         return output
0086 | 
0087 | 
0088 | class RecurrentBlock(nn.Module):
0089 |     """
0090 |     Gated Recurrent Unit (GRU) Block with RMSNorm for Stable State Updates.
0091 |     
0092 |     Implements a single recurrent step with gating mechanisms and normalization
0093 |     for stable long-term gradient flow. Used by both L and H modules in HRM.
0094 |     
0095 |     Args:
0096 |         input_dim: Dimension of input features
0097 |         hidden_dim: Dimension of hidden state
0098 |         name: Identifier for debugging and logging
0099 |     """
0100 |     def __init__(self, input_dim: int, hidden_dim: int, name: str = "RecurrentBlock"):
0101 |         super().__init__()
0102 |         self.name = name
0103 |         self.rnn_cell = nn.GRUCell(input_dim, hidden_dim)  # Gated recurrent update
0104 |         self.norm = RMSNorm(hidden_dim)  # State normalization
0105 | 
0106 |     def forward(self, z_prev: torch.Tensor, input_combined: torch.Tensor) -> torch.Tensor:
0107 |         """
0108 |         Single recurrent step: GRU update + normalization.
0109 |         
0110 |         Args:
0111 |             z_prev: Previous hidden state [batch, hidden_dim]
0112 |             input_combined: Current input features [batch, input_dim]
0113 |             
0114 |         Returns:
0115 |             Updated hidden state [batch, hidden_dim]
0116 |         """
0117 |         z_next = self.rnn_cell(input_combined, z_prev)  # GRU state update
0118 |         z_next = self.norm(z_next)  # Stabilize hidden state
0119 |         return z_next
0120 | 
0121 |     @staticmethod
0122 |     def init_state(batch_size: int, hidden_dim: int, device: torch.device) -> torch.Tensor:
0123 |         """
0124 |         Initialize hidden state with zeros.
0125 |         
0126 |         Args:
0127 |             batch_size: Number of sequences in batch
0128 |             hidden_dim: Hidden state dimension
0129 |             device: Target device for tensor allocation
0130 |             
0131 |         Returns:
0132 |             Zero-initialized hidden state [batch_size, hidden_dim]
0133 |         """
0134 |         return torch.zeros(batch_size, hidden_dim, device=device)
0135 | 
0136 | 
0137 | class InputProjector(nn.Module):
0138 |     """
0139 |     Input Embedding Projection with Normalization and Dropout.
0140 |     
0141 |     Projects high-dimensional input embeddings into the HRM's hidden space
0142 |     with regularization for improved generalization.
0143 |     
0144 |     Args:
0145 |         input_dim: Original input embedding dimension
0146 |         hidden_dim: Target HRM hidden dimension
0147 |         dropout: Dropout probability for regularization
0148 |     """
0149 |     def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
0150 |         super().__init__()
0151 |         self.project = nn.Linear(input_dim, hidden_dim)  # Linear projection
0152 |         self.norm = RMSNorm(hidden_dim)  # Output normalization
0153 |         self.drop = nn.Dropout(dropout)  # Regularization
0154 | 
0155 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
0156 |         """
0157 |         Project input to HRM hidden space: Dropout → Linear → RMSNorm.
0158 |         
0159 |         Args:
0160 |             x: Input embeddings [batch, input_dim]
0161 |             
0162 |         Returns:
0163 |             Projected features [batch, hidden_dim]
0164 |         """
0165 |         return self.norm(self.drop(self.project(x)))
0166 | 
0167 | 
0168 | class HRMModel(nn.Module):
0169 |     """
0170 |     Hierarchical Reasoning Model (HRM) - Dual-Recurrent Architecture.
0171 |     
0172 |     Implements a hierarchical reasoning process through two coupled RNNs:
0173 |     - Low-level (L) module: Fine-grained processing (T steps per cycle)
0174 |     - High-level (H) module: Abstract reasoning (1 step per cycle)
0175 |     
0176 |     The architecture enables multi-scale temporal processing where the L module
0177 |     performs detailed analysis and the H module integrates information across
0178 |     longer time horizons.
0179 |     
0180 |     Multi-Head Diagnostic Surface:
0181 |       - score_head: Quality score ∈ [0,1] with temperature calibration
0182 |       - logvar_head: Aleatoric uncertainty estimation
0183 |       - aux3_head: 3-way classification (bad/medium/good)
0184 |       - disagree_head: Prediction of model disagreement
0185 |       - consistency_head: Robustness to input perturbations
0186 |       - ood_head: Out-of-distribution detection
0187 |       - temp_head: Adaptive temperature for score calibration
0188 |       - recon_head: Input reconstruction for comprehension verification
0189 |       
0190 |     Args:
0191 |         cfg: Configuration dictionary containing model hyperparameters
0192 |         logger: Optional logger instance for training diagnostics
0193 |     """
0194 | 
0195 |     def __init__(self, cfg: Dict[str, Any], logger: Optional[logging.Logger] = None):
0196 |         super().__init__()
0197 |         self.logger = logger or log
0198 | 
0199 |         # Model hyperparameters with type conversion and defaults
0200 |         self.input_dim = int(cfg.get("input_dim", 2048))  # Input embedding dimension
0201 |         self.h_dim = int(cfg.get("h_dim", 256))           # High-level hidden dimension
0202 |         self.l_dim = int(cfg.get("l_dim", 128))           # Low-level hidden dimension
0203 |         self.n_cycles = int(cfg.get("n_cycles", 4))       # Number of H cycles
0204 |         self.t_steps = int(cfg.get("t_steps", 4))         # L steps per H cycle
0205 |         self.dropout = float(cfg.get("dropout", 0.1))     # Dropout probability
0206 |         self.consistency_mask_p = float(cfg.get("consistency_mask_p", 0.10))  # Mask probability
0207 |         self.fd_eps = float(cfg.get("fd_eps", 1e-3))      # Finite-difference epsilon
0208 | 
0209 |         # Device management (updated during .to() calls)
0210 |         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
0211 | 
0212 |         # Input projection module
0213 |         self.input_projector = InputProjector(
0214 |             self.input_dim, self.h_dim, dropout=self.dropout
0215 |         )
0216 | 
0217 |         # Hierarchical recurrent modules
0218 |         # L-module: Fine-grained processing with access to input and H-state
0219 |         self.l_module = RecurrentBlock(2 * self.h_dim, self.l_dim, name="LModule")
0220 |         # H-module: Abstract reasoning integrating L-states and previous H-state
0221 |         self.h_module = RecurrentBlock(self.l_dim + self.h_dim, self.h_dim, name="HModule")
0222 | 
0223 |         # Multi-head prediction layer with dropout
0224 |         self.head_drop = nn.Dropout(self.dropout)
0225 |         
0226 |         # Diagnostic prediction heads
0227 |         self.score_head = nn.Linear(self.h_dim, 1)        # Quality score logits
0228 |         self.logvar_head = nn.Linear(self.h_dim, 1)       # Aleatoric uncertainty
0229 |         self.aux3_head = nn.Linear(self.h_dim, 3)         # 3-way classification
0230 |         self.disagree_head = nn.Linear(self.h_dim, 1)     # Disagreement prediction
0231 |         self.consistency_head = nn.Linear(self.h_dim, 1)  # Robustness prediction
0232 |         self.ood_head = nn.Linear(self.h_dim, 1)          # OOD detection
0233 |         self.temp_head = nn.Linear(self.h_dim, 1)         # Temperature calibration
0234 |         self.recon_head = nn.Linear(self.h_dim, self.h_dim)  # Input reconstruction
0235 | 
0236 |         # Final normalization for head inputs
0237 |         self.final_norm = RMSNorm(self.h_dim)
0238 | 
0239 |     # ---------------------------
0240 |     # Core Hierarchical Rollout
0241 |     # ---------------------------
0242 | 
0243 |     def _rollout(
0244 |         self,
0245 |         x_tilde: torch.Tensor,
0246 |         *,
0247 |         max_cycles: Optional[int] = None,
0248 |         tap: Optional[TraceTap] = None,
0249 |     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
0250 |         """
0251 |         Execute hierarchical recurrent processing across cycles and steps.
0252 |         
0253 |         Processing Flow:
0254 |           1. For each of n_cycles:
0255 |              a. L-module runs t_steps with access to (x_tilde, current zH)
0256 |              b. H-module updates once using (final zL, previous zH)
0257 |           2. Track maximum H-state for evidence accumulation
0258 |         
0259 |         Args:
0260 |             x_tilde: Projected input features [batch, h_dim]
0261 |             
0262 |         Returns:
0263 |             zL_final: Final low-level state [batch, l_dim]
0264 |             zH_final: Final high-level state [batch, h_dim] 
0265 |             zH_traj_max: Maximum H-state across trajectory [batch, h_dim]
0266 |         """
0267 |         batch_size = x_tilde.size(0)
0268 |         
0269 |         # Initialize hidden states
0270 |         zL = RecurrentBlock.init_state(batch_size, self.l_dim, self.device)  # Low-level
0271 |         zH = RecurrentBlock.init_state(batch_size, self.h_dim, self.device)  # High-level
0272 |         zH_max = torch.zeros_like(zH)  # Track maximum activation
0273 | 
0274 |         # Hierarchical recurrent processing
0275 |         cycles = self.n_cycles if max_cycles is None else int(max_cycles)
0276 |         cycles = max(1, min(cycles, self.n_cycles))
0277 |         for cycle in range(cycles):            # Low-level fine-grained processing (T steps)
0278 |             for step in range(self.t_steps):
0279 |                 # L-module input: projected input + current H-state
0280 |                 l_input = torch.cat([x_tilde, zH], dim=-1)  # [batch, 2 * h_dim]
0281 |                 zL = self.l_module(zL, l_input)
0282 |                 if tap is not None:
0283 |                     tap.add("hrm/zL", zL)
0284 | 
0285 |             # High-level abstract update (1 step per cycle)
0286 |             # H-module input: final L-state + previous H-state
0287 |             h_input = torch.cat([zL, zH], dim=-1)  # [batch, l_dim + h_dim]
0288 |             zH = self.h_module(zH, h_input)
0289 |             if tap is not None:
0290 |                 tap.add("hrm/zH", zH)            
0291 | 
0292 |             # Track maximum activation for evidence accumulation
0293 |             zH_max = torch.maximum(zH_max, zH)
0294 | 
0295 |         # Final normalization for prediction heads
0296 |         zH = self.final_norm(zH)
0297 |         zH_max = self.final_norm(zH_max)
0298 |         
0299 |         if tap is not None:
0300 |             tap.add("hrm/zL_final", zL)
0301 |             tap.add("hrm/zH_final", zH)
0302 |         return zL, zH, zH_max
0303 | 
0304 |     # ---------------------------
0305 |     # Main Forward Pass
0306 |     # ---------------------------
0307 | 
0308 |     def forward(
0309 |         self,
0310 |         x: torch.Tensor,                      # Input embeddings [batch, input_dim]
0311 |         *,
0312 |         return_aux: bool = True,              # Return auxiliary diagnostics
0313 |         n_steps: Optional[int] = None,
0314 |         tap: Optional["TraceTap"] = None,
0315 |     ) -> Tuple[torch.Tensor, Dict[str, Any]]:
0316 |         """
0317 |         Complete forward pass with hierarchical reasoning and multi-head prediction.
0318 |         
0319 |         Args:
0320 |             x: Input embeddings (typically goal ⊕ response or plan)
0321 |             return_aux: Whether to compute and return auxiliary outputs
0322 |             
0323 |         Returns:
0324 |             score01: Primary quality score ∈ [0,1] [batch, 1]
0325 |             aux: Dictionary of raw and derived diagnostic outputs
0326 |         """
0327 | 
0328 | 
0329 |         # Input projection and hierarchical processing
0330 |         x_tilde = self.input_projector(x)     # [batch, h_dim]
0331 |         zL, zH, zH_max = self._rollout(x_tilde, max_cycles=self.n_cycles, tap=tap)
0332 |         
0333 |         # Prepare state for prediction heads
0334 |         zH_head = self.head_drop(zH)  # Regularization
0335 | 
0336 |         # Temperature-calibrated scoring
0337 |         tau_raw = self.temp_head(zH_head)
0338 |         tau = 0.5 + 0.5 * F.softplus(tau_raw)  # τ ∈ (0.5, ∞) with softplus
0339 |         temp01  = torch.sigmoid(tau_raw)              # nice bounded proxy for telemetry
0340 |         score_logit = self.score_head(zH_head)
0341 |         score01 = torch.sigmoid(score_logit / tau)  # Calibrated score ∈ [0,1]
0342 | 
0343 |         # Core diagnostic heads
0344 |         log_var = self.logvar_head(zH_head)           # Aleatoric uncertainty
0345 |         aux3_logits = self.aux3_head(zH_head)         # 3-way classification
0346 |         aux3_probs = F.softmax(aux3_logits, dim=-1)   # Probability distribution
0347 |         disagree_hat = torch.sigmoid(self.disagree_head(zH_head))  # Disagreement
0348 |         ood_hat = torch.sigmoid(self.ood_head(zH_head))           # OOD probability
0349 | 
0350 |         # Consistency regularization target
0351 |         mask = (torch.rand_like(zH_head) < self.consistency_mask_p).float()
0352 |         zH_masked = zH_head * (1.0 - mask)  # Randomly masked state
0353 |         consistency_hat = torch.sigmoid(self.consistency_head(zH_head))
0354 |         consistency_target = self._cos01(zH_head, zH_masked).unsqueeze(-1)
0355 | 
0356 |         # Input reconstruction (comprehension proxy)
0357 |         x_recon = self.recon_head(zH_head)  # Reconstruct projected input
0358 |         recon_sim = self._cos01(x_recon, x_tilde).unsqueeze(-1)  # Reconstruction quality
0359 | 
0360 |         # Finite-difference sensitivity analysis
0361 |         x_eps = x + self.fd_eps * F.normalize(torch.randn_like(x), dim=-1)
0362 |         with torch.no_grad():
0363 |             x_tilde_eps = self.input_projector(x_eps)
0364 |             _, zH_eps, _ = self._rollout(x_tilde_eps)
0365 |             zH_eps = self.head_drop(zH_eps)
0366 |             tau_eps = 0.5 + 0.5 * F.softplus(self.temp_head(zH_eps))
0367 |             score_eps = torch.sigmoid(self.score_head(zH_eps) / tau_eps)
0368 |         
0369 |         # Jacobian approximation via finite differences
0370 |         jacobian_fd = ((score_eps - score01).abs() / self.fd_eps).clamp(0, 10.0) / 10.0
0371 | 
0372 |         # Pseudo-halting signal (evidence accumulation)
0373 |         halt_logit = (zH_max * zH_head).mean(-1, keepdim=True) / max(self.h_dim, 1)
0374 |         halt_prob = torch.sigmoid(halt_logit)
0375 | 
0376 |         # Early return if only primary score needed
0377 |         if not return_aux:
0378 |             return score01, {}
0379 | 
0380 |         # Comprehensive auxiliary outputs dictionary
0381 |         aux: Dict[str, Any] = {
0382 |             # Raw head outputs (for loss computation)
0383 |             "score_logit": score_logit,                 # [batch, 1]
0384 |             "log_var": log_var,                         # [batch, 1]  
0385 |             "aux3_logits": aux3_logits,                 # [batch, 3]
0386 |             "disagree_logit": self.disagree_head(zH_head),  # [batch, 1]
0387 |             "consistency_logit": self.consistency_head(zH_head),  # [batch, 1]
0388 |             "x_recon": x_recon,                         # [batch, h_dim]
0389 | 
0390 |             # Derived metrics (normalized for visualization)
0391 |             "score": score01,                           # [batch, 1] ∈ [0,1]
0392 |             "certainty01": torch.sigmoid(-log_var),     # [batch, 1] certainty measure
0393 |             "uncertainty": 1.0 - torch.sigmoid(-log_var),     # [batch, 1] alias for back-compat
0394 |             "aux3_probs": aux3_probs,                   # [batch, 3] probability distribution
0395 |             "entropy_aux": (-(aux3_probs * F.log_softmax(aux3_logits, -1)).sum(-1)
0396 |                              / torch.log(torch.tensor(3.0, device=x.device))).unsqueeze(-1),  # [batch, 1]
0397 |             "disagree_hat": disagree_hat,               # [batch, 1] predicted disagreement
0398 |             "consistency_hat": consistency_hat,         # [batch, 1] robustness prediction
0399 |             "consistency_target": consistency_target,   # [batch, 1] regularization target
0400 |             "recon_sim": recon_sim,                     # [batch, 1] reconstruction quality
0401 |             "ood_hat": ood_hat,                         # [batch, 1] OOD probability
0402 |             "temp01": temp01,                           # [batch, 1] temperature proxy
0403 |             "jacobian_fd": jacobian_fd,                 # [batch, 1] input sensitivity
0404 |             "halt_prob": halt_prob,                     # [batch, 1] evidence accumulation
0405 | 
0406 |             # Internal states (for introspection/debugging)
0407 |             "zL_final": zL,                            # [batch, l_dim] final L-state
0408 |             "zH_final": zH,                            # [batch, h_dim] final H-state
0409 |         }
0410 |         
0411 |         return score01, aux
0412 | 
0413 |     # ---------------------------
0414 |     # Utility Methods
0415 |     # ---------------------------
0416 | 
0417 |     @staticmethod
0418 |     def _cos01(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
0419 |         """
0420 |         Compute cosine similarity mapped from [-1, 1] to [0, 1].
0421 |         
0422 |         Args:
0423 |             a, b: Input tensors to compare
0424 |             dim: Dimension for cosine computation
0425 |             eps: Numerical stability constant
0426 |             
0427 |         Returns:
0428 |             Cosine similarity normalized to [0, 1] range
0429 |         """
0430 |         sim = F.cosine_similarity(a, b, dim=dim, eps=eps)
0431 |         return (sim + 1.0) * 0.5
0432 | 
0433 |     def to(self, device):
0434 |         """
0435 |         Move model to specified device and update internal device reference.
0436 |         
0437 |         Args:
0438 |             device: Target device (cuda/cpu)
0439 |             
0440 |         Returns:
0441 |             self: Model instance on target device
0442 |         """
0443 |         super().to(device)
0444 |         self.device = device
0445 |         return self
0446 | 
0447 |     def self_test(self, *, device: str = "cpu", n_trials: int = 16) -> Dict[str, Any]:
0448 |         """
0449 |         Quick sanity check to detect:
0450 |         - constant outputs
0451 |         - always-near-zero outputs
0452 |         - exploding temperature
0453 |         - collapsed latents (zH/zL near 0)
0454 |         """
0455 |         from stephanie.model.model_selftest import (ModelSelfTest,
0456 |                                                     summarize_selftest)
0457 | 
0458 |         input_dim = int(getattr(self, "input_dim", 0) or getattr(self, "cfg", {}).get("input_dim", 0) or 0)
0459 |         if input_dim <= 0:
0460 |             # fall back: infer from projector weight
0461 |             for name, p in self.named_parameters():
0462 |                 if name.endswith("input_projector.project.weight") and p.ndim == 2:
0463 |                     input_dim = int(p.shape[1])
0464 |                     break
0465 | 
0466 |         def build_inputs():
0467 |             B = 8
0468 |             x = torch.randn(B, input_dim)
0469 |             return {"x": x, "return_aux": True}
0470 | 
0471 |         def extract_debug(aux: Any):
0472 |             if not isinstance(aux, dict):
0473 |                 return {}
0474 | 
0475 |             out = {}
0476 | 
0477 |             # keep these always (cheap + most diagnostic)
0478 |             always = {"score_logit", "temp01", "log_var", "entropy_aux", "jacobian_fd", "halt_prob", "ood_hat", "disagree_hat",
0479 |                     "zH_final", "zL_final", "recon_sim"}
0480 | 
0481 |             # avoid heavy tensors unless small
0482 |             max_numel = 4096  # tune later
0483 | 
0484 |             for k, v in aux.items():
0485 |                 if k in always:
0486 |                     out[k] = v
0487 |                     continue
0488 | 
0489 |                 if torch.is_tensor(v):
0490 |                     # include small tensors (or you can exclude known huge keys like x_recon)
0491 |                     if v.numel() <= max_numel:
0492 |                         out[k] = v
0493 |                 else:
0494 |                     # include simple scalar-ish values
0495 |                     if isinstance(v, (int, float, bool, str)):
0496 |                         out[k] = v
0497 | 
0498 |             return out
0499 | 
0500 |         tester = ModelSelfTest(
0501 |             name="HRMModel",
0502 |             build_inputs=build_inputs,
0503 |             extract_debug=extract_debug,
0504 |             device=device,
0505 |             n_trials=n_trials,
0506 |         )
0507 |         res = tester.run(self)
0508 |         return {"ok": res.ok, "summary": summarize_selftest(res), "details": res.details}
```


---

## F0005 — `knowledge.py`

```text
FILE_ID: F0005
PATH: knowledge.py
LANGUAGE: python
LINES: 317
BYTES_UTF8: 10920
SHA256: c21cca5d1b7e9027ebd61ce542b97e61b6f26cbf7522c62d188f756386e04f75
```

```python
0001 | # stephanie/model/knowledge.py
0002 | from __future__ import annotations
0003 | 
0004 | import json
0005 | import os
0006 | from typing import List, Optional
0007 | 
0008 | import torch
0009 | import torch.nn as nn
0010 | 
0011 | 
0012 | class CrossFeatureEncoder(nn.Module):
0013 |     def __init__(self, dim: int, hdim: int):
0014 |         super().__init__()
0015 |         self.dim = dim
0016 |         self.hdim = hdim
0017 |         self.bilinear = nn.Bilinear(dim, dim, hdim, bias=False)
0018 |         self.proj = nn.Sequential(
0019 |             nn.Linear(dim * 4, hdim),
0020 |             nn.ReLU(),
0021 |             nn.Linear(hdim, hdim),
0022 |             nn.ReLU()
0023 |         )
0024 | 
0025 |     def forward(self, goal: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
0026 |         z_bi = self.bilinear(goal, text)
0027 |         z_feats = torch.cat([goal, text, goal * text, torch.abs(goal - text)], dim=-1)
0028 |         z_proj = self.proj(z_feats)
0029 |         return z_bi + z_proj
0030 | 
0031 | 
0032 | class AuxProjector(nn.Module):
0033 |     def __init__(self, hdim: int, aux_dim: int):
0034 |         super().__init__()
0035 |         self.aux_dim = aux_dim
0036 |         if aux_dim > 0:
0037 |             self.mlp = nn.Sequential(
0038 |                 nn.Linear(aux_dim, hdim),
0039 |                 nn.ReLU(),
0040 |                 nn.Linear(hdim, hdim)
0041 |             )
0042 |         else:
0043 |             self.mlp = None
0044 | 
0045 |     def forward(self, z: torch.Tensor, aux: Optional[torch.Tensor]) -> torch.Tensor:
0046 |         if self.mlp is None or aux is None:
0047 |             return z
0048 |         return z + self.mlp(aux)
0049 | 
0050 | 
0051 | class KnowledgePredictor(nn.Module):
0052 |     def __init__(self, hdim: int):
0053 |         super().__init__()
0054 |         self.mlp = nn.Sequential(
0055 |             nn.Linear(hdim, hdim),
0056 |             nn.ReLU(),
0057 |             nn.Linear(hdim, hdim // 2),
0058 |             nn.ReLU(),
0059 |             nn.Linear(hdim // 2, 1)
0060 |         )
0061 | 
0062 |     def forward(self, z: torch.Tensor) -> torch.Tensor:
0063 |         return self.mlp(z).squeeze(-1)  # [B]
0064 | 
0065 | 
0066 | class KnowledgeModel:
0067 |     """
0068 |     Two-head knowledge scorer:
0069 |       - predictor_h: human (gold) head
0070 |       - predictor_a: AI (weak) head
0071 |     """
0072 |     def __init__(self, dim: int, hdim: int, embedding_store, aux_feature_names: Optional[List[str]] = None, device: str = "cpu"):
0073 |         self.device = device
0074 |         self.embedding_store = embedding_store
0075 |         self.aux_feature_names = aux_feature_names or []
0076 |         self.encoder = CrossFeatureEncoder(dim, hdim).to(device)
0077 |         self.aux_proj = AuxProjector(hdim, aux_dim=len(self.aux_feature_names)).to(device)
0078 |         # heads
0079 |         self.predictor_h = KnowledgePredictor(hdim).to(device)
0080 |         self.predictor_a = KnowledgePredictor(hdim).to(device)
0081 | 
0082 |     # ---- scoring API ----
0083 |     def score_h(self, z: torch.Tensor) -> torch.Tensor:
0084 |         return self.predictor_h(z).squeeze(-1)
0085 | 
0086 |     def score_a(self, z: torch.Tensor) -> torch.Tensor:
0087 |         return self.predictor_a(z).squeeze(-1)
0088 | 
0089 |     @torch.no_grad()
0090 |     def predict(
0091 |         self,
0092 |         goal_text: str,
0093 |         candidate_text: str,
0094 |         meta: Optional[dict] = None,
0095 |         *,
0096 |         return_components: bool = False,
0097 |     ) -> float | tuple[float, dict]:
0098 |         """
0099 |         Returns a blended probability in [0,1].
0100 |         If return_components=True, also returns a dict with attribution details.
0101 |         """
0102 |         meta = meta or {}
0103 | 
0104 |         # --- encode ---
0105 |         g = self._embed(goal_text)                 # [1,D]
0106 |         x = self._embed(candidate_text)            # [1,D]
0107 |         z = self.encoder(g, x)                     # [1,H]
0108 |         aux = self._aux_tensor(meta)               # [1,A] or None
0109 |         z = self.aux_proj(z, aux)                  # [1,H]
0110 | 
0111 |         # --- logits -> probs ---
0112 |         s_h = self.score_h(z)                      # [1]
0113 |         s_a = self.score_a(z)                      # [1]
0114 |         s_h_val = float(s_h.item())
0115 |         s_a_val = float(s_a.item())
0116 |         h_prob = float(torch.sigmoid(torch.tensor(s_h_val)).item())
0117 |         a_prob = float(torch.sigmoid(torch.tensor(s_a_val)).item())
0118 | 
0119 |         # --- blending rule (human-first) ---
0120 |         has_similar_human = bool(meta.get("has_similar_human", False))
0121 |         alpha = 1.0 if has_similar_human else 0.6
0122 |         p = alpha * h_prob + (1.0 - alpha) * a_prob
0123 | 
0124 |         if not return_components:
0125 |             return p
0126 | 
0127 |         # components & fractions
0128 |         human_component = alpha * h_prob
0129 |         ai_component    = (1.0 - alpha) * a_prob
0130 |         denom = human_component + ai_component
0131 |         if denom > 0.0:
0132 |             human_fraction = human_component / denom
0133 |             ai_fraction    = ai_component / denom
0134 |         else:
0135 |             human_fraction = ai_fraction = 0.5  # guard
0136 | 
0137 |         details = {
0138 |             # final
0139 |             "probability": float(p),
0140 | 
0141 |             # raw head signals
0142 |             "human_logit": round(s_h_val, 6),
0143 |             "ai_logit": round(s_a_val, 6),
0144 |             "human_prob": round(h_prob, 6),
0145 |             "ai_prob": round(a_prob, 6),
0146 | 
0147 |             # blending
0148 |             "alpha_human_weight": float(alpha),
0149 |             "has_similar_human": has_similar_human,
0150 | 
0151 |             # contributions
0152 |             "human_component": round(human_component, 6),
0153 |             "ai_component": round(ai_component, 6),
0154 |             "human_fraction": round(human_fraction, 6),
0155 |             "ai_fraction": round(ai_fraction, 6),
0156 |         }
0157 |         return p, details
0158 | 
0159 |     def _blend_scores(self, s_h: float, s_a: float, meta: Optional[dict] = None) -> float:
0160 |         meta = meta or {}
0161 |         has_similar_human = bool(meta.get("has_similar_human", False))
0162 |         # human-first sigmoid blending
0163 |         h = torch.sigmoid(torch.tensor(s_h)).item()
0164 |         a = torch.sigmoid(torch.tensor(s_a)).item()
0165 |         if has_similar_human:
0166 |             return h
0167 |         alpha = 0.6  # bias toward human
0168 |         return alpha * h + (1 - alpha) * a
0169 | 
0170 |     # ---- utils ----
0171 |     def _embed(self, text: str) -> torch.Tensor:
0172 |         v = self.embedding_store.get_or_create(text)
0173 |         t = torch.tensor(v, device=self.device, dtype=torch.float32).unsqueeze(0)
0174 |         t = t / (t.norm(dim=-1, keepdim=True) + 1e-12)
0175 |         return t
0176 | 
0177 |     def _aux_tensor(self, meta: Optional[dict]) -> Optional[torch.Tensor]:
0178 |         if not self.aux_feature_names:
0179 |             return None
0180 |         meta = meta or {}
0181 |         vals = []
0182 |         for name in self.aux_feature_names:
0183 |             try:
0184 |                 vals.append(float(meta.get(name, 0.0)))
0185 |             except Exception:
0186 |                 vals.append(0.0)
0187 |         return torch.tensor(vals, device=self.device, dtype=torch.float32).unsqueeze(0)
0188 | 
0189 |     def train(self):
0190 |         self.encoder.train()
0191 |         self.aux_proj.train()
0192 |         self.predictor_h.train()
0193 |         self.predictor_a.train()
0194 | 
0195 |     def eval(self):
0196 |         self.encoder.eval()
0197 |         self.aux_proj.eval()
0198 |         self.predictor_h.eval()
0199 |         self.predictor_a.eval()
0200 | 
0201 |     def save(
0202 |         self,
0203 |         *,
0204 |         encoder_path: str,
0205 |         head_h_path: str,     # human head
0206 |         head_a_path: str,     # AI head
0207 |         auxproj_path: str,
0208 |         manifest_path: str | None = None,
0209 |         extra: dict | None = None,       # e.g., version, dim, hdim, aux names
0210 |     ) -> dict:
0211 |         os.makedirs(os.path.dirname(os.path.abspath(encoder_path)), exist_ok=True)
0212 |         os.makedirs(os.path.dirname(os.path.abspath(head_h_path)), exist_ok=True)
0213 |         os.makedirs(os.path.dirname(os.path.abspath(head_a_path)), exist_ok=True)
0214 |         os.makedirs(os.path.dirname(os.path.abspath(auxproj_path)), exist_ok=True)
0215 | 
0216 |         torch.save(self.encoder.state_dict(), encoder_path)
0217 |         torch.save(self.predictor_h.state_dict(), head_h_path)
0218 |         torch.save(self.predictor_a.state_dict(), head_a_path)
0219 |         torch.save(self.aux_proj.state_dict(), auxproj_path)
0220 | 
0221 |         manifest = {
0222 |             "format": "knowledge.v1",
0223 |             "device": str(self.device),
0224 |             "dim": getattr(self.encoder, "dim", None),
0225 |             "hdim": getattr(self.encoder, "hdim", None),
0226 |             "aux_features": self.aux_feature_names,
0227 |             "files": {
0228 |                 "encoder": os.path.basename(encoder_path),
0229 |                 "head_h":  os.path.basename(head_h_path),
0230 |                 "head_a":  os.path.basename(head_a_path),
0231 |                 "auxproj": os.path.basename(auxproj_path),
0232 |             }, 
0233 |         }
0234 |         if extra:
0235 |             manifest["extra"] = extra
0236 | 
0237 |         if manifest_path:
0238 |             os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
0239 |             import json
0240 |             with open(manifest_path, "w", encoding="utf-8") as f:
0241 |                 json.dump(manifest, f, indent=2)
0242 | 
0243 |         return manifest
0244 | 
0245 |     @classmethod
0246 |     def load(
0247 |         cls,
0248 |         *,
0249 |         dim: int,
0250 |         hdim: int,
0251 |         embedding_store,
0252 |         aux_feature_names: list[str] | None,
0253 |         device: str,
0254 |         encoder_path: str,
0255 |         head_h_path: str,
0256 |         head_a_path: str,
0257 |         auxproj_path: str,
0258 |     ) -> KnowledgeModel:
0259 |         model = cls(dim=dim, hdim=hdim, embedding_store=embedding_store,
0260 |                     aux_feature_names=aux_feature_names, device=device)
0261 |         model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
0262 |         model.predictor_h.load_state_dict(torch.load(head_h_path, map_location=device))
0263 |         model.predictor_a.load_state_dict(torch.load(head_a_path, map_location=device))
0264 |         model.aux_proj.load_state_dict(torch.load(auxproj_path, map_location=device))
0265 |         model.eval()
0266 |         return model
0267 | 
0268 |     def save_bundle(
0269 |         self,
0270 |         dir_path: str,
0271 |         *,
0272 |         extra: dict | None = None,
0273 |     ) -> dict:
0274 |         os.makedirs(dir_path, exist_ok=True)
0275 |         encoder = os.path.join(dir_path, "encoder.pt")
0276 |         head_h  = os.path.join(dir_path, "head_h.pt")
0277 |         head_a  = os.path.join(dir_path, "head_a.pt")
0278 |         auxproj = os.path.join(dir_path, "auxproj.pt")
0279 |         manifest_path = os.path.join(dir_path, "manifest.json")
0280 | 
0281 |         return self.save(
0282 |             encoder_path=encoder,
0283 |             head_h_path=head_h,
0284 |             head_a_path=head_a,
0285 |             auxproj_path=auxproj,
0286 |             manifest_path=manifest_path,
0287 |             extra=extra,
0288 |         )
0289 | 
0290 |     @classmethod
0291 |     def load_bundle(
0292 |         cls,
0293 |         dir_path: str,
0294 |         *,
0295 |         embedding_store,
0296 |         device: str = "cpu",
0297 |     ) -> KnowledgeModel:
0298 |         manifest_path = os.path.join(dir_path, "manifest.json")
0299 |         with open(manifest_path, "r", encoding="utf-8") as f:
0300 |             m = json.load(f)
0301 | 
0302 |         dim   = int(m.get("dim"))
0303 |         hdim  = int(m.get("hdim"))
0304 |         aux   = m.get("aux_features") or []
0305 |         files = m["files"]
0306 | 
0307 |         return cls.load(
0308 |             dim=dim,
0309 |             hdim=hdim,
0310 |             embedding_store=embedding_store,
0311 |             aux_feature_names=aux,
0312 |             device=device,
0313 |             encoder_path=os.path.join(dir_path, os.path.basename(files["encoder"])),
0314 |             head_h_path=os.path.join(dir_path, os.path.basename(files["head_h"])),
0315 |             head_a_path=os.path.join(dir_path, os.path.basename(files["head_a"])),
0316 |             auxproj_path=os.path.join(dir_path, os.path.basename(files["auxproj"])),
0317 |         )
```


---

## F0006 — `model_locator_mixin.py`

```text
FILE_ID: F0006
PATH: model_locator_mixin.py
LANGUAGE: python
LINES: 74
BYTES_UTF8: 2438
SHA256: 850b51b53e3f4f473efac34092a5b970aa41551831b23e3f250dae6464a74771
```

```python
0001 | # stephanie/model/model_locator_mixin.py
0002 | from __future__ import annotations
0003 | 
0004 | import os
0005 | 
0006 | 
0007 | class ModelLocatorMixin:
0008 |     class Locator:
0009 |         def __init__(
0010 |             self,
0011 |             root_dir: str,
0012 |             model_type: str,
0013 |             target_type: str,
0014 |             dimension: str,
0015 |             version: str,
0016 |             embedding_type: str,
0017 |         ):
0018 |             self.root_dir = root_dir
0019 |             self.model_type = model_type
0020 |             self.target_type = target_type
0021 |             self.dimension = dimension
0022 |             self.version = version
0023 |             self.embedding_type = embedding_type
0024 | 
0025 |         @property
0026 |         def base_path(self) -> str:
0027 |             path = os.path.join(
0028 |                 self.root_dir,
0029 |                 self.embedding_type,
0030 |                 self.model_type,
0031 |                 self.target_type,
0032 |                 self.dimension,
0033 |                 self.version,
0034 |             )
0035 |             os.makedirs(path, exist_ok=True)
0036 |             return path
0037 | 
0038 |         # Model-specific paths
0039 |         def model_file(self, suffix: str = ".pt") -> str:
0040 |             return os.path.join(self.base_path, f"{self.dimension}{suffix}")
0041 | 
0042 |         def encoder_file(self) -> str:
0043 |             return os.path.join(self.base_path, f"{self.dimension}_encoder.pt")
0044 | 
0045 |         def get_q_head_path(self) -> str:
0046 |             return os.path.join(self.base_path, f"{self.dimension}_q.pt")
0047 | 
0048 |         def get_v_head_path(self) -> str:
0049 |             return os.path.join(self.base_path, f"{self.dimension}_v.pt")
0050 | 
0051 |         def get_pi_head_path(self) -> str:
0052 |             return os.path.join(self.base_path, f"{self.dimension}_pi.pt")
0053 | 
0054 |         def meta_file(self) -> str:
0055 |             return os.path.join(self.base_path, f"{self.dimension}.meta.json")
0056 | 
0057 |         def tuner_file(self) -> str:
0058 |             return os.path.join(self.base_path, f"{self.dimension}.tuner.json")
0059 | 
0060 |         def scaler_file(self) -> str:
0061 |             return os.path.join(self.base_path, f"{self.dimension}_scaler.joblib")
0062 | 
0063 |     def get_model_name(self) -> str:
0064 |         return f"{self.target_type}_{self.model_type}_{self.model_version}"
0065 | 
0066 |     def get_locator(self, dimension: str):
0067 |         return self.Locator(
0068 |             root_dir=self.model_path,  # Path to the root directory for models
0069 |             model_type=self.model_type,
0070 |             target_type=self.target_type,
0071 |             dimension=dimension,
0072 |             version=self.version,
0073 |             embedding_type=self.embedding_type,
0074 |         )
```


---

## F0007 — `model_selftest.py`

```text
FILE_ID: F0007
PATH: model_selftest.py
LANGUAGE: python
LINES: 162
BYTES_UTF8: 5177
SHA256: 56da08e30b90414eef43d8403498b669660bca2ce29692c455c84925ff1dd161
```

```python
0001 | # stephanie/model/model_selftest.py
0002 | from __future__ import annotations
0003 | 
0004 | import math
0005 | from dataclasses import dataclass
0006 | from typing import Any, Callable, Dict, List
0007 | 
0008 | import torch
0009 | 
0010 | 
0011 | def _tstats(x: torch.Tensor) -> Dict[str, float]:
0012 |     x = x.detach().float().cpu()
0013 |     if x.numel() == 0:
0014 |         return {"n": 0.0}
0015 |     return {
0016 |         "n": float(x.numel()),
0017 |         "shape0": float(x.shape[0]) if x.ndim >= 1 else 0.0,
0018 |         "mean": float(x.mean().item()),
0019 |         "std": float(x.std(unbiased=False).item()),
0020 |         "min": float(x.min().item()),
0021 |         "max": float(x.max().item()),
0022 |         "l2": float(torch.norm(x).item()),
0023 |         "finite_frac": float(torch.isfinite(x).float().mean().item()),
0024 |     }
0025 | 
0026 | 
0027 | @dataclass
0028 | class SelfTestResult:
0029 |     ok: bool
0030 |     name: str
0031 |     details: Dict[str, Any]
0032 | 
0033 | 
0034 | class ModelSelfTest:
0035 |     """
0036 |     Generic self-test harness for Stephanie models.
0037 |     You provide:
0038 |       - build_inputs() -> kwargs for model(...)
0039 |       - extract_debug(aux) -> dict of tensors/scalars to inspect
0040 |     """
0041 | 
0042 |     def __init__(
0043 |         self,
0044 |         *,
0045 |         name: str,
0046 |         build_inputs: Callable[[], Dict[str, Any]],
0047 |         extract_debug: Callable[[Any], Dict[str, Any]],
0048 |         device: str = "cpu",
0049 |         n_trials: int = 16,
0050 |         seed: int = 1337,
0051 |         warn_only: bool = True,
0052 |     ) -> None:
0053 |         self.name = name
0054 |         self.build_inputs = build_inputs
0055 |         self.extract_debug = extract_debug
0056 |         self.device = device
0057 |         self.n_trials = int(max(3, n_trials))
0058 |         self.seed = seed
0059 |         self.warn_only = warn_only
0060 | 
0061 |     @torch.no_grad()
0062 |     def run(self, model: torch.nn.Module) -> SelfTestResult:
0063 |         model = model.to(self.device)
0064 |         model.eval()
0065 | 
0066 |         torch.manual_seed(self.seed)
0067 | 
0068 |         scores: List[float] = []
0069 |         debug_accum: Dict[str, List[Dict[str, Any]]] = {}
0070 | 
0071 |         # Run multiple trials with fresh random inputs
0072 |         for i in range(self.n_trials):
0073 |             kw = self.build_inputs()
0074 |             # move tensors to device
0075 |             for k, v in kw.items():
0076 |                 if torch.is_tensor(v):
0077 |                     kw[k] = v.to(self.device)
0078 |             out = model(**kw)
0079 | 
0080 |             # Support (score, aux) or dict-like outputs
0081 |             if isinstance(out, tuple) and len(out) == 2:
0082 |                 score, aux = out
0083 |             else:
0084 |                 # allow direct score tensor
0085 |                 score, aux = out, {}
0086 | 
0087 |             # gather score stats
0088 |             s = (
0089 |                 float(score.detach().float().mean().item())
0090 |                 if torch.is_tensor(score)
0091 |                 else float(score)
0092 |             )
0093 |             scores.append(s)
0094 | 
0095 |             dbg = self.extract_debug(aux)
0096 |             for key, val in dbg.items():
0097 |                 if torch.is_tensor(val):
0098 |                     entry = _tstats(val)
0099 |                 else:
0100 |                     entry = {"value": val}
0101 |                 debug_accum.setdefault(key, []).append(entry)
0102 | 
0103 |         # Analyze score distribution
0104 |         s_tensor = torch.tensor(scores, dtype=torch.float32)
0105 |         s_mean = float(s_tensor.mean().item())
0106 |         s_std = float(s_tensor.std(unbiased=False).item())
0107 |         s_min = float(s_tensor.min().item())
0108 |         s_max = float(s_tensor.max().item())
0109 | 
0110 |         # Heuristics for “broken”
0111 |         # - near-constant output
0112 |         const_like = s_std < 1e-4
0113 |         # - saturated near 0 or near 1
0114 |         near_zero = s_mean < 1e-3 and s_max < 5e-3
0115 |         near_one = s_mean > 1.0 - 1e-3 and s_min > 1.0 - 5e-3
0116 |         # - non-finite scores
0117 |         nonfinite = not math.isfinite(s_mean) or not math.isfinite(s_std)
0118 | 
0119 |         ok = (
0120 |             (not nonfinite)
0121 |             and (not const_like)
0122 |             and (not near_zero)
0123 |             and (not near_one)
0124 |         )
0125 | 
0126 |         details: Dict[str, Any] = {
0127 |             "score": {
0128 |                 "mean": s_mean,
0129 |                 "std": s_std,
0130 |                 "min": s_min,
0131 |                 "max": s_max,
0132 |                 "const_like": const_like,
0133 |                 "near_zero": near_zero,
0134 |                 "near_one": near_one,
0135 |                 "nonfinite": nonfinite,
0136 |                 "samples": scores[: min(8, len(scores))],
0137 |             },
0138 |             "debug": debug_accum,
0139 |         }
0140 | 
0141 |         # if warn_only, we still return ok flag but don't raise
0142 |         return SelfTestResult(ok=ok, name=self.name, details=details)
0143 | 
0144 | 
0145 | def summarize_selftest(res: SelfTestResult) -> str:
0146 |     s = res.details.get("score", {})
0147 |     lines = [
0148 |         f"[{res.name}] ok={res.ok}",
0149 |         f"  score: mean={s.get('mean'):.6f} std={s.get('std'):.6f} min={s.get('min'):.6f} max={s.get('max'):.6f}",
0150 |         f"  flags: const_like={s.get('const_like')} near_zero={s.get('near_zero')} near_one={s.get('near_one')} nonfinite={s.get('nonfinite')}",
0151 |     ]
0152 |     # Show a couple of debug keys if present
0153 |     dbg = res.details.get("debug", {})
0154 |     for k in list(dbg.keys())[:4]:
0155 |         last = dbg[k][-1]
0156 |         if "mean" in last:
0157 |             lines.append(
0158 |                 f"  {k}: mean={last['mean']:.4f} std={last['std']:.4f} min={last['min']:.4f} max={last['max']:.4f} finite={last['finite_frac']:.3f}"
0159 |             )
0160 |         else:
0161 |             lines.append(f"  {k}: {last}")
0162 |     return "\n".join(lines)
```


---

## F0008 — `mrq.py`

```text
FILE_ID: F0008
PATH: mrq.py
LANGUAGE: python
LINES: 60
BYTES_UTF8: 2431
SHA256: 723ef1568047ad3d909841f6969d065b892d3c848786c2518e8be502079b978e
```

```python
0001 | # stephanie/model/mrq.py
0002 | from __future__ import annotations
0003 | 
0004 | import torch
0005 | 
0006 | 
0007 | class MRQModel:
0008 |     def __init__(self, encoder, predictor, embedding_store, device="cpu"):
0009 |         self.encoder = encoder.to(device)
0010 |         self.predictor = predictor.to(device)
0011 |         self.embedding_store = embedding_store
0012 |         self.device = device
0013 | 
0014 |     # --- NEW: make the model callable like a torch Module ---
0015 |     def __call__(self, ctx, doc, *, apply_sigmoid: bool = False, return_dict: bool = True):
0016 |         """
0017 |         Accepts ctx/doc embeddings as:
0018 |           - torch.Tensor [B, D] (preferred), or
0019 |           - numpy array / list (will be coerced).
0020 |         Returns:
0021 |           {"q_value": logits} by default (tensor [B]),
0022 |           or raw tensor if return_dict=False.
0023 |         If apply_sigmoid=True, q_value contains probabilities in [0,1].
0024 |         """
0025 |         ctx_t = torch.as_tensor(ctx, dtype=torch.float32, device=self.device)
0026 |         doc_t = torch.as_tensor(doc, dtype=torch.float32, device=self.device)
0027 |         if ctx_t.dim() == 1:
0028 |             ctx_t = ctx_t.unsqueeze(0)
0029 |         if doc_t.dim() == 1:
0030 |             doc_t = doc_t.unsqueeze(0)
0031 | 
0032 |         z = self.encoder(ctx_t, doc_t)              # [B, D']
0033 |         logits = self.predictor(z).view(-1)         # [B]
0034 |         out = torch.sigmoid(logits) if apply_sigmoid else logits
0035 |         return {"q_value": out} if return_dict else out
0036 | 
0037 |     # Optional: keep a PyTorch-style alias
0038 |     forward = __call__
0039 | 
0040 |     def predict(self, prompt_text: str, response_text: str, *, return_prob: bool = False) -> float:
0041 |         prompt_emb = torch.tensor(self.embedding_store.get_or_create(prompt_text),
0042 |                                   dtype=torch.float32, device=self.device).unsqueeze(0)
0043 |         response_emb = torch.tensor(self.embedding_store.get_or_create(response_text),
0044 |                                     dtype=torch.float32, device=self.device).unsqueeze(0)
0045 |         z = self.encoder(prompt_emb, response_emb)
0046 |         logit = self.predictor(z).view(-1)[0]
0047 |         return float(torch.sigmoid(logit) if return_prob else logit)
0048 | 
0049 |     def load_weights(self, encoder_path: str, predictor_path: str):
0050 |         self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
0051 |         self.encoder.eval()
0052 |         self.predictor.eval()
0053 | 
0054 |     def train(self):
0055 |         self.encoder.train()
0056 |         self.predictor.train()
0057 | 
0058 |     def eval(self):
0059 |         self.encoder.eval()
0060 |         self.predictor.eval()
```


---

## F0009 — `pacs_optimizer.py`

```text
FILE_ID: F0009
PATH: pacs_optimizer.py
LANGUAGE: python
LINES: 105
BYTES_UTF8: 3328
SHA256: 0aa450618ed67b9ce9026bc767e4cebae358d4294542d6ce2f1b290f450f5fb3
```

```python
0001 | # stephanie/model/pacs_optimizer.py
0002 | from __future__ import annotations
0003 | 
0004 | from typing import Optional
0005 | 
0006 | import torch
0007 | from torch.optim import Optimizer
0008 | 
0009 | 
0010 | class PACSOptimizer(Optimizer):
0011 |     """
0012 |     Preconditioned Adaptive Control of Stochasticity (PACS) Optimizer.
0013 | 
0014 |     Implements a variance-reduced, preconditioned gradient descent update 
0015 |     suitable for training models in the PACS framework.
0016 | 
0017 |     This optimizer maintains both:
0018 |     - A moving average of past gradients (variance reduction)
0019 |     - A diagonal preconditioner (scaling step size per-parameter)
0020 | 
0021 |     Args:
0022 |         params (iterable): model parameters
0023 |         lr (float): base learning rate
0024 |         beta (float): momentum for gradient averaging (default 0.9)
0025 |         eps (float): small value to avoid division by zero (default 1e-8)
0026 |         weight_decay (float): L2 penalty (default 0.0)
0027 |         preconditioner_decay (float): decay for preconditioner (default 0.999)
0028 |     """
0029 | 
0030 |     def __init__(
0031 |         self,
0032 |         params,
0033 |         lr: float = 1e-4,
0034 |         beta: float = 0.9,
0035 |         eps: float = 1e-8,
0036 |         weight_decay: float = 0.0,
0037 |         preconditioner_decay: float = 0.999,
0038 |     ):
0039 |         defaults = dict(
0040 |             lr=lr,
0041 |             beta=beta,
0042 |             eps=eps,
0043 |             weight_decay=weight_decay,
0044 |             preconditioner_decay=preconditioner_decay,
0045 |         )
0046 |         super().__init__(params, defaults)
0047 | 
0048 |     @torch.no_grad()
0049 |     def step(self, closure: Optional[callable] = None):
0050 |         """
0051 |         Performs a single optimization step.
0052 |         """
0053 |         loss = None
0054 |         if closure is not None:
0055 |             with torch.enable_grad():
0056 |                 loss = closure()
0057 | 
0058 |         for group in self.param_groups:
0059 |             lr = group["lr"]
0060 |             beta = group["beta"]
0061 |             eps = group["eps"]
0062 |             weight_decay = group["weight_decay"]
0063 |             preconditioner_decay = group["preconditioner_decay"]
0064 | 
0065 |             for p in group["params"]:
0066 |                 if p.grad is None:
0067 |                     continue
0068 | 
0069 |                 grad = p.grad.data
0070 | 
0071 |                 # Apply weight decay
0072 |                 if weight_decay != 0:
0073 |                     grad = grad.add(p.data, alpha=weight_decay)
0074 | 
0075 |                 state = self.state[p]
0076 | 
0077 |                 # State initialization
0078 |                 if len(state) == 0:
0079 |                     state["step"] = 0
0080 |                     # Exponential moving average of gradient (variance reduction)
0081 |                     state["grad_avg"] = torch.zeros_like(p.data)
0082 |                     # Preconditioner accumulator (like RMSprop/Adam second moment)
0083 |                     state["precond"] = torch.zeros_like(p.data)
0084 | 
0085 |                 grad_avg = state["grad_avg"]
0086 |                 precond = state["precond"]
0087 | 
0088 |                 state["step"] += 1
0089 | 
0090 |                 # Update moving average of gradient
0091 |                 grad_avg.mul_(beta).add_(grad, alpha=1 - beta)
0092 | 
0093 |                 # Update preconditioner (running avg of squared grads)
0094 |                 precond.mul_(preconditioner_decay).addcmul_(
0095 |                     grad, grad, value=1 - preconditioner_decay
0096 |                 )
0097 | 
0098 |                 # Compute preconditioned gradient
0099 |                 denom = precond.sqrt().add_(eps)
0100 |                 step = grad_avg / denom
0101 | 
0102 |                 # Update parameters
0103 |                 p.data.add_(step, alpha=-lr)
0104 | 
0105 |         return loss
```


---

## F0010 — `preference_ranker.py`

```text
FILE_ID: F0010
PATH: preference_ranker.py
LANGUAGE: python
LINES: 30
BYTES_UTF8: 869
SHA256: 872fb2f26fd860cfdbf58c7ef418b8a9071c7f7049d72e0705a21a06b837a94b
```

```python
0001 | # stephanie/model/preference_ranker.py
0002 | from __future__ import annotations
0003 | 
0004 | import torch
0005 | import torch.nn as nn
0006 | 
0007 | 
0008 | class PreferenceRanker(nn.Module):
0009 |     """Siamese network for pairwise preference ranking"""
0010 |     def __init__(self, embedding_dim=768, hidden_dim=256):
0011 |         super().__init__()
0012 |         self.encoder = nn.Sequential(
0013 |             nn.Linear(embedding_dim, hidden_dim),
0014 |             nn.ReLU(),
0015 |             nn.Dropout(0.2),
0016 |             nn.Linear(hidden_dim, hidden_dim)
0017 |         )
0018 |         self.comparator = nn.Sequential(
0019 |             nn.Linear(hidden_dim * 2, hidden_dim),
0020 |             nn.ReLU(),
0021 |             nn.Linear(hidden_dim, 1)
0022 |         )
0023 | 
0024 |     def forward(self, emb_a, emb_b):
0025 |         feat_a = self.encoder(emb_a)
0026 |         feat_b = self.encoder(emb_b)
0027 |         combined = torch.cat([feat_a, feat_b], dim=1)
0028 |         return self.comparator(combined).squeeze(1)
```


---

## F0011 — `risk_predictor.py`

```text
FILE_ID: F0011
PATH: risk_predictor.py
LANGUAGE: python
LINES: 332
BYTES_UTF8: 11464
SHA256: e8e972ec3741a52990317845abebf9ef312cbf0382eb436b4b10fb5aae063ed1
```

```python
0001 | # stephanie/model/risk_predictor.py
0002 | """
0003 | Calibrated hallucination-risk scoring with per-domain thresholds.
0004 | 
0005 | Usage:
0006 |     pred = DomainCalibratedRiskPredictor(
0007 |         bundle_path="./models/risk/bundle.joblib",
0008 |         default_domains=["science","history","geography","tech"],
0009 |         memcube=MemCubeClient(),
0010 |         featurizer=RiskFeaturizer()
0011 |     )
0012 |     risk, (low, high) = await pred.predict_risk(question, context)
0013 | 
0014 | Design:
0015 | - Bundle must expose `.clf` with `predict_proba` and `.feature_names` (ordered list).
0016 | - Domain thresholds fetched from MemCube; fallback to config defaults.
0017 | - Asynchronous API; thread-safe reads; PII sanitization in featurizer.
0018 | """
0019 | 
0020 | from __future__ import annotations
0021 | 
0022 | import logging
0023 | import math
0024 | import re
0025 | from dataclasses import dataclass, field
0026 | from typing import Any, Dict, List, Optional, Tuple
0027 | 
0028 | import joblib
0029 | import numpy as np
0030 | 
0031 | from stephanie.memcube.memcube_client import MemCubeClient
0032 | from stephanie.tools.scorable_classifier import ScorableClassifier
0033 | 
0034 | log = logging.getLogger(__name__)
0035 | 
0036 | 
0037 | class DomainRequiredError(ValueError):
0038 |     """Raised when no domain/domain_tags provided and classification is disabled upstream."""
0039 | 
0040 |     pass
0041 | 
0042 | 
0043 | # --------- PII sanitization (emails/phones) ----------
0044 | EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
0045 | PHONE_RE = re.compile(r"\+?\d[\d\-\s]{7,}\d")
0046 | 
0047 | 
0048 | def sanitize_text(s: str) -> str:
0049 |     s = EMAIL_RE.sub("<EMAIL>", s or "")
0050 |     s = PHONE_RE.sub("<PHONE>", s)
0051 |     return s
0052 | 
0053 | 
0054 | # --------- Featurizer contract ----------
0055 | @dataclass
0056 | class RiskFeaturizer:
0057 |     """
0058 |     Produces a dense, ordered feature vector for the risk model.
0059 |     Replace `featurize` with your production features:
0060 |     - semantic embeddings
0061 |     - coverage/gap metrics from MemCube
0062 |     - prior Δ-energy EMA, etc.
0063 |     """
0064 | 
0065 |     feature_order: List[str] = field(
0066 |         default_factory=lambda: [
0067 |             "q_len",
0068 |             "ctx_len",
0069 |             "overlap_ratio",
0070 |             "ner_count",
0071 |             "num_tokens_est",
0072 |             "coverage_gap",
0073 |             "prior_max_energy_ema",
0074 |         ]
0075 |     )
0076 | 
0077 |     async def featurize(self, question: str, context: str) -> Dict[str, float]:
0078 |         q = sanitize_text(question or "")
0079 |         c = sanitize_text(context or "")
0080 |         q_tokens = q.split()
0081 |         c_tokens = c.split()
0082 |         inter = len(set(q_tokens) & set(c_tokens))
0083 |         denom = max(1, len(set(q_tokens)))
0084 |         overlap = inter / denom
0085 | 
0086 |         feats = {
0087 |             "q_len": float(len(q)),
0088 |             "ctx_len": float(len(c)),
0089 |             "overlap_ratio": float(overlap),
0090 |             "ner_count": float(
0091 |                 sum(w.istitle() for w in q_tokens)
0092 |             ),  # toy NER proxy
0093 |             "num_tokens_est": float(len(q_tokens) + len(c_tokens)),
0094 |             "coverage_gap": float(1.0 - overlap),
0095 |             "prior_max_energy_ema": 0.25,  # placeholder; wire your EMA here
0096 |         }
0097 |         return feats
0098 | 
0099 | 
0100 | # --------- Model bundle ----------
0101 | @dataclass
0102 | class RiskModelBundle:
0103 |     clf: Any
0104 |     feature_names: List[str]
0105 |     version: str = "risk-bundle.v1"
0106 | 
0107 |     @classmethod
0108 |     def load(cls, path: str) -> "RiskModelBundle":
0109 |         if joblib is None:
0110 |             raise RuntimeError("joblib not available to load model bundle")
0111 |         obj = joblib.load(path)
0112 |         # Support multiple serialization styles
0113 |         if isinstance(obj, dict) and "clf" in obj and "feature_names" in obj:
0114 |             clf = obj["clf"]
0115 |             names = obj["feature_names"]
0116 |             ver = obj.get("version", "risk-bundle.v1")
0117 |             return cls(clf=clf, feature_names=names, version=ver)
0118 |         # Direct estimator with attached names
0119 |         if hasattr(obj, "predict_proba") and hasattr(obj, "feature_names"):
0120 |             return cls(
0121 |                 clf=obj,
0122 |                 feature_names=list(obj.feature_names),
0123 |                 version=getattr(obj, "version", "risk-bundle.v1"),
0124 |             )
0125 |         raise ValueError(
0126 |             "Unsupported risk bundle format; expect dict with keys ['clf','feature_names']"
0127 |         )
0128 | 
0129 | 
0130 | # --------- Predictor ----------
0131 | class DomainCalibratedRiskPredictor:
0132 |     """
0133 |     Async hallucination-risk predictor with per-domain thresholds.
0134 | 
0135 |     Methods:
0136 |       - predict_risk(question, context) -> (risk, (low, high))
0137 | 
0138 |     Thresholds:
0139 |       - fetched from MemCube (kind="risk", fields: low_threshold, high_threshold)
0140 |       - fallback to defaults if not present
0141 |     """
0142 | 
0143 |     def __init__(
0144 |         self,
0145 |         bundle_path: Optional[str] = None,
0146 |         default_domains: Optional[List[str]] = None,
0147 |         default_thresholds: Tuple[float, float] = (0.2, 0.6),
0148 |         memcube: Optional[MemCubeClient] = None,
0149 |         featurizer: Optional[RiskFeaturizer] = None,
0150 |         domain_classifier: Optional[ScorableClassifier] = None,
0151 |     ):
0152 |         self.bundle: Optional[RiskModelBundle] = None
0153 |         if bundle_path:
0154 |             self.bundle = RiskModelBundle.load(bundle_path)
0155 | 
0156 |         self.default_domains = default_domains or ["programming", "ml", "nlp", "ai"]
0157 |         self.default_thresholds = (
0158 |             float(default_thresholds[0]),
0159 |             float(default_thresholds[1]),
0160 |         )
0161 |         self.memcube = memcube or MemCubeClient()
0162 |         self.featurizer = featurizer or RiskFeaturizer()
0163 | 
0164 |         # Local cache of thresholds to avoid frequent lookups
0165 |         self.domain_classifier = domain_classifier  # <-- NEW
0166 | 
0167 |         self._domain_thresholds: Dict[str, Tuple[float, float]] = {}
0168 | 
0169 |         # Validate feature contract if bundle is present
0170 |         if self.bundle is not None:
0171 |             missing = [
0172 |                 k
0173 |                 for k in self.featurizer.feature_order
0174 |                 if k not in self.bundle.feature_names
0175 |             ]
0176 |             if missing:
0177 |                 log.warning(
0178 |                     f"Risk bundle missing features used by featurizer: {missing}"
0179 |                 )
0180 | 
0181 |     # ------------- internal helpers -------------
0182 |     async def _get_domain_thresholds(self, domain: str) -> Tuple[float, float]:
0183 |         if domain in self._domain_thresholds:
0184 |             return self._domain_thresholds[domain]
0185 | 
0186 |         rec = await self.memcube.query_calibration(
0187 |             "risk",
0188 |             filters={"domain": domain},
0189 |             sort=[("created_at", "DESC")],
0190 |             limit=1,
0191 |         )
0192 |         if rec and isinstance(rec, dict):
0193 |             lo = float(rec.get("low_threshold", self.default_thresholds[0]))
0194 |             hi = float(rec.get("high_threshold", self.default_thresholds[1]))
0195 |         else:
0196 |             lo, hi = self.default_thresholds
0197 | 
0198 |         # basic sanity
0199 |         lo = max(0.0, min(lo, 0.95))
0200 |         hi = max(lo + 0.01, min(hi, 0.99))
0201 |         self._domain_thresholds[domain] = (lo, hi)
0202 |         return lo, hi
0203 | 
0204 |     def _vectorize(self, feats: Dict[str, float]) -> np.ndarray:
0205 |         """
0206 |         Order features per bundle.feature_names, fill missing with 0.
0207 |         If bundle absent, use featurizer.feature_order.
0208 |         """
0209 |         names = (
0210 |             self.bundle.feature_names
0211 |             if self.bundle
0212 |             else self.featurizer.feature_order
0213 |         )
0214 |         x = np.array(
0215 |             [[float(feats.get(k, 0.0)) for k in names]], dtype=np.float32
0216 |         )
0217 |         return x, names
0218 | 
0219 |     def _predict_proba(self, x: np.ndarray) -> float:
0220 |         # If model bundle available → calibrated predict_proba
0221 |         if self.bundle is not None:
0222 |             proba = self.bundle.clf.predict_proba(x)[:, 1]
0223 |             return float(np.clip(proba[0], 0.0, 1.0))
0224 |         # Fallback: tiny heuristic on a couple features
0225 |         q_len_idx = (
0226 |             self.featurizer.feature_order.index("q_len")
0227 |             if "q_len" in self.featurizer.feature_order
0228 |             else 0
0229 |         )
0230 |         gap_idx = (
0231 |             self.featurizer.feature_order.index("coverage_gap")
0232 |             if "coverage_gap" in self.featurizer.feature_order
0233 |             else 0
0234 |         )
0235 |         q_len = x[0, q_len_idx]
0236 |         gap = x[0, gap_idx]
0237 |         # Risk grows with coverage gap and short questions
0238 |         risk = 0.2 + 0.6 * (gap) + 0.2 * (1.0 - math.tanh(q_len / 512.0))
0239 |         return float(max(0.0, min(1.0, risk)))
0240 | 
0241 |     # ------------- public API -------------
0242 |     async def predict_risk(
0243 |         self,
0244 |         question: str,
0245 |         context: str,
0246 |         *,
0247 |         domain: Optional[str] = None,
0248 |         domain_tags: Optional[List[str]] = None,
0249 |     ) -> Tuple[float, Tuple[float, float]]:
0250 |         """
0251 |         Returns:
0252 |             risk: float in [0,1]
0253 |             thresholds: (low, high): domain-calibrated gates
0254 |         """
0255 |         # STRICT: never guess; a domain (or tags) must be provided by caller.
0256 |         dom = (domain or "").strip().lower() or self._choose_primary_domain(
0257 |             domain_tags
0258 |         )
0259 |         if not dom:
0260 |             raise DomainRequiredError(
0261 |                 "Domain is required. Provide `domain` or `domain_tags` from the ScorableDomainAgent."
0262 |             )
0263 |         # 0) Domain
0264 |         domain = await self._guess_domain(question or "")
0265 |         low, high = await self._get_domain_thresholds(domain)
0266 | 
0267 |         # 1) Features
0268 |         feats = await self.featurizer.featurize(question or "", context or "")
0269 |         x, names = self._vectorize(feats)
0270 | 
0271 |         # 2) Score
0272 |         risk = self._predict_proba(x)
0273 | 
0274 |         return risk, (low, high)
0275 | 
0276 |     # ------------- optional: explanation -------------
0277 |     async def explain(self, question: str, context: str) -> Dict[str, Any]:
0278 |         """
0279 |         Returns SHAP-like explanation if shap is installed and bundle is a tree-based model.
0280 |         """
0281 |         try:
0282 |             import shap  # optional
0283 |         except Exception:
0284 |             return {"ok": False, "reason": "shap not installed"}
0285 | 
0286 |         if self.bundle is None:
0287 |             return {"ok": False, "reason": "no bundle"}
0288 |         if not hasattr(self.bundle.clf, "predict_proba"):
0289 |             return {"ok": False, "reason": "clf lacks predict_proba"}
0290 | 
0291 |         feats = await self.featurizer.featurize(question or "", context or "")
0292 |         x, names = self._vectorize(feats)
0293 | 
0294 |         # Try to unwrap isotonic calibration wrappers (commonly clf.base_estimator)
0295 |         base = getattr(self.bundle.clf, "base_estimator", self.bundle.clf)
0296 |         try:
0297 |             explainer = shap.TreeExplainer(base)
0298 |             sv = explainer.shap_values(x)
0299 |             sv0 = sv[0] if isinstance(sv, list) else sv
0300 |             values = {names[i]: float(sv0[0, i]) for i in range(len(names))}
0301 |             return {
0302 |                 "ok": True,
0303 |                 "expected_value": float(
0304 |                     explainer.expected_value[0]
0305 |                     if isinstance(
0306 |                         explainer.expected_value, (list, tuple, np.ndarray)
0307 |                     )
0308 |                     else explainer.expected_value
0309 |                 ),
0310 |                 "shap_values": values,
0311 |             }
0312 |         except Exception as e:
0313 |             return {"ok": False, "reason": f"shap failed: {e}"}
0314 | 
0315 |     async def _guess_domain(self, question: str) -> str:
0316 |         """
0317 |         Uses ScorableClassifier if available, otherwise falls back to MemCube.
0318 |         """
0319 |         if self.domain_classifier:
0320 |             try:
0321 |                 results = self.domain_classifier.classify(
0322 |                     question,
0323 |                     top_k=1,
0324 |                     min_value=0.4,  # or adjust threshold
0325 |                 )
0326 |                 if results:
0327 |                     return results[0][0]
0328 |             except Exception as e:
0329 |                 log.warning(f"Domain classification failed: {e}")
0330 | 
0331 |         # Fallback
0332 |         return await self.memcube.guess_domain(question or "")
```


---

## F0012 — `sicql.py`

```text
FILE_ID: F0012
PATH: sicql.py
LANGUAGE: python
LINES: 302
BYTES_UTF8: 8507
SHA256: 3642d44f4d23a01ba0af84011654b4ae0cf6555d77dce5ae6a1b1f46d079e51b
```

```python
0001 | # stephanie/model/sicql.py
0002 | from __future__ import annotations
0003 | 
0004 | import hashlib
0005 | import logging
0006 | from typing import Any, Dict, Optional, Tuple
0007 | 
0008 | import torch
0009 | import torch.nn as nn
0010 | from torch.nn import Linear, ReLU
0011 | 
0012 | from stephanie.model.text_encoder import TextEncoder
0013 | 
0014 | log = logging.getLogger(__name__)
0015 | 
0016 | 
0017 | class PolicyHead(nn.Module):
0018 |     def __init__(self, zsa_dim, hdim, num_actions=3):
0019 |         super().__init__()
0020 |         _log_expected_shapes("PolicyHead", zsa_dim, hdim, num_actions)
0021 |         self.linear = nn.Sequential(
0022 |             Linear(zsa_dim, hdim), ReLU(), Linear(hdim, num_actions)
0023 |         )
0024 |         self._init_weights()
0025 | 
0026 |     def _init_weights(self):
0027 |         """Initialize weights with Xavier"""
0028 |         for m in self.modules():
0029 |             if isinstance(m, nn.Linear):
0030 |                 torch.nn.init.xavier_normal_(m.weight)
0031 |                 torch.nn.init.zeros_(m.bias)
0032 | 
0033 |     def forward(self, zsa):
0034 |         return self.linear(zsa)
0035 | 
0036 |     def get_policy_weights(self):
0037 |         """
0038 |         Get the averaged weights of the final linear layer for policy logits.
0039 |         """
0040 |         final_linear_layer = self.linear[-1]
0041 |         return final_linear_layer.weight.data.mean(dim=0)
0042 | 
0043 | 
0044 | class QHead(nn.Module):
0045 |     def __init__(self, zsa_dim, hdim):
0046 |         """
0047 |         Q-value estimator: Q(s,a) = E[reward | state, action]
0048 | 
0049 |         Args:
0050 |             zsa_dim: Dimension of encoded state-action vector
0051 |             hdim: Hidden layer dimension
0052 |         """
0053 |         super().__init__()
0054 |         _log_expected_shapes("QHead", zsa_dim, hdim)
0055 |         self.model = nn.Sequential(
0056 |             Linear(zsa_dim, hdim), ReLU(), Linear(hdim, 1)
0057 |         )
0058 |         self._init_weights()
0059 | 
0060 |     def _init_weights(self):
0061 |         """Initialize weights with Xavier"""
0062 |         for m in self.modules():
0063 |             if isinstance(m, nn.Linear):
0064 |                 torch.nn.init.xavier_normal_(m.weight)
0065 |                 torch.nn.init.zeros_(m.bias)
0066 | 
0067 |     def forward(self, zsa):
0068 |         """
0069 |         Predict Q-value for (state, action) pair
0070 |         Args:
0071 |             zsa: Encoded state-action vector
0072 |         Returns:
0073 |             Q-value (scalar)
0074 |         """
0075 |         return self.model(zsa).squeeze()
0076 | 
0077 | 
0078 | class VHead(nn.Module):
0079 |     def __init__(self, zsa_dim, hdim):
0080 |         """
0081 |         State value estimator using expectile regression
0082 | 
0083 |         Args:
0084 |             zsa_dim: Dimension of encoded state-action vector
0085 |             hdim: Hidden layer dimension
0086 |         """
0087 |         super().__init__()
0088 |         _log_expected_shapes("VHead", zsa_dim, hdim)
0089 |         self.net = nn.Sequential(
0090 |             Linear(zsa_dim, hdim), ReLU(), Linear(hdim, 1)
0091 |         )
0092 |         self._init_weights()
0093 | 
0094 |     def _init_weights(self):
0095 |         """Initialize weights with Xavier"""
0096 |         for m in self.modules():
0097 |             if isinstance(m, nn.Linear):
0098 |                 torch.nn.init.xavier_normal_(m.weight)
0099 |                 torch.nn.init.zeros_(m.bias)
0100 | 
0101 |     def forward(self, zsa):
0102 |         """
0103 |         Predict state value V(s)
0104 |         Args:
0105 |             zsa: Encoded state-action vector
0106 |         Returns:
0107 |             State value (scalar)
0108 |         """
0109 |         return self.net(zsa).squeeze()
0110 | 
0111 | 
0112 | class InContextQModel(nn.Module):
0113 |     def __init__(
0114 |         self,
0115 |         encoder: TextEncoder,
0116 |         q_head: QHead,
0117 |         v_head: VHead,
0118 |         pi_head: PolicyHead,
0119 |         embedding_store,
0120 |         device="cpu",
0121 |     ):
0122 |         super().__init__()
0123 |         self.encoder = encoder.to(device)
0124 |         self.q_head = q_head.to(device)
0125 |         self.v_head = v_head.to(device)
0126 |         self.pi_head = pi_head.to(device)
0127 |         self.device = device
0128 |         self.embedding_store = embedding_store
0129 | 
0130 |     def forward(self, context_emb, doc_emb):
0131 |         """
0132 |         Forward pass through all heads
0133 | 
0134 |         Args:
0135 |             context_emb: Goal/prompt embedding
0136 |             doc_emb: Document/output embedding
0137 |         Returns:
0138 |             Dict containing Q-value, state value, and policy logits
0139 |         """
0140 |         # Ensure device alignment
0141 |         context_emb = context_emb.to(self.device)
0142 |         doc_emb = doc_emb.to(self.device)
0143 | 
0144 |         # Combine embeddings
0145 |         zsa = self.encoder(context_emb, doc_emb)
0146 | 
0147 |         # Forward through heads
0148 |         q_value = self.q_head(zsa)
0149 |         state_value = self.v_head(zsa)
0150 |         action_logits = self.pi_head(zsa)
0151 | 
0152 |         # Calculate advantage
0153 |         advantage = (q_value - state_value).detach()
0154 | 
0155 |         return {
0156 |             "q_value": q_value,
0157 |             "state_value": state_value,
0158 |             "action_logits": action_logits,
0159 |             "advantage": advantage,
0160 |         }
0161 | 
0162 | 
0163 | # -----------------------------
0164 | # Logging utilities (warn-once)
0165 | # -----------------------------
0166 | _WARNED: set[str] = set()
0167 | 
0168 | 
0169 | def _warn_once(key: str, msg: str, *args) -> None:
0170 |     if key in _WARNED:
0171 |         return
0172 |     _WARNED.add(key)
0173 |     log.warning(msg, *args)
0174 | 
0175 | 
0176 | def _tensor_fingerprint(t: torch.Tensor, n: int = 2048) -> str:
0177 |     """
0178 |     Stable-ish fingerprint of a tensor's contents without dumping values.
0179 |     Uses first N bytes of raw storage on CPU.
0180 |     """
0181 |     with torch.no_grad():
0182 |         x = t.detach().to("cpu").contiguous().view(-1)
0183 |         if x.numel() == 0:
0184 |             return "empty"
0185 |         # sample up to n elements, convert to bytes
0186 |         x = x[: min(x.numel(), n)].to(torch.float32)
0187 |         b = x.numpy().tobytes()
0188 |         return hashlib.sha256(b).hexdigest()[:16]
0189 | 
0190 | 
0191 | def _param_summary(m: nn.Module) -> Dict[str, Any]:
0192 |     """
0193 |     Small param stats to detect 'all zeros', NaNs, or weird scales.
0194 |     """
0195 |     with torch.no_grad():
0196 |         ps = [p.detach() for p in m.parameters() if p is not None]
0197 |         if not ps:
0198 |             return {"n_params": 0}
0199 |         flat = torch.cat([p.float().flatten().cpu() for p in ps])
0200 |         return {
0201 |             "n_params": int(flat.numel()),
0202 |             "mean": float(flat.mean().item()),
0203 |             "std": float(flat.std(unbiased=False).item()),
0204 |             "min": float(flat.min().item()),
0205 |             "max": float(flat.max().item()),
0206 |             "fp": _tensor_fingerprint(flat),
0207 |         }
0208 | 
0209 | 
0210 | def _log_expected_shapes(
0211 |     module_name: str,
0212 |     zsa_dim: int,
0213 |     hdim: int,
0214 |     num_actions: Optional[int] = None,
0215 | ) -> None:
0216 |     if num_actions is None:
0217 |         log.info(
0218 |             "%s expected: Linear(%d -> %d) -> Linear(%d -> 1)",
0219 |             module_name,
0220 |             zsa_dim,
0221 |             hdim,
0222 |             hdim,
0223 |         )
0224 |     else:
0225 |         log.info(
0226 |             "%s expected: Linear(%d -> %d) -> Linear(%d -> %d)",
0227 |             module_name,
0228 |             zsa_dim,
0229 |             hdim,
0230 |             hdim,
0231 |             num_actions,
0232 |         )
0233 | 
0234 | 
0235 | def _peek_state_dict_shapes(
0236 |     state_dict: Dict[str, torch.Tensor], keys: Tuple[str, ...]
0237 | ) -> Dict[str, Tuple[int, ...]]:
0238 |     out = {}
0239 |     for k in keys:
0240 |         v = state_dict.get(k)
0241 |         if isinstance(v, torch.Tensor):
0242 |             out[k] = tuple(v.shape)
0243 |     return out
0244 | 
0245 | 
0246 | def log_load_state_dict(
0247 |     module: nn.Module,
0248 |     state_dict: Dict[str, Any],
0249 |     *,
0250 |     module_name: str,
0251 |     strict: bool = False,
0252 | ) -> bool:
0253 |     """
0254 |     Load a state_dict with useful shape logging.
0255 |     Returns True if load succeeded, False if RuntimeError.
0256 |     """
0257 |     # common first-layer keys for these heads
0258 |     peek_keys = (
0259 |         "model.0.weight",
0260 |         "model.0.bias",
0261 |         "net.0.weight",
0262 |         "net.0.bias",
0263 |         "linear.0.weight",
0264 |         "linear.0.bias",
0265 |     )
0266 |     peek = _peek_state_dict_shapes(state_dict, peek_keys)
0267 |     if peek:
0268 |         log.info("%s ckpt shapes: %s", module_name, peek)
0269 | 
0270 |     try:
0271 |         missing, unexpected = module.load_state_dict(state_dict, strict=strict)
0272 |         # note: shape mismatches raise before this
0273 |         if missing or unexpected:
0274 |             _warn_once(
0275 |                 f"load_keys::{module_name}",
0276 |                 "%s load_state_dict strict=%s missing=%s unexpected=%s",
0277 |                 module_name,
0278 |                 strict,
0279 |                 list(missing),
0280 |                 list(unexpected),
0281 |             )
0282 |         summ = _param_summary(module)
0283 |         log.info(
0284 |             "%s loaded params: n=%s mean=%.6g std=%.6g min=%.6g max=%.6g fp=%s",
0285 |             module_name,
0286 |             summ.get("n_params"),
0287 |             summ.get("mean"),
0288 |             summ.get("std"),
0289 |             summ.get("min"),
0290 |             summ.get("max"),
0291 |             summ.get("fp"),
0292 |         )
0293 |         return True
0294 |     except RuntimeError as e:
0295 |         _warn_once(
0296 |             f"load_fail::{module_name}",
0297 |             "%s FAILED to load state_dict (strict=%s): %s",
0298 |             module_name,
0299 |             strict,
0300 |             str(e).splitlines()[0],
0301 |         )
0302 |         return False
```


---

## F0013 — `text_encoder.py`

```text
FILE_ID: F0013
PATH: text_encoder.py
LANGUAGE: python
LINES: 19
BYTES_UTF8: 559
SHA256: e19d76cd7cac7fa0fef45108a359a4b37080b80e06e5198edc0b5d9d095049f0
```

```python
0001 | # stephanie/model/text_encoder.py
0002 | from __future__ import annotations
0003 | 
0004 | import torch
0005 | import torch.nn as nn
0006 | 
0007 | 
0008 | class TextEncoder(nn.Module):
0009 |     def __init__(self, dim=4096, hdim=4096):
0010 |         super().__init__()
0011 |         self.encoder = nn.Sequential(
0012 |             nn.Linear(dim * 2, hdim),  # Concatenate context + document
0013 |             nn.ReLU(),
0014 |             nn.Linear(hdim, dim),      # Keep the output same size
0015 |         )
0016 | 
0017 |     def forward(self, context_emb, doc_emb):
0018 |         concat = torch.cat([context_emb, doc_emb], dim=1)
0019 |         return self.encoder(concat)
```


---

## F0014 — `tiny.py`

```text
FILE_ID: F0014
PATH: tiny.py
LANGUAGE: python
LINES: 507
BYTES_UTF8: 20202
SHA256: 03b7d4e3ed095ce18053772122e38ceb95d39820294b1870687c852719831c9c
```

```python
0001 | 
0002 | # stephanie/model/tiny_recursion.py
0003 | """
0004 | Tiny Recursion Model (Tiny+) - Parameter-Efficient Recursive Neural Architecture
0005 | 
0006 | This module implements a compact, recursive neural network for multi-task evaluation
0007 | of AI model responses. The architecture combines recursive state updates with
0008 | multi-head output predictions, enabling efficient quality assessment across
0009 | multiple dimensions from embedding inputs.
0010 | 
0011 | Key Innovations:
0012 | - Recursive latent state updates with halting mechanisms
0013 | - Sparse Autoencoder (SAE) bottleneck for interpretable concepts
0014 | - Multi-head prediction for comprehensive quality assessment
0015 | - Heteroscedastic uncertainty estimation
0016 | - In-graph consistency regularization
0017 | 
0018 | Architecture Overview:
0019 | 1. Recursive fusion of goal (x), response (y), and latent (z) states
0020 | 2. Core processing blocks (attention or MLP-based)
0021 | 3. SAE bottleneck for sparse concept representation
0022 | 4. Multi-head prediction for scores, uncertainty, and auxiliary tasks
0023 | 
0024 | """
0025 | 
0026 | from __future__ import annotations
0027 | 
0028 | from typing import Any, Dict, Optional, Tuple
0029 | 
0030 | import torch
0031 | import torch.nn as nn
0032 | import torch.nn.functional as F
0033 | 
0034 | from stephanie.scoring.analysis.trace_tap import TraceTap
0035 | 
0036 | # ---------------------------
0037 | # Core Building Blocks
0038 | # ---------------------------
0039 | 
0040 | class TinyBlock(nn.Module):
0041 |     """
0042 |     Basic residual block: LayerNorm → MLP → residual connection.
0043 | 
0044 |     Supports both 2D [batch, features] and 3D [batch, sequence, features] inputs.
0045 |     Uses GELU activation and dropout for regularization.
0046 |     """
0047 |     def __init__(self, d_model: int, dropout: float = 0.1):
0048 |         super().__init__()
0049 |         self.ln = nn.LayerNorm(d_model)
0050 |         self.mlp = nn.Sequential(
0051 |             nn.Linear(d_model, d_model * 4),  # Expansion factor 4
0052 |             nn.GELU(),
0053 |             nn.Dropout(dropout),
0054 |             nn.Linear(d_model * 4, d_model),  # Projection back
0055 |             nn.Dropout(dropout),
0056 |         )
0057 | 
0058 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
0059 |         """Apply residual block: x + MLP(LayerNorm(x))"""
0060 |         return x + self.mlp(self.ln(x))
0061 | 
0062 | 
0063 | class TinyBlockAttn(nn.Module):
0064 |     """
0065 |     Attention-enhanced residual block with Multi-Head Self-Attention.
0066 | 
0067 |     Architecture: LN → MHA → residual → TinyBlock → residual
0068 |     Automatically handles 2D/3D inputs and returns same dimensionality.
0069 |     """
0070 |     def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
0071 |         super().__init__()
0072 |         self.ln_attn = nn.LayerNorm(d_model)
0073 |         self.attn = nn.MultiheadAttention(
0074 |             embed_dim=d_model,
0075 |             num_heads=n_heads,
0076 |             dropout=dropout,
0077 |             batch_first=True  # [batch, seq, features]
0078 |         )
0079 |         self.drop = nn.Dropout(dropout)
0080 |         self.ff = TinyBlock(d_model, dropout=dropout)
0081 | 
0082 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
0083 |         """
0084 |         Forward pass with automatic shape handling.
0085 | 
0086 |         Args:
0087 |             x: Input tensor of shape [B, D] or [B, L, D]
0088 | 
0089 |         Returns:
0090 |             Output tensor with same shape as input
0091 |         """
0092 |         squeeze_back = False
0093 |         if x.dim() == 2:
0094 |             x = x.unsqueeze(1)  # [B, D] → [B, 1, D]
0095 |             squeeze_back = True
0096 | 
0097 |         q = k = v = self.ln_attn(x)
0098 |         h, _ = self.attn(q, k, v, need_weights=False)
0099 |         x = x + self.drop(h)  # Residual connection
0100 |         x = self.ff(x)        # Feed-forward with residual
0101 | 
0102 |         if squeeze_back:
0103 |             x = x.squeeze(1)  # [B, 1, D] → [B, D]
0104 |         return x
0105 | 
0106 | 
0107 | # ---------------------------
0108 | # Tiny Recursion Model (Tiny+)
0109 | # ---------------------------
0110 | 
0111 | class TinyModel(nn.Module):
0112 |     """
0113 |     Parameter-efficient recursive model for multi-task evaluation.
0114 | 
0115 |     Recursively updates latent state z using goal (x) and response (y) embeddings
0116 |     over multiple steps. Features comprehensive multi-head prediction and
0117 |     sparse autoencoder bottleneck for interpretable representations.
0118 | 
0119 |     Core Components:
0120 |     - Recursive state fusion: [x, y, z] → z'
0121 |     - Core processing stack: Attention or MLP blocks
0122 |     - SAE bottleneck: Sparse concept encoding
0123 |     - Multi-head prediction: 12 specialized output heads
0124 | 
0125 |     Inputs:
0126 |         x: Goal/condition embedding [B, D]
0127 |         y: Response embedding [B, D]
0128 |         z: Initial latent state [B, D] (typically zeros)
0129 | 
0130 |     Outputs:
0131 |         logits: Classification logits [B, vocab_size] (legacy compatibility)
0132 |         halt_logits: Halting signal logits [B]
0133 |         z_final: Final latent state after recursion [B, D]
0134 |         aux: Dictionary of auxiliary predictions and metrics
0135 |     """
0136 | 
0137 |     def __init__(
0138 |         self,
0139 |         d_model: int = 256,
0140 |         n_layers: int = 2,
0141 |         n_recursions: int = 6,
0142 |         vocab_size: int = 1024,
0143 |         use_attention: bool = False,
0144 |         dropout: float = 0.1,
0145 |         attn_heads: int = 4,
0146 |         step_scale: float = 0.1,           # Residual scaling for state updates
0147 |         consistency_mask_p: float = 0.10,  # Mask probability for consistency regularization
0148 |         len_norm_L: float = 512.0,         # Length normalization constant
0149 |         enable_agree_head: bool = True,    # Enable agreement prediction head
0150 |         enable_causal_sens_head: bool = True,  # Enable sensitivity prediction head
0151 |     ):
0152 |         super().__init__()
0153 | 
0154 |         # Model configuration
0155 |         self.d_model = d_model
0156 |         self.n_layers = n_layers
0157 |         self.n_recursions = n_recursions
0158 |         self.vocab_size = vocab_size
0159 |         self.use_attention = use_attention
0160 |         self.step_scale = step_scale
0161 |         self.consistency_mask_p = consistency_mask_p
0162 |         self.len_norm_L = float(len_norm_L)
0163 |         self.enable_agree_head = enable_agree_head
0164 |         self.enable_causal_sens_head = enable_causal_sens_head
0165 | 
0166 |         # Core processing stack
0167 |         if use_attention:
0168 |             blocks = [TinyBlockAttn(d_model, n_heads=attn_heads, dropout=dropout)
0169 |                       for _ in range(n_layers)]
0170 |         else:
0171 |             blocks = [TinyBlock(d_model, dropout=dropout) for _ in range(n_layers)]
0172 |         self.core = nn.Sequential(*blocks)
0173 | 
0174 |         # State fusion: combine goal, response, and latent states
0175 |         self.z_proj = nn.Linear(d_model * 3, d_model)  # [x, y, z] → z'
0176 |         self.final_ln = nn.LayerNorm(d_model)
0177 | 
0178 |         # Core prediction heads
0179 |         self.halt_head = nn.Linear(d_model, 1)            # Halting signal logits
0180 |         self.classifier = nn.Linear(d_model, vocab_size)  # Legacy classification
0181 | 
0182 |         # Extended prediction heads
0183 |         self.score_head = nn.Linear(d_model, 1)        # Quality score ∈ [0,1]
0184 |         self.logvar_head = nn.Linear(d_model, 1)       # Aleatoric uncertainty (log-variance)
0185 |         self.aux3_head = nn.Linear(d_model, 3)         # 3-way classification
0186 |         self.disagree_head = nn.Linear(d_model, 1)     # Disagreement prediction
0187 |         self.recon_head = nn.Linear(d_model, d_model)  # Embedding reconstruction
0188 |         self.consistency_head = nn.Linear(d_model, 1)  # Robustness prediction
0189 |         self.ood_head = nn.Linear(d_model, 1)          # OOD detection
0190 |         self.temp_head = nn.Linear(d_model, 1)         # Temperature calibration
0191 | 
0192 |         # Bridge heads
0193 |         self.agree_head = nn.Linear(d_model, 1)        # Cross-model agreement
0194 |         self.causal_sens_head = nn.Linear(d_model, 1)  # Perturbation sensitivity
0195 | 
0196 |         # Sparse Autoencoder (SAE) bottleneck
0197 |         self.sae_enc = nn.Sequential(
0198 |             nn.Linear(d_model, d_model // 2),  # Compression
0199 |             nn.ReLU(),
0200 |             nn.LayerNorm(d_model // 2),
0201 |         )
0202 |         self.sae_dec = nn.Linear(d_model // 2, d_model)  # Reconstruction
0203 |         self.sae_alpha = 0.05  # SAE reconstruction loss weight
0204 | 
0205 |         # Regularization
0206 |         self.head_drop = nn.Dropout(dropout)
0207 | 
0208 |     @staticmethod
0209 |     def _cos01(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
0210 |         """
0211 |         Compute cosine similarity mapped from [-1, 1] to [0, 1].
0212 | 
0213 |         Args:
0214 |             a, b: Input tensors to compare
0215 |             dim: Dimension for cosine computation
0216 |             eps: Numerical stability term
0217 | 
0218 |         Returns:
0219 |             Cosine similarity in range [0, 1] where 1 = identical
0220 |         """
0221 |         sim = F.cosine_similarity(a, b, dim=dim, eps=eps)
0222 |         return (sim + 1.0) * 0.5
0223 | 
0224 |     def _recur(
0225 |         self,
0226 |         x: torch.Tensor,
0227 |         y: torch.Tensor,
0228 |         z: torch.Tensor,
0229 |         *,
0230 |         n_steps: Optional[int] = None,
0231 |         tap: Optional[TraceTap] = None,
0232 |     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
0233 |         """
0234 |         Execute recursive state updates over n_recursions steps.
0235 | 
0236 |         Process:
0237 |           1. Fuse [x, y, z] → z_next via projection and activation
0238 |           2. Process through core network stack
0239 |           3. Update halting signals
0240 |           4. Apply residual state update: z = z + step_scale * z_next
0241 |           5. Apply SAE bottleneck to final state
0242 | 
0243 |         Args:
0244 |             x: Goal embedding [B, D]
0245 |             y: Response embedding [B, D]
0246 |             z: Initial latent state [B, D]
0247 | 
0248 |         Returns:
0249 |             z_final: Final latent state after recursion [B, D]
0250 |             z_head: SAE-processed state for prediction heads [B, D]
0251 |             halt_logits: Maximum halting logits across steps [B, 1]
0252 |             tau: Temperature parameter for score calibration [B, 1]
0253 |             c: Sparse concept codes from SAE bottleneck [B, D//2]
0254 |         """
0255 |         B = x.size(0)
0256 |         device = x.device
0257 | 
0258 |         # Initialize halting signals to very negative values
0259 |         halt_logits = torch.full((B, 1), -1e9, device=device)
0260 |         z_cur = z  # Current latent state
0261 | 
0262 |         # Recursive state updates
0263 |         steps = self.n_recursions if n_steps is None else int(n_steps)
0264 |         steps = max(1, min(steps, self.n_recursions))
0265 |         for step_idx in range(steps):
0266 |             fused = torch.cat([x, y, z_cur], dim=-1)   # [B, 3 * D]
0267 |             z_next = torch.tanh(self.z_proj(fused))    # [B, D] with saturation
0268 |             z_next = self.core(z_next)                 # [B, D] core processing
0269 | 
0270 |             # Update halting signal (track maximum across steps)
0271 |             step_halt = self.halt_head(self.final_ln(z_next))  # [B, 1]
0272 |             halt_logits = torch.maximum(halt_logits, step_halt)
0273 | 
0274 |             # Residual state update with step scaling
0275 |             z_cur = z_cur + self.step_scale * z_next
0276 | 
0277 |             if tap is not None:
0278 |                 tap.add("tiny/z_cur", z_cur)
0279 |                 tap.add("tiny/z_next", z_next)
0280 |                 tap.add("tiny/halt_logits", halt_logits)
0281 | 
0282 |         # Final normalization
0283 |         z_final = self.final_ln(z_cur)  # [B, D]
0284 | 
0285 |         # Sparse Autoencoder bottleneck
0286 |         c = self.sae_enc(z_final)                  # [B, D//2] concept codes
0287 |         z_head = z_final + self.sae_dec(c)         # [B, D] with SAE reconstruction
0288 |         z_head = self.head_drop(z_head)            # Regularization
0289 | 
0290 |         # Temperature calibration parameter (τ ∈ (0.5, ∞))
0291 |         tau_raw = self.temp_head(z_head)
0292 |         tau = 0.5 + 0.5 * F.softplus(tau_raw)  # Lower bound at 0.5
0293 | 
0294 |         return z_final, z_head, halt_logits, tau, c
0295 | 
0296 |     def forward(
0297 |         self,
0298 |         x: torch.Tensor,                    # Goal embedding [B, D]
0299 |         y: torch.Tensor,                    # Response embedding [B, D]
0300 |         z: torch.Tensor,                    # Initial latent state [B, D]
0301 |         *,
0302 |         seq_len: Optional[torch.Tensor] = None,  # Response length [B] (optional)
0303 |         return_aux: bool = True,                 # Whether to return auxiliary outputs
0304 |         with_consistency_target: bool = True,    # Compute consistency regularization
0305 |         n_steps: Optional[int] = None,
0306 |         tap: Optional["TraceTap"] = None,
0307 |     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
0308 |         """
0309 |         Complete forward pass with recursive processing and multi-head prediction.
0310 |         """
0311 |         # Main recursive processing
0312 |         z = z.clone()  # Ensure we don't modify input
0313 |         z_final, z_head, halt_logits, tau, c = self._recur(x, y, z)
0314 |         z_final, z_head, halt_logits, tau, c = self._recur(x, y, z, n_steps=n_steps, tap=tap)
0315 | 
0316 |         if tap is not None:
0317 |             tap.add("tiny/z_final", z_final)
0318 |             tap.add("tiny/z_head", z_head)
0319 | 
0320 |         # Core prediction heads
0321 |         logits = self.classifier(z_head)                    # [B, vocab_size]
0322 |         score_logit = self.score_head(z_head)               # [B, 1]
0323 |         log_var = self.logvar_head(z_head)                  # [B, 1] uncertainty
0324 | 
0325 |         # ----- NUMERICAL SAFETY -----
0326 |         LOGVAR_MIN, LOGVAR_MAX = -5.0, 5.0
0327 |         log_var = log_var.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX)
0328 | 
0329 |         # Use tau for calibration; keep a stable proxy for telemetry
0330 |         # NOTE: move temp01 to sigmoid(tau_raw) for cross-model alignment
0331 |         tau_raw = self.temp_head(z_head) 
0332 |         tau = 0.5 + 0.5 * F.softplus(tau_raw)
0333 |         tau_safe = torch.clamp(tau, min=1e-2)
0334 |         s = torch.sigmoid(score_logit / tau_safe)
0335 | 
0336 |         # ----- Core auxiliaries
0337 |         aux3_logits = self.aux3_head(z_head)
0338 |         aux3_probs  = F.softmax(aux3_logits, dim=-1)
0339 |         disagree_logit = self.disagree_head(z_head)
0340 |         y_recon     = self.recon_head(z_head)
0341 |         ood_logit   = self.ood_head(z_head)
0342 | 
0343 |         # Optional bridge heads
0344 |         agree01 = torch.sigmoid(self.agree_head(z_head)) if self.enable_agree_head else None
0345 |         sens01  = torch.sigmoid(self.causal_sens_head(z_head)) if self.enable_causal_sens_head else None
0346 | 
0347 |         # Consistency target 
0348 |         mask = (torch.rand_like(z_head) < self.consistency_mask_p).float()
0349 |         z_masked = z_head * (1.0 - mask)
0350 |         cos_consistency = self._cos01(z_head, z_masked).unsqueeze(-1)
0351 |         consistency_logit = self.consistency_head(z_head)
0352 | 
0353 |         # Finite-difference sensitivity
0354 |         eps = 1e-3
0355 |         y_eps = y + eps * F.normalize(torch.randn_like(y), dim=-1)
0356 |         with torch.no_grad():
0357 |             _, z_head_eps, _, tau_eps, _ = self._recur(x, y_eps, z)
0358 |         tau_eps_safe = torch.clamp(tau_eps, min=1e-2)
0359 |         score_eps    = torch.sigmoid(self.score_head(z_head_eps) / tau_eps_safe)
0360 |         jac_fd       = ((score_eps - s).abs() / eps).clamp(0, 10.0) / 10.0
0361 | 
0362 |         # Length effect
0363 |         if seq_len is not None:
0364 |             len_effect = torch.tanh((seq_len.float() / self.len_norm_L)).unsqueeze(-1)
0365 |         else:
0366 |             len_effect = torch.zeros_like(s)
0367 |         length_norm01 = (len_effect + 1.0) * 0.5
0368 | 
0369 |         # ----- Aligned telemetry keys -----
0370 |         certainty01   = torch.sigmoid(-log_var)
0371 |         uncertainty01 = 1.0 - certainty01
0372 |         temp01        = torch.sigmoid(tau_raw)  # aligned proxy in [0,1]
0373 |         ood_hat01     = torch.sigmoid(ood_logit)
0374 |         halt_prob     = torch.sigmoid(halt_logits).unsqueeze(-1) if halt_logits.dim()==1 else torch.sigmoid(halt_logits)
0375 | 
0376 |         # Device-safe normalized entropy (in [0,1])
0377 |         logK = torch.log(torch.tensor(3.0, device=z_head.device, dtype=z_head.dtype))
0378 |         entropy_aux = (-(aux3_probs * F.log_softmax(aux3_logits, -1)).sum(-1) / logK).unsqueeze(-1)
0379 | 
0380 |         aux: Dict[str, Any] = {
0381 |             # raw heads you need for training
0382 |             "score_logit": score_logit,
0383 |             "log_var": log_var,
0384 |             "aux3_logits": aux3_logits,
0385 |             "disagree_logit": disagree_logit,
0386 |             "y_recon": y_recon,
0387 |             "consistency_logit": consistency_logit,
0388 |             "consistency_target": cos_consistency.detach(),
0389 | 
0390 |             # aligned derived telemetry (all ∈ [0,1])
0391 |             "score": s,
0392 |             "certainty01": certainty01,
0393 |             "uncertainty01": uncertainty01,     # <— NEW (correct)
0394 |             "uncertainty": uncertainty01,       # <— OPTIONAL alias for back-compat
0395 |             "aux3_probs": aux3_probs,
0396 |             "entropy_aux": entropy_aux,
0397 |             "disagree_hat": torch.sigmoid(disagree_logit),
0398 |             "recon_sim": self._cos01(y_recon, y).unsqueeze(-1),
0399 |             "consistency_hat": torch.sigmoid(consistency_logit),
0400 |             "concept_sparsity": (c > 0).float().mean(dim=-1, keepdim=True),
0401 |             "ood_hat01": ood_hat01,             # <— NEW aligned name
0402 |             "temp01": temp01,                   # <— changed to sigmoid(tau_raw)
0403 |             "jacobian_fd": jac_fd,
0404 |             "len_effect": len_effect,
0405 |             "length_norm01": length_norm01,     # <— NEW 0..1 length proxy
0406 |             "halt_prob": halt_prob,             # <— NEW
0407 |         }
0408 | 
0409 |         if agree01 is not None:
0410 |             aux["agree01"] = agree01
0411 |         if sens01 is not None:
0412 |             aux["sens01"] = sens01
0413 | 
0414 |         return logits, halt_logits.squeeze(-1), z_final, (aux if return_aux else {})
0415 | 
0416 |     def self_test(self, *, device: str = "cpu", n_trials: int = 8) -> dict:
0417 |         import math
0418 | 
0419 |         import torch
0420 | 
0421 |         self.eval()
0422 |         dev = torch.device(device)
0423 |         self.to(dev)
0424 | 
0425 |         # Infer input dims safely
0426 |         def _infer_dim(attr_names, fallback=None):
0427 |             for a in attr_names:
0428 |                 v = getattr(self, a, None)
0429 |                 if isinstance(v, int) and v > 0:
0430 |                     return v
0431 |             return fallback
0432 | 
0433 |         # Prefer explicit attrs if you have them
0434 |         x_dim = _infer_dim(["x_dim", "d_x", "input_x_dim"], None)
0435 |         y_dim = _infer_dim(["y_dim", "d_y", "input_y_dim"], None)
0436 |         z_dim = _infer_dim(["z_dim", "d_z", "state_dim"], None)
0437 | 
0438 |         # Fall back to layer shapes
0439 |         # If you have projections: x_proj / y_proj
0440 |         if x_dim is None:
0441 |             xp = getattr(self, "x_proj", None)
0442 |             if xp is not None and hasattr(xp, "in_features"):
0443 |                 x_dim = int(xp.in_features)
0444 |         if y_dim is None:
0445 |             yp = getattr(self, "y_proj", None)
0446 |             if yp is not None and hasattr(yp, "in_features"):
0447 |                 y_dim = int(yp.in_features)
0448 | 
0449 |         # z_proj sees concat([x, y, z_cur]) so in_features = x+y+z
0450 |         zp = getattr(self, "z_proj", None)
0451 |         if zp is not None and hasattr(zp, "in_features"):
0452 |             fused_in = int(zp.in_features)
0453 |             if x_dim is None:
0454 |                 x_dim = max(1, fused_in // 3)
0455 |             if y_dim is None:
0456 |                 y_dim = max(1, fused_in // 3)
0457 |             if z_dim is None:
0458 |                 z_dim = max(1, fused_in - int(x_dim) - int(y_dim))
0459 | 
0460 |         # Last resort
0461 |         if x_dim is None: x_dim = 768
0462 |         if y_dim is None: y_dim = 768
0463 |         if z_dim is None: z_dim = 128
0464 | 
0465 |         scores = []
0466 |         reasons = []
0467 |         details = {"x_dim": x_dim, "y_dim": y_dim, "z_dim": z_dim, "trials": int(n_trials)}
0468 | 
0469 |         with torch.no_grad():
0470 |             for _ in range(max(1, int(n_trials))):
0471 |                 B = 4
0472 |                 x = torch.randn(B, x_dim, device=dev)
0473 |                 y = torch.randn(B, y_dim, device=dev)
0474 |                 z = torch.randn(B, z_dim, device=dev)
0475 | 
0476 |                 logits, halt_logits, z_final, aux = self(x, y, z, return_aux=True)
0477 | 
0478 |                 if not isinstance(aux, dict):
0479 |                     return {"ok": False, "summary": f"Tiny self_test: expected dict, got {type(aux)}", "details": details}
0480 | 
0481 |                 s = aux.get("score") 
0482 |                 if not torch.is_tensor(s):
0483 |                     return {"ok": False, "summary": "Tiny self_test: missing tensor score in aux['score'] or aux['score']", "details": details}
0484 | 
0485 |                 s = s.float().reshape(-1)
0486 |                 if not torch.isfinite(s).all():
0487 |                     return {"ok": False, "summary": "Tiny self_test: non-finite outputs (nan/inf)", "details": details}
0488 | 
0489 |                 scores.append(s.mean().item())
0490 | 
0491 |         mean = sum(scores) / len(scores)
0492 |         var = sum((v - mean) ** 2 for v in scores) / max(1, len(scores) - 1)
0493 |         std = math.sqrt(var)
0494 | 
0495 |         ok = True
0496 |         if std < 1e-6:
0497 |             ok = False
0498 |             reasons.append("collapsed outputs (std too small)")
0499 | 
0500 |         summary = (
0501 |             "Tiny self_test\n"
0502 |             f"  score: mean={mean:.6f} std={std:.6f} min={min(scores):.6f} max={max(scores):.6f}\n"
0503 |             f"  ok={ok} reasons={reasons}\n"
0504 |         )
0505 | 
0506 |         details.update({"mean": mean, "std": std, "min": min(scores), "max": max(scores), "reasons": reasons})
0507 |         return {"ok": ok, "summary": summary, "details": details}
```


---

## F0015 — `value_predictor.py`

```text
FILE_ID: F0015
PATH: value_predictor.py
LANGUAGE: python
LINES: 20
BYTES_UTF8: 614
SHA256: 1944fd2c59855d0a03594fd0efa1d8e85b049617aa83b2c159ef06ec601dfc27
```

```python
0001 | # stephanie/scoring/mrq/value_predictor.py
0002 | from __future__ import annotations
0003 | 
0004 | from torch import nn
0005 | 
0006 | 
0007 | class ValuePredictor(nn.Module):
0008 |     """Predicts a quality score for a document given its contextual embedding."""
0009 | 
0010 |     def __init__(self, zsa_dim=4096, hdim=2048):
0011 |         super().__init__()
0012 |         self.value_net = nn.Sequential(
0013 |             nn.Linear(zsa_dim, hdim), nn.ReLU(), nn.Linear(hdim, 1)
0014 |         )
0015 | 
0016 |     def forward(self, zsa_embedding):
0017 |         assert len(zsa_embedding.shape) == 2, (
0018 |             f"Expected 2D input, got {zsa_embedding.shape}"
0019 |         )
0020 |         return self.value_net(zsa_embedding)
```


---

## F0016 — `vpm.py`

```text
FILE_ID: F0016
PATH: vpm.py
LANGUAGE: python
LINES: 259
BYTES_UTF8: 8681
SHA256: e919df285562f80f5476dd9fd7270727d66670d07e64c21d238e03ba468ad86c
```

```python
0001 | # stephanie/model/vpm_model.py
0002 | from __future__ import annotations
0003 | 
0004 | import logging
0005 | from dataclasses import dataclass
0006 | from enum import Enum
0007 | from typing import Dict, List, Optional, Tuple
0008 | 
0009 | import numpy as np
0010 | import torch
0011 | import torch.nn as nn
0012 | 
0013 | log = logging.getLogger(__file__)
0014 | 
0015 | class VPMDimension(str, Enum):
0016 |     """Cognitive dimensions for scoring VPMs"""
0017 |     CLARITY = "clarity"
0018 |     NOVELTY = "novelty"
0019 |     CONFIDENCE = "confidence"
0020 |     CONTRADICTION = "contradiction"
0021 |     COHERENCE = "coherence"
0022 |     COMPLEXITY = "complexity"
0023 |     ALIGNMENT = "alignment"
0024 | 
0025 | @dataclass
0026 | class AttentionMap:
0027 |     """Container for attention map data for visualization"""
0028 |     layer: int
0029 |     head: int
0030 |     attention_weights: np.ndarray  # Shape: (num_patches, num_patches)
0031 |     patch_positions: np.ndarray   # Shape: (num_patches, 2) - (row, col)
0032 |     dimension: str                 # Which dimension this attention relates to
0033 | 
0034 | class PatchEmbedding(nn.Module):
0035 |     """Convert image patches to embeddings for transformer processing"""
0036 |     
0037 |     def __init__(self, img_size: int = 64, patch_size: int = 8, in_channels: int = 3, embed_dim: int = 128):
0038 |         super().__init__()
0039 |         self.img_size = img_size
0040 |         self.patch_size = patch_size
0041 |         self.n_patches = (img_size // patch_size) ** 2
0042 |         self.embed_dim = embed_dim
0043 |         
0044 |         # Convolutional approach to patch embedding (more efficient than linear projection)
0045 |         self.projection = nn.Conv2d(
0046 |             in_channels, 
0047 |             embed_dim, 
0048 |             kernel_size=patch_size, 
0049 |             stride=patch_size
0050 |         )
0051 |         
0052 |         # Positional embeddings
0053 |         self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
0054 |         
0055 |         # Class token
0056 |         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
0057 |         
0058 |         self.dropout = nn.Dropout(0.1)
0059 |         
0060 |         # Initialize weights
0061 |         self._init_weights()
0062 |     
0063 |     def _init_weights(self):
0064 |         """Initialize weights properly"""
0065 |         nn.init.normal_(self.position_embeddings, std=0.02)
0066 |         nn.init.normal_(self.cls_token, std=0.02)
0067 |         
0068 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
0069 |         """
0070 |         Convert image to patch embeddings.
0071 |         
0072 |         Args:
0073 |             x: Input tensor of shape (B, C, H, W)
0074 |             
0075 |         Returns:
0076 |             Patch embeddings of shape (B, n_patches + 1, embed_dim)
0077 |         """
0078 |         B, C, H, W = x.shape
0079 |         
0080 |         # Project patches
0081 |         x = self.projection(x)  # (B, embed_dim, n_patches_h, n_patches_w)
0082 |         x = x.flatten(2)        # (B, embed_dim, n_patches)
0083 |         x = x.transpose(1, 2)   # (B, n_patches, embed_dim)
0084 |         
0085 |         # Add class token
0086 |         cls_tokens = self.cls_token.expand(B, -1, -1)
0087 |         x = torch.cat((cls_tokens, x), dim=1)  # (B, n_patches + 1, embed_dim)
0088 |         
0089 |         # Add positional embeddings
0090 |         x = x + self.position_embeddings
0091 |         
0092 |         x = self.dropout(x)
0093 |         return x
0094 | 
0095 | class TransformerBlock(nn.Module):
0096 |     """Single transformer block with multi-head self-attention"""
0097 |     
0098 |     def __init__(self, embed_dim: int = 128, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
0099 |         super().__init__()
0100 |         self.norm1 = nn.LayerNorm(embed_dim)
0101 |         self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
0102 |         self.norm2 = nn.LayerNorm(embed_dim)
0103 |         
0104 |         # MLP feed-forward network
0105 |         mlp_hidden_dim = int(embed_dim * mlp_ratio)
0106 |         self.mlp = nn.Sequential(
0107 |             nn.Linear(embed_dim, mlp_hidden_dim),
0108 |             nn.GELU(),
0109 |             nn.Dropout(dropout),
0110 |             nn.Linear(mlp_hidden_dim, embed_dim),
0111 |             nn.Dropout(dropout)
0112 |         )
0113 |         
0114 |         self.dropout = dropout
0115 |         self.num_heads = num_heads
0116 |         self.embed_dim = embed_dim
0117 |         
0118 |     def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
0119 |         """
0120 |         Forward pass with optional attention return for introspection.
0121 |         
0122 |         Args:
0123 |             x: Input tensor of shape (B, n_patches + 1, embed_dim)
0124 |             return_attention: Whether to return attention weights
0125 |             
0126 |         Returns:
0127 |             Processed tensor and optionally attention weights
0128 |         """
0129 |         # Self-attention
0130 |         x_norm = self.norm1(x)
0131 |         attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=return_attention)
0132 |         
0133 |         # Residual connection
0134 |         x = x + attn_out
0135 |         
0136 |         # MLP
0137 |         x = x + self.mlp(self.norm2(x))
0138 |         
0139 |         if return_attention:
0140 |             return x, attn_weights
0141 |         return x, None
0142 | 
0143 | class TinyVisionTransformer(nn.Module):
0144 |     """Compact Vision Transformer optimized for VPM scoring"""
0145 |     
0146 |     def __init__(
0147 |         self,
0148 |         img_size: int = 64,
0149 |         patch_size: int = 8,
0150 |         in_channels: int = 3,
0151 |         embed_dim: int = 128,
0152 |         depth: int = 4,
0153 |         num_heads: int = 8,
0154 |         mlp_ratio: float = 4.0,
0155 |         dropout: float = 0.1,
0156 |         num_dimensions: int = 7
0157 |     ):
0158 |         super().__init__()
0159 |         self.embed_dim = embed_dim
0160 |         self.num_dimensions = num_dimensions
0161 |         
0162 |         # Patch embedding
0163 |         self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
0164 |         
0165 |         # Transformer blocks
0166 |         self.blocks = nn.ModuleList([
0167 |             TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) 
0168 |             for _ in range(depth)
0169 |         ])
0170 |         
0171 |         # Layer norm
0172 |         self.norm = nn.LayerNorm(embed_dim)
0173 |         
0174 |         # Head for multi-dimensional scoring
0175 |         self.head = nn.Sequential(
0176 |             nn.Linear(embed_dim, embed_dim // 2),
0177 |             nn.GELU(),
0178 |             nn.Dropout(dropout),
0179 |             nn.Linear(embed_dim // 2, num_dimensions),
0180 |             nn.Sigmoid()  # Output in [0,1] range for all dimensions
0181 |         )
0182 |         
0183 |         # Initialize weights
0184 |         self._init_weights()
0185 |         
0186 |         log.info(f"TinyVisionTransformer initialized: "
0187 |                 f"{depth} layers, {num_heads} heads, {embed_dim} embedding dim")
0188 |     
0189 |     def _init_weights(self):
0190 |         """Initialize weights for better training stability"""
0191 |         for m in self.modules():
0192 |             if isinstance(m, nn.Linear):
0193 |                 nn.init.xavier_uniform_(m.weight)
0194 |                 if m.bias is not None:
0195 |                     nn.init.constant_(m.bias, 0)
0196 |             elif isinstance(m, nn.LayerNorm):
0197 |                 nn.init.constant_(m.bias, 0)
0198 |                 nn.init.constant_(m.weight, 1.0)
0199 |     
0200 |     def forward_features(self, x: torch.Tensor) -> torch.Tensor:
0201 |         """Extract features from the transformer (without head)"""
0202 |         x = self.patch_embed(x)
0203 |         
0204 |         for block in self.blocks:
0205 |             x, _ = block(x)
0206 |             
0207 |         x = self.norm(x)
0208 |         # Use CLS token output
0209 |         return x[:, 0]
0210 |     
0211 |     def forward(
0212 |         self, 
0213 |         x: torch.Tensor, 
0214 |         return_attention: bool = False,
0215 |         attention_layers: Optional[List[int]] = None
0216 |     ) -> Dict[str, torch.Tensor]:
0217 |         """
0218 |         Forward pass with optional attention return.
0219 |         
0220 |         Args:
0221 |             x: Input tensor of shape (B, C, H, W)
0222 |             return_attention: Whether to return attention maps
0223 |             attention_layers: Which layers to return attention from (None = last layer)
0224 |             
0225 |         Returns:
0226 |             Dictionary with scores and optionally attention maps
0227 |         """
0228 |         x = self.patch_embed(x)
0229 |         
0230 |         attention_maps = []
0231 |         
0232 |         for i, block in enumerate(self.blocks):
0233 |             if return_attention and (attention_layers is None or i in attention_layers):
0234 |                 x, attn = block(x, return_attention=True)
0235 |                 attention_maps.append(attn)
0236 |             else:
0237 |                 x, _ = block(x)
0238 |         
0239 |         x = self.norm(x)
0240 |         # Use CLS token output
0241 |         cls_output = x[:, 0]
0242 |         
0243 |         # Get scores
0244 |         scores = self.head(cls_output)
0245 |         
0246 |         result = {"scores": scores}
0247 |         
0248 |         if return_attention:
0249 |             result["attention_maps"] = attention_maps
0250 |             result["patch_positions"] = self._get_patch_positions(x.shape[0])
0251 |             
0252 |         return result
0253 |     
0254 |     def _get_patch_positions(self, batch_size: int) -> torch.Tensor:
0255 |         """Get positions of patches in the original image"""
0256 |         # This would be calculated based on patch size and image dimensions
0257 |         # For now, return a placeholder
0258 |         device = next(self.parameters()).device
0259 |         return torch.zeros(batch_size, self.patch_embed.n_patches, 2, device=device)
```


---

## F0017 — `vpm_thought_policy.py`

```text
FILE_ID: F0017
PATH: vpm_thought_policy.py
LANGUAGE: python
LINES: 60
BYTES_UTF8: 2184
SHA256: 247d192fd9d2de9c1ac6d109b1dd10a23b5341453b39ed669b4b09769e6d287d
```

```python
0001 | from __future__ import annotations
0002 | 
0003 | from dataclasses import dataclass
0004 | from typing import Tuple
0005 | 
0006 | import torch
0007 | import torch.nn as nn
0008 | 
0009 | 
0010 | @dataclass
0011 | class VPMThoughtModelConfig:
0012 |     in_channels: int = 3
0013 |     hidden_dim: int = 512
0014 |     goal_dim: int = 128
0015 |     n_ops: int = 5
0016 |     param_dim: int = 8
0017 |     dropout: float = 0.1
0018 | 
0019 | class VPMSpatialEncoder(nn.Module):
0020 |     def __init__(self, in_channels: int, out_dim: int = 512):
0021 |         super().__init__()
0022 |         self.net = nn.Sequential(
0023 |             nn.Conv2d(in_channels, 64, 3, 2, 1), nn.ReLU(),
0024 |             nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
0025 |             nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
0026 |             nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(),
0027 |             nn.AdaptiveAvgPool2d(1), nn.Flatten(),
0028 |         )
0029 |         self.proj = nn.Sequential(nn.Linear(512, out_dim), nn.LayerNorm(out_dim), nn.ReLU())
0030 | 
0031 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
0032 |         return self.proj(self.net(x))
0033 | 
0034 | class VPMThoughtPolicy(nn.Module):
0035 |     """
0036 |     Stephanie-native wrapper around your thought policy.
0037 |     API: forward(vpm[B,C,H,W], goal_vec[B,G]) -> (op_logits[B,K], param_mean[B,D], param_log_std[B,D], value[B,1])
0038 |     """
0039 |     def __init__(self, cfg: VPMThoughtModelConfig):
0040 |         super().__init__()
0041 |         self.cfg = cfg
0042 |         C, H, G = cfg.in_channels, cfg.hidden_dim, cfg.goal_dim
0043 |         self.encoder = VPMSpatialEncoder(C, H)
0044 |         self.goal_proj = nn.Sequential(nn.Linear(C, G), nn.ReLU(), nn.Linear(G, G))
0045 |         self.fuser = nn.Sequential(nn.Linear(H + G, H), nn.ReLU(), nn.Dropout(cfg.dropout))
0046 |         self.op_head = nn.Linear(H, cfg.n_ops)
0047 |         self.param_head = nn.Linear(H, cfg.param_dim * 2)
0048 |         self.value_head = nn.Linear(H, 1)
0049 | 
0050 |     def forward(self, vpm: torch.Tensor, goal_vec: torch.Tensor) -> Tuple[torch.Tensor, ...]:
0051 |         s = self.encoder(vpm)
0052 |         g = self.goal_proj(goal_vec)
0053 |         h = self.fuser(torch.cat([s, g], dim=-1))
0054 |         op_logits = self.op_head(h)
0055 |         pr = self.param_head(h)
0056 |         D = self.cfg.param_dim
0057 |         param_mean = torch.tanh(pr[:, :D])
0058 |         param_log_std = pr[:, D:]
0059 |         value = self.value_head(h)
0060 |         return op_logits, param_mean, param_log_std, value
```


---

## F0018 — `vpm_vit.py`

```text
FILE_ID: F0018
PATH: vpm_vit.py
LANGUAGE: python
LINES: 138
BYTES_UTF8: 5443
SHA256: d6e3d02d2595bac347686f40daead3ea1f641038896b8f1b9d71dadaa0dfa967
```

```python
0001 | # stephanie/model/vpm_vit.py
0002 | from __future__ import annotations
0003 | 
0004 | import torch
0005 | import torch.nn as nn
0006 | 
0007 | 
0008 | # ---------- utils ----------
0009 | def _build_2d_sincos_pos_embed(h: int, w: int, d: int, cls_token: bool = True, device=None):
0010 |     """2D sin-cos positional embedding (ViT-style), shape (1, 1+N, D) if cls_token."""
0011 |     assert d % 4 == 0, "D must be divisible by 4 for 2D sincos."
0012 |     yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
0013 |     omega = torch.arange(d // 4, dtype=torch.float32)
0014 |     omega = 1.0 / (10000 ** (omega / (d // 4)))
0015 |     yy = yy.reshape(-1, 1).float()
0016 |     xx = xx.reshape(-1, 1).float()
0017 |     out = torch.cat([
0018 |         torch.sin(yy * omega), torch.cos(yy * omega),
0019 |         torch.sin(xx * omega), torch.cos(xx * omega)
0020 |     ], dim=1)  # (N, D)
0021 |     out = out.unsqueeze(0)  # (1, N, D)
0022 |     if cls_token:
0023 |         out = torch.cat([torch.zeros(1, 1, d), out], dim=1)
0024 |     return out.to(device=device)
0025 | 
0026 | # ---------- modules ----------
0027 | class PatchEmbed(nn.Module):
0028 |     def __init__(self, in_ch: int, d_model: int = 384, patch: int = 8):
0029 |         super().__init__()
0030 |         self.patch = patch
0031 |         self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)
0032 | 
0033 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
0034 |         # x: (B,C,H,W) -> (B,N,D)
0035 |         x = self.proj(x)
0036 |         x = x.flatten(2).transpose(1, 2)
0037 |         return x  # (B, N, D)
0038 | 
0039 | class TransformerBlock(nn.Module):
0040 |     def __init__(self, d_model: int = 384, n_heads: int = 6, mlp_ratio: float = 4.0, p: float = 0.1):
0041 |         super().__init__()
0042 |         self.norm1 = nn.LayerNorm(d_model)
0043 |         self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p, batch_first=True)
0044 |         self.norm2 = nn.LayerNorm(d_model)
0045 |         self.mlp = nn.Sequential(
0046 |             nn.Linear(d_model, int(d_model * mlp_ratio)),
0047 |             nn.GELU(),
0048 |             nn.Dropout(p),
0049 |             nn.Linear(int(d_model * mlp_ratio), d_model),
0050 |             nn.Dropout(p)
0051 |         )
0052 | 
0053 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
0054 |         y = self.norm1(x)
0055 |         y, _ = self.attn(y, y, y, need_weights=False)
0056 |         x = x + y
0057 |         x = x + self.mlp(self.norm2(x))
0058 |         return x
0059 | 
0060 | class VPMViT(nn.Module):
0061 |     """
0062 |     VPM-native ViT with optional MPM (masked patch token reconstruction).
0063 |     - Input: (B,C,H,W)
0064 |     - Output: {'reg': (B,T), 'cls': (B,K), 'mpm_rec': (M,D)} where M = total masked tokens across batch
0065 |     """
0066 |     def __init__(
0067 |         self,
0068 |         in_ch: int,
0069 |         d_model: int = 384,
0070 |         depth: int = 6,
0071 |         n_heads: int = 6,
0072 |         patch: int = 8,
0073 |         num_reg_targets: int = 5,
0074 |         num_risk_classes: int | None = 3,
0075 |         mlp_ratio: float = 4.0,
0076 |         p: float = 0.1,
0077 |         use_mpm: bool = True,
0078 |     ):
0079 |         super().__init__()
0080 |         self.patch = patch
0081 |         self.use_mpm = use_mpm
0082 |         self.patch_embed = PatchEmbed(in_ch, d_model, patch)
0083 |         self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
0084 |         self.pos_embed_cache: dict[tuple[int,int], torch.Tensor] = {}
0085 | 
0086 |         self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, mlp_ratio, p) for _ in range(depth)])
0087 |         self.norm = nn.LayerNorm(d_model)
0088 | 
0089 |         self.head_reg = nn.Linear(d_model, num_reg_targets) if num_reg_targets > 0 else None
0090 |         self.head_cls = nn.Linear(d_model, num_risk_classes) if num_risk_classes is not None else None
0091 |         self.mpm_head = nn.Linear(d_model, d_model) if use_mpm else None
0092 | 
0093 |         nn.init.trunc_normal_(self.cls_token, std=0.02)
0094 | 
0095 |     def _pos_embed(self, H: int, W: int, D: int, device) -> torch.Tensor:
0096 |         # cache by (H', W') where H' = H//p, W' = W//p
0097 |         h = H // self.patch
0098 |         w = W // self.patch
0099 |         key = (h, w)
0100 |         pe = self.pos_embed_cache.get(key)
0101 |         if pe is None or pe.device != device or pe.shape[-1] != D:
0102 |             pe = _build_2d_sincos_pos_embed(h, w, D, cls_token=True, device=device)  # (1,1+N,D)
0103 |             self.pos_embed_cache[key] = pe
0104 |         return pe
0105 | 
0106 |     def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
0107 |         B, C, H, W = x.shape
0108 |         patches = self.patch_embed(x)                 # (B, N, D)
0109 |         N, D = patches.shape[1], patches.shape[2]
0110 | 
0111 |         # tokens + pos
0112 |         cls = self.cls_token.expand(B, -1, -1)       # (B,1,D)
0113 |         xseq = torch.cat([cls, patches], dim=1)      # (B,1+N,D)
0114 |         xseq = xseq + self._pos_embed(H, W, D, x.device)
0115 | 
0116 |         for blk in self.blocks:
0117 |             xseq = blk(xseq)
0118 |         xseq = self.norm(xseq)
0119 | 
0120 |         cls_tok = xseq[:, 0]                         # (B,D)
0121 |         patch_tok = xseq[:, 1:]                      # (B,N,D)
0122 | 
0123 |         out: dict[str, torch.Tensor] = {}
0124 |         if self.head_reg is not None:
0125 |             out["reg"] = self.head_reg(cls_tok)
0126 |         if self.head_cls is not None:
0127 |             out["cls"] = self.head_cls(cls_tok)
0128 | 
0129 |         if self.use_mpm and mask is not None:
0130 |             # mask shape must be (B,N) boolean
0131 |             assert mask.dim() == 2 and mask.shape == (B, N), f"mask must be (B,N); got {tuple(mask.shape)}"
0132 |             rec = self.mpm_head(patch_tok)           # (B,N,D)
0133 |             out["mpm_rec"] = rec[mask]               # (M,D)
0134 |         return out
0135 | 
0136 | def vpm_vit_small(in_ch: int, targets: int = 5, classes: int = 3) -> VPMViT:
0137 |     return VPMViT(in_ch=in_ch, d_model=384, depth=6, n_heads=6, patch=8,
0138 |                   num_reg_targets=targets, num_risk_classes=classes, use_mpm=True)
```
