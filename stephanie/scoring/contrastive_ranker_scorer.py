import torch
import numpy as np
from joblib import load
from tqdm import tqdm
from torch import nn 
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json


class PreferenceRanker(nn.Module):
    """Siamese network architecture (must match trainer)"""
    def __init__(self, embedding_dim=1024, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.comparator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, emb_a, emb_b):
        feat_a = self.encoder(emb_a)
        feat_b = self.encoder(emb_b)
        combined = torch.cat([feat_a, feat_b], dim=1)
        return self.comparator(combined).squeeze(1)


class ContrastiveRankerScorer(BaseScorer):
    def __init__(self, cfg: dict, memory, logger):
        super().__init__(cfg, memory, logger)
        self.model_type = "contrastive_ranker"
        self.models = {}        # dim -> (scaler, model)
        self.tuners = {}        # dim -> RegressionTuner
        self.metas = {}         # dim -> model metadata
        self.baselines = {}     # dim -> baseline embedding
        self._load_all_dimensions()

    def _load_all_dimensions(self):
        """Preload all dimension models with baseline caching"""
        for dim in tqdm(self.dimensions, desc="Loading contrastive rankers"):
            locator = self.get_locator(dim)
            
            # Load metadata first
            meta = load_json(locator.meta_file())
            self.metas[dim] = meta
            
            # Load scaler
            scaler = load(locator.scaler_file())
            
            # Initialize model with correct dimensions
            input_dim = scaler.mean_.shape[0]
            model = PreferenceRanker(
                embedding_dim=input_dim,
                hidden_dim=meta["hidden_dim"]
            )
            
            # Load weights
            model.load_state_dict(torch.load(locator.model_file(suffix=".pt")))
            model.eval()
            self.models[dim] = (scaler, model)
            
            # Load tuner
            tuner = RegressionTuner(dimension=dim, logger=self.logger)
            tuner.load(locator.tuner_file())
            self.tuners[dim] = tuner
            
            # Precompute baseline embedding
            baseline_text = meta["baseline"]
            baseline_emb = np.array(self.memory.embedding.get_or_create(baseline_text))
            self.baselines[dim] = baseline_emb

    def score(self, goal: dict, scorable: Scorable, dimensions: list[str]) -> ScoreBundle:
        """Generate absolute scores via baseline comparison"""
        goal_text = goal.get("goal_text", "")
        ctx_emb = np.array(self.memory.embedding.get_or_create(goal_text))
        doc_emb = np.array(self.memory.embedding.get_or_create(scorable.text))
        
        results = {}
        for dim in dimensions:
            scaler, model = self.models[dim]
            tuner = self.tuners[dim]
            meta = self.metas[dim]
            baseline_emb = self.baselines[dim]
            
            # Create comparison inputs
            input_doc = np.concatenate([ctx_emb, doc_emb])
            input_baseline = np.concatenate([ctx_emb, baseline_emb])
            
            # Scale inputs
            input_doc_scaled = scaler.transform(input_doc.reshape(1, -1))
            input_baseline_scaled = scaler.transform(input_baseline.reshape(1, -1))
            
            # Convert to tensors
            doc_tensor = torch.tensor(input_doc_scaled, dtype=torch.float32)
            baseline_tensor = torch.tensor(input_baseline_scaled, dtype=torch.float32)
            
            # Get preference score
            with torch.no_grad():
                raw_score = model(doc_tensor, baseline_tensor).item()
            
            # Calibrate to absolute score
            tuned_score = tuner.transform(raw_score)
            final_score = max(min(tuned_score, meta["max_value"]), meta["min_value"])

            attributes = {
                "raw_score": round(raw_score, 4),
                "normalized_score": round(tuned_score, 4),
                "final_score": final_score,
                "energy": raw_score,  # Using raw_score as energy
            }

            results[dim] = ScoreResult(
                dimension=dim,
                score=final_score,
                rationale=f"PrefScore(raw={raw_score:.4f}, tuned={tuned_score:.2f})",
                weight=1.0,
                source=self.model_type,
                attributes=attributes,
                )
        
        return ScoreBundle(results=results)