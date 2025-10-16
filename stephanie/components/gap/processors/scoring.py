# stephanie/components/gap/processors/scoring.py
from __future__ import annotations
import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
import numpy as np

from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.services.workers.metrics_worker import MetricsWorkerInline
from stephanie.services.workers.vpm_worker import VPMWorkerInline
from ..models import GapConfig, TripleSample, GapRunManifest


logger = logging.getLogger(__name__)


class ScoringProcessor:
    """Handles sample preparation, scoring, and timeline generation."""
    
    def __init__(self, config: GapConfig, container, logger):
        self.config = config
        self.container = container
        self.logger = logger
    
    async def prepare_samples(self, dimensions: List[str], memory) -> Dict[str, List[TripleSample]]:
        """Collect and deduplicate training samples across dimensions."""
        from stephanie.scoring.training.preference_pair_builder import PreferencePairBuilder
        
        pair_builder = PreferencePairBuilder(memory, self.logger)
        triples_by_dim = {}
        
        for dimension in dimensions:
            pairs = pair_builder.get_training_pairs_by_dimension(dimension=dimension)
            samples = pairs.get(dimension, [])
            triples = self._flatten_samples(samples, dimension)
            triples_by_dim[dimension] = triples
        
        return self._deduplicate_samples(triples_by_dim)
    
    async def execute_scoring(self, triples_data: Dict[str, List[TripleSample]], 
                            run_id: str, manifest: GapRunManifest) -> Dict[str, Any]:
        """Execute model scoring and generate timelines."""
        scoring_service = self.container.get("scoring")
        zm = self.container.get("zeromodel")
        
        # Initialize workers
        hrm_worker = MetricsWorkerInline(scoring_service, self.config.hrm_scorers, self.config.dimensions)
        tiny_worker = MetricsWorkerInline(scoring_service, self.config.tiny_scorers, self.config.dimensions)
        vpm_worker = VPMWorkerInline(zm, self.logger)
        
        # Execute scoring pipeline
        results = await self._score_all_triples(
            triples_data, hrm_worker, tiny_worker, vpm_worker, run_id
        )
        
        return results
    
    def _flatten_samples(self, samples: List[Dict[str, Any]], dimension: str) -> List[TripleSample]:
        """Convert raw samples to TripleSample objects."""
        triples = []
        
        for i, sample in enumerate(samples):
            # Extract goal text
            goal_text = (sample.get("goal_text") or sample.get("title") or "").strip()
            
            # Handle different sample formats
            if "output" in sample and ("score" in sample or "target_score" in sample):
                output_text = (sample.get("output") or "").strip()
                value = sample.get("target_score", sample.get("score"))
                
                if goal_text and output_text and value is not None:
                    fingerprint = self._compute_fingerprint(goal_text, output_text)
                    triple = TripleSample(
                        node_id=f"{dimension}|{i:06d}",
                        dimension=dimension,
                        goal_text=goal_text,
                        output_text=output_text,
                        target_value=float(value),
                        fingerprint=fingerprint
                    )
                    triples.append(triple)
            
            # Handle pairwise samples
            elif all(k in sample for k in ("output_a", "output_b", "value_a", "value_b")):
                for suffix in ["a", "b"]:
                    output_text = (sample.get(f"output_{suffix}") or "").strip()
                    value = sample.get(f"value_{suffix}")
                    
                    if goal_text and output_text and value is not None:
                        fingerprint = self._compute_fingerprint(goal_text, output_text)
                        triple = TripleSample(
                            node_id=f"{dimension}|{i:06d}_{suffix}",
                            dimension=dimension,
                            goal_text=goal_text,
                            output_text=output_text,
                            target_value=float(value),
                            fingerprint=fingerprint
                        )
                        triples.append(triple)
        
        return triples
    
    def _deduplicate_samples(self, triples_by_dim: Dict[str, List[TripleSample]]) -> Dict[str, List[TripleSample]]:
        """Deduplicate samples across dimensions using configured policy."""
        if self.config.dedupe_policy == "first_wins":
            return self._deduplicate_first_wins(triples_by_dim)
        elif self.config.dedupe_policy == "round_robin":
            return self._deduplicate_round_robin(triples_by_dim)
        else:
            raise ValueError(f"Unknown deduplication policy: {self.config.dedupe_policy}")
    
    def _deduplicate_first_wins(self, triples_by_dim: Dict[str, List[TripleSample]]) -> Dict[str, List[TripleSample]]:
        """First dimension that encounters a sample keeps it."""
        seen_fingerprints = set()
        deduped = {dim: [] for dim in triples_by_dim.keys()}
        
        for dim, triples in triples_by_dim.items():
            for triple in triples:
                if triple.fingerprint not in seen_fingerprints:
                    deduped[dim].append(triple)
                    seen_fingerprints.add(triple.fingerprint)
            
            # Apply per-dimension cap
            if self.config.per_dim_cap and len(deduped[dim]) > self.config.per_dim_cap:
                deduped[dim] = deduped[dim][:self.config.per_dim_cap]
        
        return deduped
    
    def _deduplicate_round_robin(self, triples_by_dim: Dict[str, List[TripleSample]]) -> Dict[str, List[TripleSample]]:
        """Distribute samples evenly across dimensions."""
        # Collect all unique samples
        unique_samples = {}
        for dim, triples in triples_by_dim.items():
            for triple in triples:
                if triple.fingerprint not in unique_samples:
                    unique_samples[triple.fingerprint] = triple
        
        # Distribute evenly
        dimensions = list(triples_by_dim.keys())
        deduped = {dim: [] for dim in dimensions}
        
        for i, (fingerprint, triple) in enumerate(unique_samples.items()):
            target_dim = dimensions[i % len(dimensions)]
            if (self.config.per_dim_cap is None or 
                len(deduped[target_dim]) < self.config.per_dim_cap):
                deduped[target_dim].append(triple)
        
        return deduped
    
    def _compute_fingerprint(self, goal_text: str, output_text: str) -> str:
        """Compute fingerprint for deduplication."""
        content = (goal_text.strip() + "\n␟\n" + output_text.strip()).encode("utf-8")
        return hashlib.sha1(content).hexdigest()
    
    async def _score_all_triples(self, triples_data: Dict[str, List[TripleSample]],
                               hrm_worker, tiny_worker, vpm_worker, run_id: str) -> Dict[str, Any]:
        """Score all triples with both models and generate timelines."""
        # Combine all triples across dimensions
        all_triples = []
        for dim_triples in triples_data.values():
            all_triples.extend(dim_triples)
        
        # Initialize timelines
        zm = self.container.get("zeromodel")
        hrm_timeline_id = f"{run_id}_hrm"
        tiny_timeline_id = f"{run_id}_tiny"
        zm.timeline_open(run_id=hrm_timeline_id)
        zm.timeline_open(run_id=tiny_timeline_id)
        
        # Score each triple
        hrm_vectors, tiny_vectors = [], []
        hrm_names, tiny_names = [], []
        
        with tqdm(total=len(all_triples), desc="[GAP] Scoring triples", unit="turn") as pbar:
            for i, triple in enumerate(all_triples):
                scorable = Scorable(triple.output_text, ScorableType.CONVERSATION_TURN)
                
                # Score with both models
                hrm_metrics = await hrm_worker.score(scorable, triple.goal_text, hrm_timeline_id)
                tiny_metrics = await tiny_worker.score(scorable, triple.goal_text, tiny_timeline_id)
                
                # Append to timelines
                await vpm_worker.append(hrm_timeline_id, triple.node_id, hrm_metrics)
                await vpm_worker.append(tiny_timeline_id, triple.node_id, tiny_metrics)
                
                # Extract vectors
                hrm_vec = self._extract_metrics_vector(hrm_metrics)
                tiny_vec = self._extract_metrics_vector(tiny_metrics)
                
                # Handle vector alignment
                if i == 0:
                    hrm_names = list(hrm_vec.keys()) if isinstance(hrm_vec, dict) else []
                    tiny_names = list(tiny_vec.keys()) if isinstance(tiny_vec, dict) else []
                
                hrm_vectors.append(self._align_vector(hrm_vec, hrm_names))
                tiny_vectors.append(self._align_vector(tiny_vec, tiny_names))
                
                if (i + 1) % self.config.progress_log_every == 0:
                    self.logger.log("ScoringProgress", {
                        "processed": i + 1,
                        "total": len(all_triples)
                    })
                
                pbar.update(1)
                await asyncio.sleep(0)  # Cooperative yield
        
        # Finalize timelines
        hrm_gif = await vpm_worker.finalize(hrm_timeline_id, f"vpm_phos_run_{hrm_timeline_id}.gif")
        tiny_gif = await vpm_worker.finalize(tiny_timeline_id, f"vpm_phos_run_{tiny_timeline_id}.gif")
        
        return {
            "hrm_vectors": np.array(hrm_vectors, dtype=np.float32),
            "tiny_vectors": np.array(tiny_vectors, dtype=np.float32),
            "hrm_names": hrm_names,
            "tiny_names": tiny_names,
            "hrm_gif": hrm_gif,
            "tiny_gif": tiny_gif,
            "triples_count": len(all_triples)
        }
    
    def _extract_metrics_vector(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract metrics vector from metrics payload."""
        # Implementation from your _safe_vec function
        vector = metrics.get("vector")
        if isinstance(vector, dict) and vector:
            return {k: float(v) for k, v in vector.items()}
        
        columns = metrics.get("columns")
        values = metrics.get("values")
        if (isinstance(columns, list) and 
            isinstance(values, list) and 
            len(columns) == len(values)):
            return {str(col): float(val) for col, val in zip(columns, values)}
        
        return {}
    
    def _align_vector(self, vector: Dict[str, float], target_names: List[str]) -> List[float]:
        """Align vector to target names, filling missing values with 0.0."""
        return [float(vector.get(name, 0.0)) for name in target_names]