# stephanie/utils/ebt_buffer.py
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class EBTTrainingBuffer:
    """
    A training buffer for Energy-Based Transformers (EBT).
    Stores (context, candidate, llm_score) pairs for retraining.
    """
    
    def __init__(self, logger, path: str = "training_data/ebt_buffer.jsonl"):
        """
        Initialize the buffer with a file path for persistent storage.
        
        Args:
            path: Path to store the buffer (JSONL format)
        """
        self.logger = logger
        self.path = Path(path)
        self.buffer: List[Dict] = []
        self.max_size = 10000  # Max number of examples to keep in memory
        self._initialize_buffer()

    def _initialize_buffer(self):
        """Load existing buffer data or create a new one"""
        try:
            if self.path.exists():
                with open(self.path, "r") as f:
                    for line in f:
                        if line.strip():
                            self.buffer.append(json.loads(line))
                self.logger.log("EBTBufferLoaded", {
                    "size": len(self.buffer),
                    "path": str(self.path)
                })
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.logger.log("EBTBufferCreated", {"path": str(self.path)})
        except Exception as e:
            self.logger.log("EBTBufferError", {"error": str(e)})
            self.buffer = []

    def add(
        self,
        context: str,
        candidate: str,
        llm_score: float,
        ebt_score: float = None,
        metadata: Dict = None,
        source: str = "auto"
    ) -> None:
        """
        Add a new example to the buffer.
        
        Args:
            context: Goal or prompt text
            candidate: Generated or scored document
            llm_score: Ground truth score from LLM
            ebt_score: EBT's predicted score (optional)
            metadata: Additional context (e.g., dimension, task type)
            source: How this example was added (e.g., "auto", "manual")
        """
        example = {
            "context": context,
            "candidate": candidate,
            "llm_score": llm_score,
            "ebt_score": ebt_score,
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
            "disagreement": abs(llm_score - (ebt_score or 0)),
            "meta": metadata or {}
        }
        
        self.buffer.append(example)
        
        # Keep buffer size bounded
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)  # Remove oldest entry
            
        self._persist()
        self.logger.log("EBTExampleAdded", {
            "context_hash": hash(context[:50]),
            "candidate_hash": hash(candidate[:50]),
            "disagreement": round(example["disagreement"], 4)
        })

    def maybe_add(
        self,
        context: str,
        candidate: str,
        llm_score: float,
        ebt_score: float,
        threshold: float = 0.15,
        metadata: Dict = None
    ) -> bool:
        """
        Conditionally add to buffer based on disagreement threshold
        
        Args:
            threshold: Minimum disagreement to qualify for retraining
        Returns:
            True if example was added, False otherwise
        """
        if abs(llm_score - ebt_score) > threshold:
            self.add(context, candidate, llm_score, ebt_score, metadata, source="disagreement")
            return True
        return False

    def get_top_k_disagreements(self, k: int = 100) -> List[Dict]:
        """
        Get the top K most contentious examples for retraining
        
        Args:
            k: Number of examples to return
        """
        sorted_buffer = sorted(
            self.buffer,
            key=lambda x: x["disagreement"],
            reverse=True
        )
        return sorted_buffer[:k]

    def get_all(self) -> List[Dict]:
        """Get all examples in buffer"""
        return self.buffer

    def clear(self) -> None:
        """Clear the buffer (e.g., after retraining)"""
        self.buffer = []
        self._persist()
        self.logger.log("EBTBufferCleared", {"path": str(self.path)})

    def _persist(self) -> None:
        """Persist buffer to disk in JSONL format"""
        try:
            with open(self.path, "w") as f:
                for entry in self.buffer:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self.logger.log("EBTBufferSaveError", {"error": str(e)})

    def load(self, path: str = None) -> List[Dict]:
        """Load buffer from disk"""
        load_path = Path(path or self.path)
        if not load_path.exists():
            return []
            
        loaded = []
        with open(load_path, "r") as f:
            for line in f:
                if line.strip():
                    loaded.append(json.loads(line))
        self.buffer = loaded
        return loaded

    def analyze(self) -> Dict:
        """
        Analyze buffer contents for model improvement insights
        
        Returns:
            dict: {
                "total": int,
                "avg_disagreement": float,
                "worst_dimensions": List[str],
                "most_disputed_candidates": List[str]
            }
        """
        if not self.buffer:
            return {"error": "Buffer is empty"}
            
        # Calculate basic stats
        total = len(self.buffer)
        avg_disagreement = sum(e["disagreement"] for e in self.buffer) / total
        
        # Group by dimension
        from collections import defaultdict
        dim_disagreements = defaultdict(list)
        for entry in self.buffer:
            dim = entry["meta"].get("dimension", "unknown")
            dim_disagreements[dim].append(entry["disagreement"])
        
        # Find dimensions with highest disagreement
        avg_by_dim = {
            dim: sum(vals)/len(vals) for dim, vals in dim_disagreements.items()
        }
        sorted_dims = sorted(avg_by_dim.items(), key=lambda x: x[1], reverse=True)
        
        # Get most disputed examples
        top_disagreements = self.get_top_k_disagreements(k=5)
        
        analysis = {
            "total_examples": total,
            "avg_disagreement": round(avg_disagreement, 4),
            "disagreement_by_dimension": {k: round(v, 4) for k, v in sorted_dims},
            "top_disputed_examples": [
                {
                    "context": e["context"][:200] + "...",
                    "candidate": e["candidate"][:200] + "...",
                    "llm_score": e["llm_score"],
                    "ebt_score": e.get("ebt_score"),
                    "disagreement": e["disagreement"]
                }
                for e in top_disagreements
            ],
            "dimensions_with_most_disagreement": [d[0] for d in sorted_dims[:3]],
            "last_updated": datetime.utcnow().isoformat()
        }
        
        self.logger.log("EBTBufferAnalysis", analysis)
        return analysis

    def to_training_dataset(self, output_path: str = "training_data/ebt_retrain.jsonl") -> None:
        """
        Export buffer to training dataset
        
        Args:
            output_path: Where to save the dataset
        """
        with open(output_path, "w") as f:
            for entry in self.buffer:
                # Convert to training example
                training_example = {
                    "context": entry["context"],
                    "candidate": entry["candidate"],
                    "score": entry["llm_score"]
                }
                f.write(json.dumps(training_example) + "\n")
                
        self.logger.log("EBTTrainingDatasetSaved", {
            "examples": len(self.buffer),
            "path": output_path
        })

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def __contains__(self, item):
        # Implement based on your use case
        # For example, check by context-candidate hash
        return any(
            hash(e["context"]) == hash(item["context"]) and 
            hash(e["candidate"]) == hash(item["candidate"])
            for e in self.buffer
        ) 
