# stephanie/components/gap/io/storage.py
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from stephanie.utils.json_sanitize import dumps_safe
from stephanie.components.gap.models import GapRunManifest, TripleSample


class GapStorage:
    """Handles all file system operations for GAP analysis."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
    
    def initialize_run_directory(self, run_id: str) -> Dict[str, Path]:
        """Create directory structure for a run."""
        run_path = self.base_dir / run_id
        
        directories = {
            "root": run_path,
            "raw": run_path / "raw",
            "aligned": run_path / "aligned", 
            "visuals": run_path / "visuals",
            "metrics": run_path / "metrics",
            "reports": run_path / "reports",
        }
        
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)
            
        return directories
    
    def save_manifest(self, manifest: GapRunManifest, run_path: Path) -> None:
        """Save manifest to disk."""
        manifest_path = run_path / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(dumps_safe(self._manifest_to_dict(manifest), indent=2))
    
    def load_manifest(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load manifest from disk."""
        manifest_path = self.base_dir / run_id / "manifest.json"
        if not manifest_path.exists():
            return None
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def save_triples(self, triples: List[TripleSample], run_id: str) -> Dict[str, Path]:
        """Save triples data in multiple formats."""
        run_path = self.base_dir / run_id
        raw_dir = run_path / "raw"
        
        # JSONL format
        jsonl_path = raw_dir / "triples.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for triple in triples:
                f.write(dumps_safe(self._triple_to_dict(triple)))
                f.write("\n")
        
        # Parquet format for fast loading
        parquet_path = raw_dir / "triples.parquet"
        df = pd.DataFrame([self._triple_to_dict(triple) for triple in triples])
        df.to_parquet(parquet_path, index=False)
        
        # Head sample for inspection
        head_path = raw_dir / "triples.head.json"
        with open(head_path, "w", encoding="utf-8") as f:
            json.dump([self._triple_to_dict(triple) for triple in triples[:200]], f, indent=2)
        
        return {
            "jsonl": jsonl_path,
            "parquet": parquet_path,
            "head": head_path
        }
    
    def save_matrix(self, matrix: np.ndarray, names: List[str], run_id: str, tag: str) -> Dict[str, Path]:
        """Save aligned matrix and metric names."""
        aligned_dir = self.base_dir / run_id / "aligned"
        
        matrix_path = aligned_dir / f"{tag}_matrix.npy"
        names_path = aligned_dir / f"{tag}_metric_names.json"
        
        np.save(matrix_path, matrix.astype(np.float32))
        with open(names_path, "w", encoding="utf-8") as f:
            json.dump(names, f, indent=2)
        
        return {
            "matrix": matrix_path,
            "names": names_path
        }
    
    def load_matrix(self, run_id: str, tag: str) -> tuple[np.ndarray, List[str]]:
        """Load aligned matrix and metric names."""
        aligned_dir = self.base_dir / run_id / "aligned"
        
        matrix_path = aligned_dir / f"{tag}_matrix.npy"
        names_path = aligned_dir / f"{tag}_metric_names.json"
        
        matrix = np.load(matrix_path)
        with open(names_path, "r", encoding="utf-8") as f:
            names = json.load(f)
        
        return matrix, names
    
    def copy_visual_artifact(self, source_path: str, run_id: str, artifact_name: str) -> Path:
        """Copy visual artifact to run directory."""
        visuals_dir = self.base_dir / run_id / "visuals"
        destination_path = visuals_dir / artifact_name
        
        if Path(source_path).exists():
            shutil.copy2(source_path, destination_path)
        
        return destination_path
    
    def save_metrics_report(self, metrics: Dict[str, Any], run_id: str, report_name: str) -> Path:
        """Save metrics report to JSON."""
        metrics_dir = self.base_dir / run_id / "metrics"
        report_path = metrics_dir / f"{report_name}.json"
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        
        return report_path
    
    def _manifest_to_dict(self, manifest: GapRunManifest) -> Dict[str, Any]:
        """Convert manifest to dictionary for serialization."""
        return {
            "run_id": manifest.run_id,
            "dataset": manifest.dataset,
            "models": manifest.models,
            "dimensions": manifest.dimensions,
            "preproc_version": manifest.preproc_version,
            "created_at": manifest.created_at,
            "paths": manifest.paths,
            "stats": manifest.stats,
            "artifacts": manifest.artifacts
        }
    
    def _triple_to_dict(self, triple: TripleSample) -> Dict[str, Any]:
        """Convert triple to dictionary for serialization."""
        return {
            "node_id": triple.node_id,
            "dimension": triple.dimension,
            "goal_text": triple.goal_text,
            "output_text": triple.output_text,
            "target": triple.target_value,
            "fingerprint": triple.fingerprint
        }