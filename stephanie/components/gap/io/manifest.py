# stephanie/components/gap/io/manifest.py
from __future__ import annotations
import json
from typing import Any, Dict
from pathlib import Path

from ..models import GapRunManifest
from .storage import GapStorage


class ManifestManager:
    """Manages manifest operations and updates."""
    
    def __init__(self, storage: GapStorage):
        self.storage = storage
    
    async def initialize_run(self, manifest: GapRunManifest) -> None:
        """Initialize a new run with directory structure and manifest."""
        # Create directory structure
        paths = self.storage.initialize_run_directory(manifest.run_id)
        manifest.paths = {k: str(v) for k, v in paths.items()}
        
        # Save initial manifest
        self.storage.save_manifest(manifest, paths["root"])
    
    async def update_manifest(self, run_id: str, updates: Dict[str, Any]) -> None:
        """Update manifest with new data."""
        current = self.storage.load_manifest(run_id)
        if current is None:
            raise ValueError(f"Manifest not found for run: {run_id}")
        
        # Deep merge updates
        self._deep_merge(current, updates)
        
        # Save updated manifest
        run_path = self.storage.base_dir / run_id
        with open(run_path / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
    
    async def finalize_run(self, manifest: GapRunManifest, results: Dict[str, Any]) -> None:
        """Finalize run by adding results to manifest."""
        updates = {
            "artifacts": results,
            "stats": {
                "completed_at": ...  # Add completion timestamp
            }
        }
        
        await self.update_manifest(manifest.run_id, updates)
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Recursively merge source into target."""
        for key, value in source.items():
            if (isinstance(value, dict) and 
                isinstance(target.get(key), dict)):
                self._deep_merge(target[key], value)
            else:
                target[key] = value