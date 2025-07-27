# stephanie/memory/belief_cartridge_store.py

import json
import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
import yaml
from sqlalchemy.orm import Session

from stephanie.models.belief_cartridge import BeliefCartridgeORM
from stephanie.scoring.score_bundle import ScoreBundle


class BeliefCartridgeStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "belief_cartridges"

    def add_or_update_cartridge(self, data: dict) -> BeliefCartridgeORM:
        existing = self.session.query(BeliefCartridgeORM).filter_by(id=data["id"]).first()

        if existing:
            # Update only certain fields
            existing.updated_at = datetime.utcnow()
            existing.markdown_content = data.get("markdown_content", existing.markdown_content)
            existing.idea_payload = data.get("idea_payload", existing.idea_payload)
            existing.rationale = data.get("rationale", existing.rationale)
            existing.source_url = data.get("source_url", existing.source_url)
            existing.is_active = data.get("is_active", existing.is_active)
            self.session.commit()
            return existing

        # Create new
        cartridge = BeliefCartridgeORM(**data)
        self.session.add(cartridge)
        self.session.commit()
        return cartridge

    def bulk_add(self, items: list[dict]) -> list[BeliefCartridgeORM]:
        cartridges = [BeliefCartridgeORM(**item) for item in items]
        self.session.add_all(cartridges)
        self.session.commit()
        return cartridges

    def get_by_id(self, belief_id: str) -> BeliefCartridgeORM | None:
        return self.session.query(BeliefCartridgeORM).filter_by(id=belief_id).first()

    def get_by_source(self, source_url: str) -> list[BeliefCartridgeORM]:
        return self.session.query(BeliefCartridgeORM).filter_by(source_url=source_url).all()

    def get_all(self, limit: int = 100) -> list[BeliefCartridgeORM]:
        return self.session.query(BeliefCartridgeORM).order_by(BeliefCartridgeORM.created_at.desc()).limit(limit).all()

    def delete_by_id(self, belief_id: str) -> bool:
        belief = self.get_by_id(belief_id)
        if belief:
            self.session.delete(belief)
            self.session.commit()
            return True
        return False

    def deactivate_by_id(self, belief_id: str) -> bool:
        belief = self.get_by_id(belief_id)
        if belief:
            belief.is_active = False
            belief.updated_at = datetime.utcnow()
            self.session.commit()
            return True
        return False
    

    def exists_by_source(self, source_id: int) -> bool:
        count = self.session.query(BeliefCartridgeORM).filter(
            BeliefCartridgeORM.source_id == str(source_id)
        ).count()
        return count > 0
    
    def export_to_yaml(self, cartridge_id: str, export_dir: str = "exports/belief_cartridges"):
        """Export belief cartridge to YAML file"""
        try:
            # Get cartridge from DB
            cartridge = self.session.query(BeliefCartridgeORM).get(cartridge_id)
            if not cartridge:
                raise ValueError(f"Cartridge {cartridge_id} not found")
            
            # Convert to dictionary
            data = self._to_serializable_dict(cartridge)
            
            # Save to YAML
            os.makedirs(export_dir, exist_ok=True)
            file_path = os.path.join(export_dir, f"{cartridge_id}.yaml")
            
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)
            
            self.logger.log("BeliefCartridgeExported", {
                "cartridge_id": cartridge_id,
                "format": "yaml",
                "path": file_path
            })
            return file_path
            
        except Exception as e:
            self.logger.log("ExportFailed", {
                "cartridge_id": cartridge_id,
                "format": "yaml",
                "error": str(e)
            })
            raise

    def export_to_json(self, cartridge_id: str, export_dir: str = "exports/belief_cartridges"):
        """Export belief cartridge to JSON file"""
        try:
            # Get cartridge from DB
            cartridge = self.session.query(BeliefCartridgeORM).get(cartridge_id)
            if not cartridge:
                raise ValueError(f"Cartridge {cartridge_id} not found")
            
            # Convert to dictionary
            data = self._to_serializable_dict(cartridge)
            
            # Save to JSON
            os.makedirs(export_dir, exist_ok=True)
            file_path = os.path.join(export_dir, f"{cartridge_id}.json")
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=self._default_serializer)
            
            self.logger.log("BeliefCartridgeExported", {
                "cartridge_id": cartridge_id,
                "format": "json",
                "path": file_path
            })
            return file_path
            
        except Exception as e:
            self.logger.log("ExportFailed", {
                "cartridge_id": cartridge_id,
                "format": "json",
                "error": str(e)
            })
            raise

    def bulk_export(self, export_dir: str = "exports/belief_cartridges", format: str = "yaml"):
        """Export all active belief cartridges to files"""
        try:
            # Get active cartridges
            cartridges = self.session.query(BeliefCartridgeORM).filter_by(is_active=True).all()
            
            # Export each one
            paths = []
            for cartridge in cartridges:
                if format == "yaml":
                    path = self.export_to_yaml(cartridge.id, export_dir)
                elif format == "json":
                    path = self.export_to_json(cartridge.id, export_dir)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                paths.append(path)
            
            self.logger.log("BulkExportComplete", {
                "cartridge_count": len(cartridges),
                "format": format,
                "export_dir": export_dir
            })
            return paths
            
        except Exception as e:
            self.logger.log("BulkExportFailed", {
                "error": str(e),
                "format": format
            })
            raise

    def _to_serializable_dict(self, cartridge: BeliefCartridgeORM) -> Dict[str, Any]:
        """Convert ORM object to serializable dictionary"""
        data = {
            "id": cartridge.id,
            "created_at": cartridge.created_at,
            "updated_at": cartridge.updated_at,
            "source_type": cartridge.source_type,
            "source_id": cartridge.source_id,
            "source_url": cartridge.source_url,
            "markdown_content": cartridge.markdown_content,
            "idea_payload": self._deserialize_json(cartridge.idea_payload),
            "goal_tags": cartridge.goal_tags,
            "domain_tags": cartridge.domain_tags,
            "derived_from": cartridge.derived_from,
            "applied_in": cartridge.applied_in,
            "version": cartridge.version,
            "memcube_id": cartridge.memcube_id,
            "is_active": cartridge.is_active,
            "evaluations": [e.id for e in cartridge.evaluations]
        }
        
        # Add scoring metadata
        if cartridge.evaluations:
            scores = {}
            for eval in cartridge.evaluations:
                for dim, score in eval.scores.items():
                    scores[dim] = round(float(score), 2)
            data["scores"] = scores
        
        return data

    def _default_serializer(self, obj):
        """Custom JSON serializer for non-serializable objects"""
        if isinstance(obj, ScoreBundle):
            return obj.to_dict()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")

    def _deserialize_json(self, value: Any) -> Any:
        """Ensure JSON values are serializable"""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {k: self._deserialize_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._deserialize_json(v) for v in value]
        return str(value)  # Fallback for non-serializable types