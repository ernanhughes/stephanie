from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from sqlalchemy.orm import Session

from stephanie.data.score_bundle import ScoreBundle
from stephanie.models.belief_cartridge import BeliefCartridgeORM
from stephanie.memory.base_store import BaseSQLAlchemyStore


class BeliefCartridgeStore(BaseSQLAlchemyStore):
    orm_model = BeliefCartridgeORM
    default_order_by = "created_at"   # ✅ let Base handle .desc()

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger=logger)
        self.name = "belief_cartridges"

    def add_or_update_cartridge(self, data: dict) -> BeliefCartridgeORM:
        def op(s):
            existing = s.query(BeliefCartridgeORM).filter_by(id=data["id"]).first()
            if existing:
                existing.updated_at = datetime.now()
                existing.markdown_content = data.get("markdown_content", existing.markdown_content)
                existing.idea_payload = data.get("idea_payload", existing.idea_payload)
                existing.rationale = data.get("rationale", existing.rationale)
                existing.source_url = data.get("source_url", existing.source_url)
                existing.is_active = data.get("is_active", existing.is_active)
                return existing
            cartridge = BeliefCartridgeORM(**data)
            s.add(cartridge)
            return cartridge
        return self._run(op)

    def bulk_add(self, items: List[dict]) -> List[BeliefCartridgeORM]:
        def op(s):
            cartridges = [BeliefCartridgeORM(**item) for item in items]
            s.add_all(cartridges)
            return cartridges
        return self._run(op)

    def get_by_id(self, belief_id: str) -> BeliefCartridgeORM | None:
        def op(s):
            return s.get(BeliefCartridgeORM, belief_id)
        return self._run(op)

    def get_by_source(self, source_url: str) -> list[BeliefCartridgeORM]:
        def op(s):
            return s.query(BeliefCartridgeORM).filter_by(source_url=source_url).all()
        return self._run(op)

    def get_all(self, limit: int = 100) -> list[BeliefCartridgeORM]:
        def op(s):
            return (
                s.query(BeliefCartridgeORM)
                .order_by(BeliefCartridgeORM.created_at.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)

    def delete_by_id(self, belief_id: str) -> bool:
        def op(s):
            belief = s.get(BeliefCartridgeORM, belief_id)
            if not belief:
                return False
            s.delete(belief)
            return True
        return self._run(op)

    def deactivate_by_id(self, belief_id: str) -> bool:
        def op(s):
            belief = s.get(BeliefCartridgeORM, belief_id)
            if not belief:
                return False
            belief.is_active = False
            belief.updated_at = datetime.now()
            return True
        return self._run(op)

    def exists_by_source(self, source_id: int) -> bool:
        def op(s):
            return (
                s.query(BeliefCartridgeORM)
                .filter(BeliefCartridgeORM.source_id == str(source_id))
                .count()
                > 0
            )
        return self._run(op)

    # --- Export methods can also wrap scope (since they read from DB) ---
    def export_to_yaml(self, cartridge_id: str, export_dir: str = "exports/belief_cartridges"):
        def op(s):
            cartridge = s.get(BeliefCartridgeORM, cartridge_id)
            if not cartridge:
                raise ValueError(f"Cartridge {cartridge_id} not found")
            data = self._to_serializable_dict(cartridge)
            os.makedirs(export_dir, exist_ok=True)
            file_path = os.path.join(export_dir, f"{cartridge_id}.yaml")
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)
            return file_path
        return self._run(op)

    def export_to_json(self, cartridge_id: str, export_dir: str = "exports/belief_cartridges"):
        def op(s):
            cartridge = s.get(BeliefCartridgeORM, cartridge_id)
            if not cartridge:
                raise ValueError(f"Cartridge {cartridge_id} not found")
            data = self._to_serializable_dict(cartridge)
            os.makedirs(export_dir, exist_ok=True)
            file_path = os.path.join(export_dir, f"{cartridge_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=self._default_serializer)
            return file_path
        return self._run(op)

    def bulk_export(self, export_dir: str = "exports/belief_cartridges", format: str = "yaml"):
        def op(s):
            cartridges = s.query(BeliefCartridgeORM).filter_by(is_active=True).all()
            paths = []
            for cartridge in cartridges:
                if format == "yaml":
                    paths.append(self.export_to_yaml(cartridge.id, export_dir))
                elif format == "json":
                    paths.append(self.export_to_json(cartridge.id, export_dir))
                else:
                    raise ValueError(f"Unsupported format: {format}")
            return paths
        return self._run(op)

    # --- serialization helpers unchanged ---
    def _to_serializable_dict(self, cartridge: BeliefCartridgeORM) -> Dict[str, Any]:
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
            "evaluations": [e.id for e in cartridge.evaluations],
        }
        if cartridge.evaluations:
            scores = {}
            for eval in cartridge.evaluations:
                for dim, score in eval.scores.items():
                    scores[dim] = round(float(score), 2)
            data["scores"] = scores
        return data

    def _default_serializer(self, obj):
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
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {k: self._deserialize_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._deserialize_json(v) for v in value]
        return str(value)
