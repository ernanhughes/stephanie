# stephanie/memory/memcube_store.py
from __future__ import annotations

import json
from typing import Optional

from sqlalchemy import text

from stephanie.memcube.memcube import MemCube
from stephanie.memcube.memcube_factory import MemCubeFactory
from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.memcube import MemCubeORM


class MemcubeStore(BaseSQLAlchemyStore):
    orm_model = MemCubeORM
    default_order_by = MemCubeORM.created_at.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "memcubes"

    # --------------------
    # GET / SAVE
    # --------------------
    def get_memcube(self, cube_id: str) -> Optional[MemCube]:
        """Get MemCube from persistent storage"""
        def op(s):
            result = s.execute(text("SELECT * FROM memcubes WHERE id = :id"), {"id": cube_id}).fetchone()
            if not result:
                return None
            return MemCubeFactory.from_dict(dict(result))
        return self._run(op)

    def save_memcube(self, memcube: MemCube):
        """Save MemCube with versioning and conflict handling"""
        def op(s):
            # Check for existing MemCube
            check_query = "SELECT COUNT(*) FROM memcubes WHERE id = :id"
            exists = s.execute(text(check_query), {"id": memcube.id}).scalar()

            # Versioning logic
            if memcube.version == "auto":
                next_version = self._get_next_version(memcube.scorable.id, s)
                memcube.version = f"v{next_version}"
            else:
                version_str = memcube.version or "v1"
                if version_str.startswith("v") and version_str[1:].isdigit():
                    current_version_num = int(version_str[1:])
                    if exists:
                        memcube.version = f"v{current_version_num + 1}"
                else:
                    memcube.version = "v1"

            # Regenerate ID with version suffix
            base_id = memcube.id.rsplit("_", 1)[0]
            memcube.id = f"{base_id}_{memcube.version}"

            insert_query = """
            INSERT INTO memcubes (
                id, scorable_id, scorable_type, content,
                version, source, model, priority, 
                sensitivity, ttl, usage_count, extra_data,
                created_at, last_modified
            ) VALUES (
                :id, :scorable_id, :scorable_type, :content,
                :version, :source, :model, :priority,
                :sensitivity, :ttl, :usage_count, :extra_data,
                NOW(), NOW()
            )
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                version = EXCLUDED.version,
                source = EXCLUDED.source,
                model = EXCLUDED.model,
                priority = EXCLUDED.priority,
                sensitivity = EXCLUDED.sensitivity,
                ttl = EXCLUDED.ttl,
                usage_count = EXCLUDED.usage_count,
                extra_data = EXCLUDED.extra_data,
                last_modified = NOW()
            """

            params = {
                "id": memcube.id,
                "scorable_id": memcube.scorable.id,
                "scorable_type": self.get_target_type(memcube.scorable.target_type),
                "content": memcube.scorable.text,
                "version": memcube.version,
                "source": memcube.source,
                "model": memcube.model,
                "priority": memcube.priority,
                "sensitivity": memcube.sensitivity,
                "ttl": memcube.ttl,
                "usage_count": memcube.usage_count,
                "extra_data": json.dumps(memcube.extra_data),
            }

            s.execute(text(insert_query), params)

            if self.logger:
                self.logger.log("MemCubeSaved", {
                    "id": memcube.id,
                    "version": memcube.version,
                    "type": memcube.scorable.target_type,
                })
            return memcube.id
        return self._run(op)

    # --------------------
    # UTILS
    # --------------------
    def get_target_type(self, target_type) -> str:
        if isinstance(target_type, str):
            return target_type.lower()
        raise ValueError(f"Unsupported target type: {target_type}")

    # --------------------
    # TRAINING DATA
    # --------------------
    def get_training_data(self, dimension: str, model_version: str = "latest") -> list[dict]:
        """Get versioned training data from MemCube"""
        def op(s):
            query = """
            SELECT m.content, s.score
            FROM memcubes m
            JOIN scoring_events s ON m.id = s.memcube_id
            WHERE m.scorable_type = 'document'
              AND s.dimension = :dim
              AND m.version = :version
            """
            rows = s.execute(text(query), {"dim": dimension, "version": model_version}).fetchall()
            return [dict(r) for r in rows]
        return self._run(op)

    def get_versioned_training_data(self, dimension: str, version: str = "latest"):
        def op(s):
            query = """
            SELECT *
            FROM memcubes
            WHERE scorable_type = :type
              AND sensitivity = :sensitivity
              AND version = :version
            """
            rows = s.execute(text(query), {
                "type": "document",
                "sensitivity": "public",
                "version": version,
            }).fetchall()

            return [{
                "title": "Goal Context",
                "output_a": r.content,
                "output_b": getattr(r, "refined_content", None),
                "value_a": getattr(r, "original_score", None),
                "value_b": getattr(r, "refined_score", None),
                "dimension": dimension,
            } for r in rows]
        return self._run(op)

    def _get_next_version(self, scorable_id: int, session) -> int:
        query = """
            SELECT version
            FROM memcubes
            WHERE scorable_id = :scorable_id
            ORDER BY created_at DESC
            LIMIT 1
        """
        result = session.execute(text(query), {"scorable_id": scorable_id}).fetchone()

        if result and result.version:
            version_str = result.version.strip().lower()
            if version_str.startswith("v") and version_str[1:].isdigit():
                return int(version_str[1:]) + 1
            if version_str.isdigit():
                return int(version_str) + 1
        return 1


    async def query_calibration(self, kind: str, filters: dict, sort=None, limit: int = 1):
        """
        Async-style query for calibration records (e.g., domain thresholds).
        Compatible with DomainCalibratedRiskPredictor.

        Expected schema:
            memcubes.extra_data -> JSON with keys like "low_threshold", "high_threshold"
            where memcube.source == kind (e.g. "risk")

        Args:
            kind: Calibration category, e.g. "risk".
            filters: Dict with at least {"domain": <domain_name>}
            sort: Optional sort criteria, defaults to created_at DESC.
            limit: Number of records to return, defaults to 1.

        Returns:
            dict | None — calibration record or None if not found.
        """
        domain = filters.get("domain", "general")

        def op(s):
            query = """
                SELECT extra_data
                FROM memcubes
                WHERE source = :kind
                  AND (extra_data->>'domain') = :domain
                ORDER BY created_at DESC
                LIMIT :limit
            """
            row = s.execute(
                text(query),
                {"kind": kind, "domain": domain, "limit": limit}
            ).fetchone()

            if not row:
                if self.logger:
                    self.logger.log("NoCalibration", {"record": f"{kind}:{domain}"})
                return None

            try:
                data = json.loads(row.extra_data) if isinstance(row.extra_data, str) else dict(row.extra_data)
                if self.logger:
                    self.logger.info(f"Loaded calibration for {domain}: {data}")
                return data
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to parse calibration data for {domain}: {e}")
                return None

        # run in sync context and emulate async yield
        result = self._run(op)
        return result
