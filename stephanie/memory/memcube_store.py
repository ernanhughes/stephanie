# stephanie/memory/memcube_store.py
import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from stephanie.memcubes.memcube import MemCube
from stephanie.memcubes.memcube_factory import MemCubeFactory


class MemcubeStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "memcube"

    def name(self) -> str:
        return "memcube"
   
    def get_memcube(self, cube_id: str) -> Optional[MemCube]:
        """Get MemCube from persistent storage"""
        query = "SELECT * FROM memcubes WHERE id = :id"
        result = self.db.execute(query, {"id": cube_id}).fetchone()
        
        if not result:
            return None
            
        return MemCubeFactory.from_dict(result)
    
    def save_memcube(self, memcube: MemCube):
        """Save MemCube with versioning and conflict handling"""

        # Check for existing MemCube by ID
        check_query = "SELECT COUNT(*) FROM memcubes WHERE id = :id"
        exists = self.session.execute(text(check_query), {"id": memcube.id}).scalar()

        # Version logic
        if memcube.version == "auto":
            next_version = self._get_next_version(memcube.scorable.id)
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
            "extra_data": json.dumps(memcube.extra_data)
        }

        self.session.execute(text(insert_query), params)
        self.session.commit() 
        if self.logger:
            self.logger.log("MemCubeSaved", {
                "id": memcube.id,
                "version": memcube.version,
                "type": memcube.scorable.target_type
            })

    def get_target_type(self, target_type) -> str:
        if isinstance(target_type, str):
            return target_type.lower()
        elif hasattr(target_type, 'value'):
            return target_type.value.lower()
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
        
    # In DocumentEBTTrainerAgent
    def get_training_data(self, dimension: str) -> list[dict]:
        """Get versioned training data from MemCube"""
        query = """ I
        SELECT m.content, s.score
        FROM memcubes m
        JOIN scoring_events s ON m.id = s.memcube_id
        WHERE m.scorable_type = 'document'
        AND s.dimension = :dim
        AND m.version = :version
        """
        return self.db.execute(text(query), {
            "dim": dimension,
            "version": self.model_version
        }).fetchall()
    
    # In DocumentEBTTrainerAgent
    def get_versioned_training_data(self, dimension: str, version: str = "latest"):
        query = """
        SELECT * FROM memcubes
        WHERE scorable_type = :type
        AND sensitivity = :sensitivity
        AND version = :version
        """
        results = self.db.execute(text(query), {
            "type": "document",
            "sensitivity": "public",
            "version": version
        }).fetchall()
        
        return [{
            "title": "Goal Context",
            "output_a": r.content,
            "output_b": r.refined_content,
            "value_a": r.original_score,
            "value_b": r.refined_score,
            "dimension": dimension
        } for r in results]
    

    def _get_next_version(self, scorable_id: int) -> int:
        query = """
            SELECT version FROM memcubes
            WHERE scorable_id = :scorable_id
            ORDER BY created_at DESC
            LIMIT 1
        """
        result = self.session.execute(text(query), {"scorable_id": scorable_id}).fetchone()

        if result and result.version:
            version_str = result.version.strip().lower()
            if version_str.startswith("v") and version_str[1:].isdigit():
                next_version = int(version_str[1:]) + 1
            elif version_str.isdigit():
                next_version = int(version_str) + 1
            else:
                next_version = 1
        else:
            next_version = 1

        return next_version
