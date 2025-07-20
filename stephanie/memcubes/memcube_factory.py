# stephanie/memcubes/memcube_factory.py
from datetime import datetime

from sqlalchemy.sql import text

from stephanie.memcubes.memcube import MemCube
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import ScorableFactory


class MemCubeFactory:
    @staticmethod
    def from_scorable(
        scorable: Scorable,
        version: str = "v1",
        source: str = "scorable_factory",
        model: str = "default"
    ) -> MemCube:
        """
        Convert Scorable to MemCube with enhanced metadata
        """
        return MemCube(
            scorable=scorable,
            version=version,
            source=source,
            model=model,
            priority=5,  # Default priority
            sensitivity="public",  # Default sensitivity
            extra_data={
                "scorable": scorable.to_dict()
            }
        )

    @staticmethod
    def from_dict(data: dict) -> MemCube:

        scorable = ScorableFactory.from_dict(data.get("scorable", {}))

        return MemCube(
            scorable=scorable,
            version=data.get("version", "v1"),
            created_at=datetime.fromisoformat(data.get("created_at")) if data.get("created_at") else None,
            last_modified=datetime.fromisoformat(data.get("last_modified")) if data.get("last_modified") else None,
            source=data.get("source", "user_input"),
            model=data.get("model", "llama3"),
            priority=data.get("priority", 5),
            original_score=data.get("original_score", None),
            refined_score=data.get("refined_score", None),
            refined_content=data.get("refined_content", None),
            sensitivity=data.get("sensitivity", "public"),
            ttl=data.get("ttl", None),
            usage_count=data.get("usage_count", 0),
            extra_data=data.get("extra_data", {})
        )

    
    def _generate_version(self, scorable: Scorable) -> str:
        """Generate version based on content stability"""
        query = """
        SELECT version FROM memcubes
        WHERE scorable_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """
        result = self.db.execute(text(query), [scorable.id]).fetchone()
        
        if not result:
            return "v1"
        
        # Increment version based on content change
        current_version = result.version
        content_hash = hash(scorable.text)
        
        if content_hash != self._get_content_hash(current_version):
            return f"v{int(current_version[1:]) + 1}"
        
        return current_version  # Same content, return current version