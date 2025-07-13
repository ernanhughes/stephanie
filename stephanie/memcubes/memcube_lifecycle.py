# stephanie/memcubes/memcube_lifecycle.py

import json

from stephanie.memcubes.memcube import MemCube


class MemCubeLifecycle:
    def __init__(self, db, logger):
        self.db = db
        self.logger = logger
        
    def archive_old_cubes(self, days: int = 30):
        """Archive cubes older than N days"""
        query = """
        SELECT * FROM memcubes
        WHERE created_at < NOW() - INTERVAL '%s days'
          AND sensitivity != 'permanent'
        """
        results = self.db.execute(query, [days]).fetchall()
        
        for cube in results:
            self._archive(cube)
    
    def _archive(self, cube: MemCube):
        """Move cube to long-term storage"""
        archive_path = f"archive/{cube.scorable.target_type}/{cube.id}.json"
        with open(archive_path, "w") as f:
            json.dump(cube.to_dict(), f)
            
        # Remove from active store
        self.db.execute("DELETE FROM memcubes WHERE id = %s", [cube.id])
        self.logger.log("MemCubeArchived", {
            "cube_id": cube.id,
            "path": archive_path
        })