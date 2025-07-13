# stephanie/worldview/db/locator.py

import hashlib
import os

from sqlalchemy import create_engine

from stephanie.models.base import WorldviewBase
from stephanie.models.worldview import WorldviewORM
from stephanie.utils.slug import slugify_with_max_length


class WorldviewDBLocator:
    def __init__(self, base_dir: str = "worldviews"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def generate_worldview_id(self, goal_text: str) -> str:
        slug = slugify_with_max_length(goal_text, 60)
        hash_part = hashlib.sha1(goal_text.encode()).hexdigest()[:8]
        return f"{slug}-{hash_part}"

    def get_db_path(self, goal_text: str) -> str:
        name = self.generate_worldview_id(goal_text)
        return os.path.join(self.base_dir, f"{name}.db")

    def exists(self, goal_text: str) -> bool:
        return os.path.exists(self.get_db_path(goal_text))

    def delete(self, goal_text: str):
        path = self.get_db_path(goal_text)
        if os.path.exists(path):
            os.remove(path)

    def list_all(self) -> list[str]:
        return [f for f in os.listdir(self.base_dir) if f.endswith(".db")]

    def register_worldview(self, session, goal_text: str, db_path: str):
        name = self.generate_worldview_id(goal_text)
        existing = session.query(WorldviewORM).filter_by(name=name).first()
        description = f"Worldview for goal: {goal_text}"
        if not existing:
            worldview = WorldviewORM(
                name=name,
                goal=goal_text,
                db_path=db_path,
                description=description,
            )
            session.add(worldview)
            session.commit()

    def create_worldview(self, goal_text: str, session) -> str:
        """
        Creates and registers a new worldview SQLite database and
        adds metadata to main registry.
        Returns path to SQLite DB.
        """
        db_path = self.get_db_path(goal_text)
        if not os.path.exists(db_path):
            engine = create_engine(f"sqlite:///{db_path}")
            WorldviewBase.metadata.create_all(engine)

        self.register_worldview(session, goal_text, db_path)
        return db_path
