# stephanie/memory/idea_store.py
# stores/idea_store.py
from stephanie.memory import BaseStore
from stephanie.models.idea import IdeaORM


class IdeaStore(BaseStore):
    def __init__(self, session, logger):
        super().__init__(session, logger)
        self.name = "ideas"

    def name(self) -> str:
        return "ideas"

    def add_idea(self, idea_data: dict) -> IdeaORM:
        """
        Add a single idea to the database.
        """
        idea = IdeaORM(**idea_data)
        self.db.add(idea)
        self.db.commit()
        self.db.refresh(idea)
        return idea

    def bulk_add_ideas(self, ideas_data: list[dict]) -> list[IdeaORM]:
        """
        Add multiple ideas at once.
        """
        ideas = [IdeaORM(**data) for data in ideas_data]
        self.db.bulk_save_objects(ideas)
        self.db.commit()
        return ideas

    def get_by_goal_id(self, goal_id: int) -> list[IdeaORM]:
        """
        Retrieve all ideas associated with a specific goal.
        """
        return self.db.query(IdeaORM).filter(IdeaORM.goal_id == goal_id).all()

    def get_top_ranked_ideas(self, limit: int = 5) -> list[IdeaORM]:
        """
        Get top-ranked ideas based on score or other criteria.
        (This assumes you have a scoring system stored in extra_data or another table)
        """
        # Example: Filter by novelty + feasibility scores from extra_data
        return (
            self.db.query(IdeaORM)
            .order_by(
                IdeaORM.extra_data["novelty_score"].desc(),
                IdeaORM.extra_data["feasibility_score"].desc(),
            )
            .limit(limit)
            .all()
        )

    def get_by_focus_area_and_strategy(
        self, focus_area: str, strategy: str
    ) -> list[IdeaORM]:
        """
        Retrieve ideas filtered by domain and strategy.
        """
        return (
            self.db.query(IdeaORM)
            .filter(IdeaORM.focus_area == focus_area, IdeaORM.strategy == strategy)
            .all()
        )

    def get_by_source(self, source: str) -> list[IdeaORM]:
        """
        Retrieve ideas by their origin (e.g., 'llm', 'survey_agent', 'evolved').
        """
        return self.db.query(IdeaORM).filter(IdeaORM.source == source).all()

    def delete_by_goal_id(self, goal_id: int) -> None:
        """
        Delete all ideas linked to a given goal.
        """
        self.db.query(IdeaORM).filter(IdeaORM.goal_id == goal_id).delete()
        self.db.commit()

    def clear_all(self) -> None:
        """
        Clear all ideas — useful for testing.
        """
        self.db.query(IdeaORM).delete()
        self.db.commit()
