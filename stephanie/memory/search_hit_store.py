# stephanie/memory/search_hit_store.py
# memory/search_hit_store.py

from sqlalchemy.orm import Session

from stephanie.models.search_hit import SearchHitORM


class SearchHitStore:
    def __init__(self, db: Session):
        self.db = db

    def add_hit(self, hit_data: dict) -> SearchHitORM:
        hit = SearchHitORM(**hit_data)
        self.db.add(hit)
        self.db.commit()
        self.db.refresh(hit)
        return hit

    def bulk_add_hits(self, hit_data_list: list[dict]) -> list[SearchHitORM]:
        hits = [SearchHitORM(**data) for data in hit_data_list]
        self.db.bulk_save_objects(hits)
        self.db.commit()
        return hits

    def get_by_id(self, hit_id: int) -> SearchHitORM:
        return self.db.query(SearchHitORM).get(hit_id)

    def get_all_for_goal(self, goal_id: int) -> list[SearchHitORM]:
        return self.db.query(SearchHitORM).filter_by(goal_id=goal_id).all()

    def delete_by_id(self, hit_id: int) -> bool:
        hit = self.db.query(SearchHitORM).get(hit_id)
        if hit:
            self.db.delete(hit)
            self.db.commit()
            return True
        return False
