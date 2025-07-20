# stephanie/memory/belief_cartridge_store.py

from datetime import datetime

from sqlalchemy.orm import Session

from stephanie.models.belief_cartridge import BeliefCartridgeORM


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