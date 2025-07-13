# stephanie/agents/world/worldview_merger.py
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from stephanie.models.belief import BeliefORM
from stephanie.models.cartridge import CartridgeORM
from stephanie.models.icl_example import ICLExampleORM
from stephanie.models.world_view import WorldviewORM


class WorldviewMergerAgent:
    def __init__(self, db: Session, embedding, logger=None):
        self.db = db
        self.embedding = embedding
        self.logger = logger

    def merge(self, source_ids: list[int], target_id: int) -> int:
        """
        Merge source worldviews into the target worldview (by ID).
        """
        target = self.db.query(WorldviewORM).filter_by(id=target_id).first()
        if not target:
            raise ValueError(f"Target worldview {target_id} not found")

        for src_id in source_ids:
            src = self.db.query(WorldviewORM).filter_by(id=src_id).first()
            if not src:
                continue

            self._merge_beliefs(src_id, target_id)
            self._merge_icl_examples(src_id, target_id)
            self._merge_cartridges(src_id, target_id)

            if self.logger:
                self.logger.log(
                    "WorldviewMerged",
                    {
                        "source": src_id,
                        "target": target_id,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )

        self.db.commit()
        return target_id

    def _merge_beliefs(self, source_id, target_id):
        beliefs = self.db.query(BeliefORM).filter_by(worldview_id=source_id).all()
        target_beliefs = (
            self.db.query(BeliefORM).filter_by(worldview_id=target_id).all()
        )

        existing_summaries = set(b.summary for b in target_beliefs)

        for b in beliefs:
            if b.summary not in existing_summaries:
                merged = BeliefORM(
                    worldview_id=target_id,
                    summary=b.summary,
                    rationale=f"[Merged from worldview {source_id}] {b.rationale or ''}",
                    utility_score=b.utility_score,
                    novelty_score=b.novelty_score,
                    domain=b.domain,
                    status="active",
                    created_at=datetime.utcnow(),
                )
                self.db.add(merged)

    def _merge_icl_examples(self, source_id, target_id):
        examples = self.db.query(ICLExampleORM).filter_by(worldview_id=source_id).all()
        for e in examples:
            new_e = ICLExampleORM(
                worldview_id=target_id,
                prompt=e.prompt,
                response=e.response,
                task_type=e.task_type,
                source=f"Merged from {source_id}",
                created_at=datetime.utcnow(),
            )
            self.db.add(new_e)

    def _merge_cartridges(self, source_id, target_id):
        cartridges = self.db.query(CartridgeORM).filter_by(worldview_id=source_id).all()
        for c in cartridges:
            merged_cart = CartridgeORM(
                worldview_id=target_id,
                goal=c.goal,
                generation=c.generation,
                schema=c.schema,
                created_at=datetime.utcnow(),
                source=f"Merged from worldview {source_id}",
            )
            self.db.add(merged_cart)
