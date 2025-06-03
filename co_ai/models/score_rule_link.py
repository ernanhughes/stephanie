from sqlalchemy import Column, DateTime, ForeignKey, Integer, func
from sqlalchemy.orm import relationship

from co_ai.models.base import Base


class ScoreRuleLinkORM(Base):
    __tablename__ = "score_rule_links"

    id = Column(Integer, primary_key=True)
    score_id = Column(Integer, ForeignKey("scores.id", ondelete="CASCADE"), nullable=False)
    rule_application_id = Column(Integer, ForeignKey("rule_applications.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
