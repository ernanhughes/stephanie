from sqlalchemy import Column, DateTime, ForeignKey, Integer, func

from co_ai.models.base import Base


class EvaluationRuleLinkORM(Base):
    __tablename__ = "evaluation_rule_links"

    id = Column(Integer, primary_key=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False)
    rule_application_id = Column(Integer, ForeignKey("rule_applications.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
