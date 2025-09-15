from datetime import datetime

from sqlalchemy import (JSON, Column, DateTime, ForeignKey, Integer, String,
                        Text)

from stephanie.models.base import Base


class DynamicScorableORM(Base):
    __tablename__ = "dynamic_scorables"

    id = Column(Integer, primary_key=True)
    pipeline_run_id = Column(String, nullable=False)           # pipeline run that generated it
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=True)
    scorable_type = Column(String, nullable=False)    # e.g. "draft", "goal_score", "trace"
    source = Column(String, nullable=True)            # e.g. "draft_generator", "policy_eval"

    source_scorable_id   = Column(Integer, nullable=True)
    source_scorable_type = Column(String, nullable=True)


    text = Column(Text, nullable=True)                # optional, can be large
    meta = Column(JSON, nullable=True)
    role = Column(String, nullable=True)              # e.g. "user", "assistant", "system"
    created_at = Column(DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "pipeline_run_id": self.pipeline_run_id,
            "case_id": self.case_id,
            "scorable_type": self.scorable_type,
            "source": self.source,
            "text": self.text,
            "meta": self.meta or {},
            "created_at": self.created_at.isoformat()
        }
