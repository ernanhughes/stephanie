# stephanie/models/vpm.py
from __future__ import annotations

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

from stephanie.models.base import Base


class VPMORM(Base):
    """
    Visual Policy Map row snapshot (1 row = 1 timestep of metric vector).
    Stores metric order, values, and optional rendered artifacts from ZeroModel.
    """
    __tablename__ = "vpms"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False)              # timeline/group id
    step = Column(Integer, nullable=False, default=0)    # row index in the timeline
    metric_names = Column(JSON, nullable=False)          # list[str] — preserved order
    values = Column(JSON, nullable=False)                # list[float] — raw or normalized

    # Optional artifacts produced by ZeroModelService
    img_png = Column(Text, nullable=True)                # static heatmap for this run or step
    img_gif = Column(Text, nullable=True)                # animated timeline (usually per run)
    summary_json = Column(Text, nullable=True)           # path to meta JSON

    extra = Column(JSON, nullable=True)                  # any additional metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step": self.step,
            "metric_names": self.metric_names or [],
            "values": self.values or [],
            "img_png": self.img_png,
            "img_gif": self.img_gif,
            "summary_json": self.summary_json,
            "extra": self.extra or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<VPM[{self.run_id} step={self.step} cols={len(self.metric_names or [])}]>"
