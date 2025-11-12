# stephanie/models/cartridge_domain.py
from __future__ import annotations

from sqlalchemy import Column, Float, ForeignKey, Integer, String

from stephanie.models.base import Base


class CartridgeDomainORM(Base):
    __tablename__ = "cartridge_domains"

    id = Column(Integer, primary_key=True)
    cartridge_id = Column(
        Integer, ForeignKey("cartridges.id", ondelete="CASCADE"), nullable=False
    )
    domain = Column(String, nullable=False)
    score = Column(Float, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "cartridge_id": self.cartridge_id,
            "domain": self.domain,
            "score": self.score,
        }
