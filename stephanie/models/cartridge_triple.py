# stephanie/models/cartridge_triple.py

from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class CartridgeTripleORM(Base):
    __tablename__ = "cartridge_triples"

    id = Column(Integer, primary_key=True)
    cartridge_id = Column(
        Integer, ForeignKey("cartridges.id", ondelete="CASCADE"), nullable=False
    )

    subject = Column(String, nullable=False)
    predicate = Column(String, nullable=False)
    object = Column(String, nullable=False)
    confidence = Column(
        Float, default=1.0
    )  # Optional: confidence score from the extractor

    # Relationship to the parent cartridge
    cartridge = relationship("CartridgeORM", back_populates="triples_rel")

    def to_dict(self):
        return {
            "id": self.id,
            "cartridge_id": self.cartridge_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
        }

    def __repr__(self):
        return (
            f"<CartridgeTripleORM(id={self.id}, cartridge_id={self.cartridge_id}, "
            f"subject='{self.subject}', predicate='{self.predicate}', object='{self.object}', "
            f"confidence={self.confidence:.2f})>"
        )
