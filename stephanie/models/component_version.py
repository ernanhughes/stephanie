# stephanie/registry/models.py

from sqlalchemy import (JSON, Boolean, Column, DateTime, ForeignKey, Integer,
                        String, func)
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class ComponentVersionORM(Base):
    __tablename__ = 'component_versions'

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    protocol = Column(String, nullable=False)
    class_path = Column(String, nullable=False)
    version = Column(String, nullable=False)
    performance = Column(JSON, nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    last_used = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0)
    metadata = Column(JSON, nullable=True)

    interfaces = relationship("ComponentInterfaceORM", back_populates="component")


class ComponentInterfaceORM(Base):
    __tablename__ = 'component_interfaces'

    id = Column(Integer, primary_key=True, autoincrement=True)
    component_id = Column(String, ForeignKey('component_versions.id'))
    protocol = Column(String, nullable=False)
    implemented = Column(Boolean, default=True)
    last_checked = Column(DateTime, default=func.now())

    component = relationship("ComponentVersionORM", back_populates="interfaces")
