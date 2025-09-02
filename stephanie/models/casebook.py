# stephanie/models/casebook.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from stephanie.models.base import Base

class CaseBookORM(Base):
    __tablename__ = "casebooks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    cases = relationship("CaseORM", back_populates="casebook", cascade="all, delete-orphan")

class CaseORM(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    casebook_id = Column(Integer, ForeignKey("casebooks.id"), nullable=False)
    goal_id = Column(String, nullable=False)
    goal_text = Column(Text, nullable=False)
    agent_name = Column(String, nullable=False)
    mars_summary = Column(JSON, nullable=True)
    scores = Column(JSON, nullable=True)
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    casebook = relationship("CaseBookORM", back_populates="cases")
    scorables = relationship("CaseScorableORM", back_populates="case", cascade="all, delete-orphan")

class CaseScorableORM(Base):
    __tablename__ = "case_scorables"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    scorable_id = Column(String, nullable=False)
    scorable_type = Column(String, nullable=True)
    role = Column(String, nullable=True)  # input/output/supporting
    created_at = Column(DateTime, default=datetime.utcnow)

    case = relationship("CaseORM", back_populates="scorables")
