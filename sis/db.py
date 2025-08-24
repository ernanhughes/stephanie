# sis/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from stephanie.models.base import Base

# You’ll load this from your config.yaml
def init_db(db_url: str):
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    return SessionLocal
