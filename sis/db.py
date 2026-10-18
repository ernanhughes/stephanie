# sis/db.py
from sqlalchemy.orm import sessionmaker
from stephanie.orm.base import Base
from stephanie.orm.base import create_database_engine

def init_db(db_config, *, create_schema: bool = False):
    """Create a configured session factory without implicit schema changes.

    ``create_schema`` is an explicit local/test opt-in only. It is not a
    migration mechanism and must not be enabled as a production upgrade path.
    """
    engine = create_database_engine(db_config)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    if create_schema:
        Base.metadata.create_all(bind=engine)
    return SessionLocal
