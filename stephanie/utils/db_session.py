# stephanie/utils/db_session.py
from __future__ import annotations
from sqlalchemy.orm import Session
from contextlib import contextmanager

@contextmanager
def db_session(SessionLocal: Session) -> Session:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
