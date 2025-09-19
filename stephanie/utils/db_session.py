# stephanie/utils/db_session.py
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy.orm import Session


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
