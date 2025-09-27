# stephanie/utils/db_scope.py
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy.exc import OperationalError

from stephanie.models.base import engine


@contextmanager
def session_scope(session_maker):
    s = session_maker()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

def retry(op, tries=2):
    last = None
    for _ in range(tries):
        try:
            return op()
        except OperationalError as e:
            last = e
            engine.dispose()  # drop dead pooled conns
    raise last
