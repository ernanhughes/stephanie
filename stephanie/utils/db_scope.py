# stephanie/utils/db_scope.py
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy.exc import OperationalError


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

def retry(op, tries=2, engine=None):
    last = None
    for _ in range(tries):
        try:
            return op()
        except OperationalError as e:
            last = e
            if engine is not None:
                engine.dispose()  # drop dead connections from this store's pool
    raise last
