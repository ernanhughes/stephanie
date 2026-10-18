# stephanie/utils/db_retry.py
from __future__ import annotations

from sqlalchemy.exc import OperationalError


def with_db_retry(fn, tries: int = 2, engine=None):
    def wrapper(*args, **kwargs):
        last = None
        for i in range(tries):
            try:
                return fn(*args, **kwargs)
            except OperationalError as e:
                last = e
                bind = engine
                if bind is None:
                    for value in (*args, *kwargs.values()):
                        if hasattr(value, "get_bind"):
                            bind = value.get_bind()
                            break
                if bind is not None:
                    bind.dispose()  # drop connections from the operation's pool
        raise last
    return wrapper
