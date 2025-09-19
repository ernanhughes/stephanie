# stephanie/utils/db_retry.py
from sqlalchemy.exc import OperationalError

from stephanie.models.base import engine


def with_db_retry(fn, tries: int = 2):
    def wrapper(*args, **kwargs):
        last = None
        for i in range(tries):
            try:
                return fn(*args, **kwargs)
            except OperationalError as e:
                last = e
                engine.dispose()  # drop bad pool conns
        raise last
    return wrapper
