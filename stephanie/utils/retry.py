# stephanie/utils/retry.py
from __future__ import annotations

import functools
import random
import time
from typing import Callable, TypeVar

T = TypeVar('T')

def retry_with_backoff(max_retries: int = 3, backoff_in_seconds: float = 1.0):
    """
    Decorator to retry a function with exponential backoff.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            retries = 0
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if retries == max_retries:
                        raise
                    retries += 1
                    sleep_time = backoff_in_seconds * (2 ** (retries - 1)) + random.uniform(0, 1)
                    time.sleep(sleep_time)
            return None  # unreachable
        return wrapper
    return decorator