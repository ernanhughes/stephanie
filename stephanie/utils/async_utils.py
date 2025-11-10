# stephanie/utils/async_utils.py
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)

async def retry(coro_factory, attempts=2, base_delay=0.25):
    """
    Minimal retry helper: run a zero-arg coro factory up to `attempts` times.
    Backoff: base_delay, base_delay*2, ...
    """
    delay = float(base_delay)
    for i in range(attempts):
        try:
            return await coro_factory()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if i == attempts - 1:
                raise
            log.warning("retry: retrying after error: %s", e)
            await asyncio.sleep(delay)
            delay *= 2.0