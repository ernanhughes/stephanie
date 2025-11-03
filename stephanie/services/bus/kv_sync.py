# stephanie/services/bus/kv_sync.py
from __future__ import annotations

import asyncio
import threading
from typing import Optional


class SyncKV:
    def __init__(self, js, bucket, max_age_seconds=None, description=None):
        self._loop = asyncio.new_event_loop()
        self._t = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._t.start()
        self._js = js
        self._bucket = bucket
        self._kv = self._run(self._ensure(bucket, max_age_seconds, description))

    async def _ensure(self, bucket, max_age_seconds, description):
        try:
            return await self._js.key_value(bucket=bucket)
        except:
            return await self._js.create_key_value(bucket=bucket, description=description, max_age=max_age_seconds)

    def _run(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def get(self, key: str) -> Optional[bytes]:
        async def _get():
            try:
                e = await self._kv.get(key)
                return None if e is None else e.value
            except:
                return None
        return self._run(_get())

    def put(self, key: str, value: bytes) -> None:
        async def _put():
            try:
                await self._kv.put(key, value)
            except:
                pass
        self._run(_put())
