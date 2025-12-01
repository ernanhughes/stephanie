# stephanie/utils/async_utils.py
from __future__ import annotations

import asyncio
import logging

from typing import Any, Dict, Optional
from stephanie.types.model import ModelSpec
from stephanie.utils.llm_utils import remove_think_blocks
import litellm

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


async def acomplete_messages(
    *,
    model: ModelSpec,
    messages: list[dict],
    params: Optional[Dict[str, Any]] = None,
) -> str:
    call_params = dict(model.params or {})
    if params:
        call_params.update(params)

    resp = await litellm.acompletion(
        model=model.name,
        messages=messages,
        api_base=(model.api_base or "http://localhost:11434"),
        api_key=(model.api_key or ""),
        **call_params,
    )
    out = resp["choices"][0]["message"]["content"]
    return remove_think_blocks(out or "")

async def acomplete(
    *,
    prompt: str,
    model: ModelSpec,
    sys_preamble: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    messages = []
    if sys_preamble:
        messages.append({"role": "system", "content": sys_preamble})
    messages.append({"role": "user", "content": prompt})
    return await acomplete_messages(model=model, messages=messages, params=params)
