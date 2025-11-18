# stephanie/utils/win_eventloop.py
# Use the selector loop on Windows so pyzmq asyncio works.
from __future__ import annotations

import asyncio
import sys


def ensure_selector_event_loop():
    if sys.platform.startswith("win"):
        # Must be set before any loop is created or zmq.asyncio is imported.
        try:
            asyncio.set_event_loop_policy(
                asyncio.WindowsSelectorEventLoopPolicy()
            )
        except AttributeError:
            # Non-Windows or very old Python – ignore.
            pass
