# stephanie/utils/llm_utils.py
from __future__ import annotations

import re


def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

