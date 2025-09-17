# stephanie/knowledge/simple_ner.py
from __future__ import annotations

import re


def extract_entities(text: str) -> dict:
    # super light: upper-case tokens + paper-ish keys; replace with your NER later
    ABBR = re.findall(r"\b([A-Z]{2,})\b", text)
    MATH = re.findall(r"\b(L2|L1|softmax|attention|grad)\b", text, flags=re.I)
    return {"ABBR": list(set(ABBR)), "TERMS": list(set([m.lower() for m in MATH]))}
