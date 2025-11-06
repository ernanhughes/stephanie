# stephanie/utils/llm_text_parsers.py
from __future__ import annotations

import re
from typing import Any, Dict, List

_SCORE_PATTERNS = [
    r"score\s*[:=\[]\s*(?P<num>\d{1,3}(?:\.\d+)?)\s*%?\]?",
    r'"score"\s*:\s*(?P<num>\d{1,3}(?:\.\d+)?)',
]

def _extract_bullets(text: str) -> list[str]:
    bullets = []
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^[-*•]\s+", line):
            bullets.append(re.sub(r"^[-*•]\s+", "", line))
    return bullets

def parse_scored_block(raw: str) -> Dict[str, Any]:
    text = (raw or "").replace("\r\n", "\n").strip()

    # score → 0..1
    score = None
    for p in _SCORE_PATTERNS:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                val = float(m.group("num"))
                score = val / 100.0 if val > 1.0 else val
            except Exception:
                pass
            break

    # one-line rationale (don’t swallow following sections)
    rm = re.search(r"(?im)^\s*rationale\s*:\s*(.+)$", text)
    rationale = rm.group(1).strip() if rm else None

    # keep headings + bullets; only drop the Score line, and strip the "Rationale:" label
    cleaned_lines: List[str] = []
    for ln in text.splitlines():
        if re.match(r"(?i)^\s*score\s*[:=\[].*$", ln):
            continue  # drop
        if re.match(r"(?i)^\s*rationale\s*:\s*", ln):
            ln = re.sub(r"(?i)^\s*rationale\s*:\s*", "", ln).strip()
            if not ln:
                # if the line is only the label, skip; the rest of the doc remains
                continue
        cleaned_lines.append(ln)

    cleaned = "\n".join(cleaned_lines).strip()

    # hard fallback: never return empty content
    if not cleaned:
        cleaned = text

    return {
        "content": cleaned,
        "llm_score": score,
        "rationale": rationale,
        "bullets": _extract_bullets(text),
        "raw": raw,
    }
