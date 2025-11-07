# stephanie/utils/parser_utils.py
from __future__ import annotations

import re
from typing import Dict, List


def extract_hypotheses(text: str):
    # First attempt: Try precise regex-based extraction
    pattern = re.compile(
        r"(# Hypothesis\s+\d+\s*\n.*?)(?:\n(?=# Hypothesis\s+\d+)|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    matches = list(pattern.finditer(text))

    if matches:
        return [match.group(1).strip() for match in matches]

    # Fallback (if needed)
    split_parts = re.split(r"\bHypothesis\s+\d+\b", text, flags=re.IGNORECASE)
    if len(split_parts) <= 1:
        return [text]

    hypotheses = []
    for i, part in enumerate(split_parts[1:], start=1):
        cleaned = part.strip()
        if cleaned:
            hypotheses.append(f"Hypothesis {i} {cleaned}")

    return hypotheses



def extract_hypotheses_with_score(text: str) -> List[Dict]:
    """
    Extract hypotheses with rationale and score from model output.

    Expected format:
        rationale: <brief explanation>
        score: <0–100>

        # Hypothesis N
        <hypothesis text>

    Returns:
        List of dicts: [{"rationale": str, "score": float, "text": str}, ...]
    """
    hypotheses = []

    # Split into sections by Hypothesis headers
    blocks = re.split(r"#\s*Hypothesis\s*\d+", text, flags=re.IGNORECASE)
    headers = re.findall(r"#\s*Hypothesis\s*\d+", text, flags=re.IGNORECASE)

    for i, block in enumerate(blocks[1:]):  # skip text before first hypothesis
        header = headers[i] if i < len(headers) else f"Hypothesis {i+1}"

        # Extract rationale
        rationale_match = re.search(r"rationale\s*:\s*(.+)", block, re.IGNORECASE)
        rationale = rationale_match.group(1).strip() if rationale_match else ""

        # Extract score (coerce to float, fallback 0)
        score_match = re.search(r"score\s*:\s*([\d\.]+)", block, re.IGNORECASE)
        try:
            score = float(score_match.group(1)) if score_match else 0.0
        except ValueError:
            score = 0.0

        # Extract main hypothesis text (strip rationale/score lines)
        cleaned = re.sub(r"rationale\s*:.+", "", block, flags=re.IGNORECASE)
        cleaned = re.sub(r"score\s*:.+", "", cleaned, flags=re.IGNORECASE)
        hypothesis_text = cleaned.strip()

        hypotheses.append({
            "rationale": rationale,
            "score": score,
            "text": hypothesis_text,
            "header": header.strip()
        })

    return hypotheses
