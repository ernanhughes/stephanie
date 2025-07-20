# stephanie/utils/parser_utils.py
import re


def extract_hypotheses(text: str):
    # First attempt: Try precise regex-based extraction
    pattern = re.compile(
        r"(# Hypothesis\s+\d+\s*\n(?:.*?))(?:\n(?=# Hypothesis\s+\d+)|\Z)",
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
