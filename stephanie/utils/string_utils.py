# stephanie/utils/string_utils.py
from __future__ import annotations

import re


def trunc(s: str | None, n: int = 200) -> str | None:
    if not isinstance(s, str):
        return s
    return s if len(s) <= n else s[:n] + "…"


def normalize_key(name: str) -> str:
    """
    Normalize a string into a compact, lowercase key.

    - Lowercases
    - Strips leading/trailing whitespace
    - Removes all non [a-z0-9] characters

    Useful for building stable dictionary keys or lookup maps.
    """
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]", "", name.lower().strip())


def clean_heading(heading: str) -> str:
    """
    Clean a heading/title string by:

    - Stripping leading section numbers like '1.2.3 ' or '3 '
    - Removing 'Section/Chapter/Part N' prefixes
    - Dropping non-word punctuation
    - Collapsing multiple spaces

    This mirrors the logic previously embedded in DocumentSectionParser.
    """
    if not heading:
        return ""
    # Remove leading numbering like "1.2.3 " or "2 "
    heading = re.sub(r"^\s*[\d\.\s]+\s*", " ", heading)
    # Remove 'section/chapter/part <whatever>' prefixes
    heading = re.sub(
        r"^(section|chapter|part)\s+\w+",
        "",
        heading,
        flags=re.IGNORECASE,
    )
    # Remove non-word punctuation
    heading = re.sub(r"[^\w\s]", "", heading)
    # Normalize whitespace
    heading = re.sub(r"\s+", " ", heading).strip()
    return heading
