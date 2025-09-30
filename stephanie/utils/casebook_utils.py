# stephanie/utils/casebook_utils.py
from __future__ import annotations


def generate_casebook_name(action_type: str, title: str) -> str:
    return f"{action_type} - {title}"