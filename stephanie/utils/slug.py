# stephanie/utils/slug.py
import re


def simple_slugify(text: str) -> str:
    # Lowercase the text
    text = text.lower()
    # Replace non-alphanumeric characters with hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text)
    # Strip hyphens from the start and end
    text = text.strip("-")
    return text


def slugify_with_max_length(text: str, max_length: int = 50) -> str:
    """
    Slugify the text and ensure it does not exceed max_length.
    If it does, truncate and append a hash.
    """
    slug = simple_slugify(text)
    if len(slug) > max_length:
        # Truncate to max_length - 8 for the hash
        truncated_slug = slug[: max_length - 8]
        # Append a hash of the original text
        hash_part = re.sub(r"[^a-z0-9]", "", text)[:8]
        return f"{truncated_slug}-{hash_part}"
    return slug
