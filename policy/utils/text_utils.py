import re


def clean_wiki_markup(text: str) -> str:
    # [[Page|Display]] → Display
    text = re.sub(r"\[\[[^\|\]]+\|([^\]]+)\]\]", r"\1", text)

    # [[Page]] → Page
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # Replace underscores with spaces
    text = text.replace("_", " ")

    return text

def split_into_sentences(text: str):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences

def split_into_paragraphs(text: str):
    """
    Split document into paragraph-level chunks.
    Handles both double-newline and fallback chunking.
    """
    # First try real paragraph breaks
    paras = re.split(r"\n\s*\n", text)

    # If document has no paragraph breaks (CNN style), fallback to length chunking
    if len(paras) <= 1:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunk_size = 5  # 5 sentences per pseudo-paragraph
        paras = [
            " ".join(sentences[i:i+chunk_size])
            for i in range(0, len(sentences), chunk_size)
        ]

    # Clean empty
    paras = [p.strip() for p in paras if p.strip()]

    return paras
