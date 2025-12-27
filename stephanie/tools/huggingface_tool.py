# stephanie/tools/huggingface_tool.py
from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

from gradio_client import Client
from huggingface_hub import HfApi

from stephanie.utils.hash_utils import hash_text

log = logging.getLogger(__name__)

CACHE_DIR = Path(".paper_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _get_cache_path(paper_url: str) -> Path:
    # Create hash from URL to use as filename
    key = hash_text(paper_url)
    return CACHE_DIR / f"{key}.pkl"

def recommend_similar_by_id(arxiv_id: str) -> list[dict]:
    """
    Convenience wrapper: take a bare arxiv_id like '2505.08827',
    build the PDF URL, and call recommend_similar_papers.
    """
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return recommend_similar_papers(pdf_url)

def recommend_similar_papers(
    paper_url: str = "https://arxiv.org/pdf/2505.08827",
) -> list[dict]:
    cache_path = _get_cache_path(paper_url)

    # Return from cache if exists
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Otherwise run the real request I
    try:
        client = Client("librarian-bots/recommend_similar_papers")
        result = client.predict(paper_url, None, False, api_name="/predict")
        paper_ids = re.findall(r"https://huggingface\.co/papers/(\d+\.\d+)", result)

        hits = [
            {
                "query": paper_url,
                "source": "recommend_similar_papers",
                "result_type": "url",
                "url": f"https://arxiv.org/pdf/{pid}",
                "title": pid,
                "summary": "Not yet processed",
            }
            for pid in paper_ids
        ]

        # Save to cache
        with open(cache_path, "wb") as f:
            pickle.dump(hits, f)

        return hits

    except Exception as e:
        log.error(f"Failed to get similar papers: {e}")
        return []


def search_huggingface_datasets(queries: list[str], max_results: int = 5) -> list[dict]:
    api = HfApi()
    results = []

    for query in queries:
        try:
            matches = api.list_datasets(search=query, limit=max_results)
            for ds in matches:
                results.append(
                    {
                        "name": ds.id,
                        "description": ds.cardData.get(
                            "description", "No description available"
                        )
                        if ds.cardData
                        else "No card data",
                    }
                )
        except Exception as e:
            results.append({"name": query, "description": f"Error searching: {str(e)}"})

    return results



def discover_similar_spaces(
    query: str = "papers",
    author: Optional[str] = None,
    limit: int = 20,
) -> List[Dict]:
    """
    Discover Hugging Face Spaces that look like 'tools'
    similar to recommend_similar_papers.

    query  – free-text search over Space name/description
    author – restrict to a particular org/user (e.g. 'librarian-bots')
    limit  – max number of spaces to return
    """
    api = HfApi()
    spaces = api.list_spaces(
        search=query,
        author=author,
        sort="likes",
        direction=-1,
        limit=limit,
    )

    results = []
    for s in spaces:
        card = getattr(s, "cardData", None) or {}
        results.append(
            {
                "id": s.id,                          # e.g. "librarian-bots/recommend_similar_papers"
                "author": s.author,                  # org or user
                "likes": getattr(s, "likes", None),
                "title": card.get("title"),
                "tags": card.get("tags", []),
                "sdk": card.get("sdk"),              # "gradio", "streamlit", etc.
                "raw": s,                            # original SpaceInfo if you need more
            }
        )
    return results
