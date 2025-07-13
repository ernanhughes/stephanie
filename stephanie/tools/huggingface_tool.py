# stephanie/tools/huggingface_tool.py
import hashlib
import pickle
import re
from pathlib import Path

from gradio_client import Client
from huggingface_hub import HfApi

CACHE_DIR = Path(".paper_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _get_cache_path(paper_url: str) -> Path:
    # Create hash from URL to use as filename
    key = hashlib.md5(paper_url.encode()).hexdigest()
    return CACHE_DIR / f"{key}.pkl"


def recommend_similar_papers(
    paper_url: str = "https://arxiv.org/pdf/2505.08827",
) -> list[dict]:
    cache_path = _get_cache_path(paper_url)

    # Return from cache if exists
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Otherwise run the real request
    try:
        client = Client("librarian-bots/recommend_similar_papers")
        result = client.predict(paper_url, None, False, api_name="/predict")
        paper_ids = re.findall(r"https://huggingface\.co/papers/(\d+\.\d+)", result)

        hits = [
            {
                "query": paper_url,
                "source": "recommend_similar_papers",
                "result_type": "url",
                "url": f"https://arxiv.org/pdf/{pid}.pdf",
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
        print(f"Failed to get similar papers: {e}")
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
