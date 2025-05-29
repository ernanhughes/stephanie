from huggingface_hub import HfApi


def search_huggingface_datasets(queries: list[str], max_results: int = 5) -> list[dict]:
    api = HfApi()
    results = []

    for query in queries:
        try:
            matches = api.list_datasets(search=query, limit=max_results)
            for ds in matches:
                results.append({
                    "name": ds.id,
                    "description": ds.cardData.get("description", "No description available") if ds.cardData else "No card data"
                })
        except Exception as e:
            results.append({
                "name": query,
                "description": f"Error searching: {str(e)}"
            })

    return results
