import arxiv


def search_arxiv(queries: list[str], max_results: int = 5) -> list[dict]:
    results = []
    for query in queries:
        search = arxiv.Search(query=query, max_results=max_results)
        for r in search.results():
            results.append({
                "query": query,
                "title": r.title,
                "summary": r.summary,
                "authors": [a.name for a in r.authors],
                "url": r.pdf_url
            })
    return results
