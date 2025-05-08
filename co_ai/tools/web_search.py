# co_ai/tools/web_search.py

from duckduckgo_search import DDGS


class WebSearchTool:
    async def search(self, query: str, max_results: int = 5) -> list[str]:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"{r['title']}: {r['body']}")
        return results
