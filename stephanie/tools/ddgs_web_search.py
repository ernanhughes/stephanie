# stephanie/tools/ddgs_web_search.py
import asyncio

from duckduckgo_search import DDGS


class DDGSWebSearchTool:
    def search(self, query: str, max_results: int = 5):
        return DDGS().text(query, max_results=max_results)

    async def search2(self, query: str, max_results: int = 5) -> list[str]:
        def _run_search():
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    title = r.get("title", "").strip()
                    body = r.get("body", "").strip()
                    href = r.get("href", "")
                    results.append(f"{title}: {body}\n{href}")
            return results

        try:
            return await asyncio.to_thread(_run_search)
        except Exception as e:
            print(f"❌ Exception [WebSearchTool] Search error: {type(e).__name__}: {e}")
            return [f"Search failed: {str(e)}"]
