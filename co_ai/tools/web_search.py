import asyncio

import httpx
from bs4 import BeautifulSoup


class WebSearchTool:
    def __init__(self, cfg: dict):
        self.base_url = f'{cfg.get("instance_url", "localhost:8080")}/search'
        self.max_results = cfg.get("max_results", 15)
        self.categories = cfg.get("categories", "general")
        self.language = cfg.get("language", "en")

    async def search(self, query: str, max_results: int = 15) -> list[str]:
        max_results = max_results or self.max_results

        params = {
            "q": query,
            "categories": "general",
            "language": self.language,
            "formats": ["html", "json"]
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(self.base_url, params=params)
                resp.raise_for_status()
                html = resp.text
        except Exception as e:
            print(f"[WebSearchTool] SearXNG search error: {e}")
            return [f"Search failed: {str(e)}"]

        return self._parse_html_results(html, max_results)
    
    def _parse_html_results(self, html: str, max_results: int) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        results = []

        for result in soup.select(".result")[:max_results]:
            title_tag = result.select_one(".result__title")
            content_tag = result.select_one(".result__snippet")
            link_tag = title_tag.find("a") if title_tag else None

            title = title_tag.text.strip() if title_tag else ""
            snippet = content_tag.text.strip() if content_tag else ""
            url = link_tag["href"].strip() if link_tag and link_tag.has_attr("href") else ""

            if url:
                results.append(f"{title}: {snippet}\n{url}")

        return results