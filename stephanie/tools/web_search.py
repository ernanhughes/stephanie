import asyncio

import httpx
import requests
from bs4 import BeautifulSoup
from readability import Document

from stephanie.utils.file_utils import write_text_to_file


class WebSearchTool:
    def __init__(self, cfg: dict, logger):
        self.base_url = f'{cfg.get("instance_url", "localhost:8080")}/search'
        self.max_results = cfg.get("max_results", 15)
        self.fetch_page = cfg.get("fetch_page", False)
        self.categories = cfg.get("categories", "general")
        self.language = cfg.get("language", "en")
        self.logger = logger

    async def search(self, query: str, max_results: int = 15) -> list[str] | None:
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
            print(f"❌ Exception:  {type(e).__name__}: {e}")
            return None

        return self.parse_searxng_results(html, max_results)

    from bs4 import BeautifulSoup

    def parse_searxng_results(self, html: str, max_results:int=20):
        soup = BeautifulSoup(html, "html.parser")
        results = []

        for i, article in enumerate(soup.find_all("article", class_="result")):
            if i > max_results:
                continue
            link_tag = article.find("a", class_="url_header")
            href = link_tag["href"] if link_tag else None

            title_tag = article.find("h3")
            title = title_tag.get_text(strip=True) if title_tag else None

            snippet_tag = article.find("p", class_="content")
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else None

            cleand_page = ""
            if self.fetch_page:
                cleand_page = self.fetch_html(href)

            if href and title:
                results.append(
                    {
                        "title": title,
                        "url": href,
                        "snippet": snippet,
                        "page": cleand_page,
                    }
                )

        return results

    import requests

    def fetch_html(self, url: str) -> str | None:
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            if self.logger:
                self.logger.log("FetchHTMLFailed", {"url": url, "error": str(e)})
            return None  # or return ""

    def fetch_and_parse_readable(self, url:str):
        html = self.fetch_html(url)
        title, clean_text = self.extract_main_text(html)
        return {"url": url, "title": title, "text": clean_text}


    def extract_main_text(self, html):
        doc = Document(html)
        title = doc.short_title()
        summary_html = doc.summary()

        # Use BeautifulSoup to clean text
        soup = BeautifulSoup(summary_html, 'html.parser')
        clean_text = soup.get_text(separator='\n', strip=True)
        return title, clean_text