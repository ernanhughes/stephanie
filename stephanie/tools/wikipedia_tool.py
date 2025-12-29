# stephanie/tools/wikipedia_tool.py
from __future__ import annotations

import wikipedia

from stephanie.tools.similar_tool import get_top_k_similar
import logging

log = logging.getLogger(__name__)


class WikipediaTool:
    def __init__(self, cfg, memory, logger):
        self.cfg = cfg    
        self.memory = memory
        self.logger = logger
        lang = cfg.get("lang", "en")
        wikipedia.set_lang(lang)
        self.top_k = cfg.get("top_k", 3)

    def search(self, query: str) -> list[dict]:
        self.logger.log("WikipediaSearchStart", {"query": query})
        search_results = wikipedia.search(query)
        articles = []

        for title in search_results[:10]:
            try:
                page = wikipedia.page(title)
                summary = page.summary[:2000]
                article = {"title": title, "summary": summary, "url": page.url}
                articles.append(article)
                log.info("WikipediaArticleFetched  article: %s", article)
            except wikipedia.exceptions.DisambiguationError:
                log.error("WikipediaDisambiguationSkipped itle: %s", title)
                continue
            except Exception as e:
                log.error(
                    "WikipediaFetchFailed title: %s, error: %s", title, str(e)
                )
                continue

        log.info(
            "WikipediaSearchComplete query: %s, count: %d", query, len(articles)
        )
        return articles

    def find_similar(self, query: str) -> list[dict]:
        self.logger.log("WikipediaSimilaritySearchStart", {"query": query})
        raw_articles = self.search(query)
        if not raw_articles:
            self.logger.log("WikipediaNoResults", {"query": query})
            return []

        summaries = [a["summary"] for a in raw_articles]
        scored = get_top_k_similar(query, summaries, embed=self.memory.embedding.get_or_create, top_k=self.top_k)
        log.info(
            "WikipediaSimilarityScores scores: %s",
            [{"summary": s, "score": sc} for s, sc in scored],
        )

        final = []
        for summary, score in scored:
            match = next((a for a in raw_articles if a["summary"] == summary), None)
            if match:
                result = match | {"score": round(score, 4)}
                final.append(result)
                log.info("WikipediaMatchSelected result: %s", result)

        log.info(
            "WikipediaSimilaritySearchComplete query: %s, top_k: %d", query, len(final)
        )
        return final
