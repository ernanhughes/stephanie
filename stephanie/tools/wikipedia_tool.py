# stephanie/tools/wikipedia_tool.py
import wikipedia

from stephanie.tools.cos_sim_tool import get_top_k_similar


class WikipediaTool:
    def __init__(self, memory, logger, lang="en", top_k=3):
        self.memory = memory
        self.logger = logger
        wikipedia.set_lang(lang)
        self.top_k = top_k

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
                self.logger.log("WikipediaArticleFetched", {"article": article})
            except wikipedia.exceptions.DisambiguationError as e:
                self.logger.log("WikipediaDisambiguationSkipped", {"title": title})
                continue
            except Exception as e:
                self.logger.log(
                    "WikipediaFetchFailed", {"title": title, "error": str(e)}
                )
                continue

        self.logger.log(
            "WikipediaSearchComplete", {"query": query, "count": len(articles)}
        )
        return articles

    def find_similar(self, query: str) -> list[dict]:
        self.logger.log("WikipediaSimilaritySearchStart", {"query": query})
        raw_articles = self.search(query)
        if not raw_articles:
            self.logger.log("WikipediaNoResults", {"query": query})
            return []

        summaries = [a["summary"] for a in raw_articles]
        scored = get_top_k_similar(query, summaries, self.memory, top_k=self.top_k)
        self.logger.log(
            "WikipediaSimilarityScores",
            {"scores": [{"summary": s, "score": sc} for s, sc in scored]},
        )

        final = []
        for summary, score in scored:
            match = next((a for a in raw_articles if a["summary"] == summary), None)
            if match:
                result = match | {"score": round(score, 4)}
                final.append(result)
                self.logger.log("WikipediaMatchSelected", {"result": result})

        self.logger.log(
            "WikipediaSimilaritySearchComplete", {"query": query, "top_k": len(final)}
        )
        return final
