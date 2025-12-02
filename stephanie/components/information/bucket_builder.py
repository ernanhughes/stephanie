# stephanie/components/information/bucket_builder.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from stephanie.components.information.models import Bucket, BucketEdge, BucketNode
from stephanie.components.information.source_profile import SourceProfile
from stephanie.memory.memcube_store import MemCubeStore
from stephanie.tools.arxiv_tool import search_arxiv
from stephanie.tools.wikipedia_tool import WikipediaTool
from stephanie.tools.web_search import WebSearchTool
from stephanie.utils.date_utils import iso_now


class BucketBuilder:
    """
    Gathers raw information for a topic from multiple sources and builds a Bucket.

    Dependencies:
      - memcube_store: MemCubeStore (for internal memcubes)
      - memory: Stephanie memory (for embeddings / vector search if you want)
      - logger: unified logger with .log(event, payload)
      - cfg: configuration dict for tools (like web_search)
    """

    def __init__(
        self,
        memcube_store: MemCubeStore,
        memory,
        logger,
        cfg: Dict[str, Any] | None = None,
    ) -> None:
        self.memcube_store = memcube_store
        self.memory = memory
        self.logger = logger
        self.cfg = cfg or {}

        # tools that need cfg
        web_cfg = self.cfg.get("web_search", {})
        self.web_search = WebSearchTool(web_cfg, logger=self.logger)

        wiki_lang = self.cfg.get("wikipedia", {}).get("lang", "en")
        self.wikipedia = WikipediaTool(
            memory=self.memory, logger=self.logger, lang=wiki_lang
        )

    async def build_bucket(
        self,
        topic: str,
        profile: SourceProfile,
    ) -> Bucket:
        bucket = Bucket(topic=topic)
        nodes: list[BucketNode] = []
        edges: list[BucketEdge] = []

        # 1) arXiv
        if profile.use_arxiv:
            await self._collect_arxiv(topic, profile, nodes, edges)

        # 2) Wikipedia
        if profile.use_wikipedia:
            await self._collect_wikipedia(topic, profile, nodes, edges)

        # 3) Web
        if profile.use_web:
            await self._collect_web(topic, profile, nodes, edges)

        # 4) Internal MemCubes (info from your own system)
        if profile.use_internal_memcubes:
            await self._collect_internal_memcubes(topic, profile, nodes, edges)

        # TODO: 5) Vector store / embeddings if you want extra recall

        # Global cap
        if len(nodes) > profile.max_nodes:
            nodes = sorted(nodes, key=lambda n: n.score, reverse=True)[
                : profile.max_nodes
            ]

        bucket.nodes = nodes
        bucket.edges = edges
        bucket.meta["source_profile"] = profile.name
        bucket.meta["created_at"] = iso_now()

        self.logger.log(
            "InfoBucketBuilt",
            {
                "topic": topic,
                "profile": profile.name,
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
        )
        return bucket

    # -------------------- collectors --------------------

    async def _collect_arxiv(
        self,
        topic: str,
        profile: SourceProfile,
        nodes: list[BucketNode],
        edges: list[BucketEdge],
    ) -> None:
        try:
            results = search_arxiv([topic], max_results=profile.max_results_arxiv)
        except Exception as e:
            self.logger.log(
                "InfoBucketArxivError", {"topic": topic, "error": str(e)}
            )
            return

        for r in results:
            arxiv_id = r.get("id") or r.get("arxiv_id")
            title = r.get("title", "").strip()
            summary = (r.get("summary") or "").strip()
            url = r.get("url")

            node_id = f"arxiv:{arxiv_id}"
            score = 1.0  # crude; you can refine (e.g. by similarity)
            n = BucketNode(
                id=node_id,
                source_type="arxiv_meta",
                title=title or f"arxiv {arxiv_id}",
                snippet=summary[:1500],
                url=url,
                arxiv_id=arxiv_id,
                doc_id=arxiv_id,
                section="abstract",
                score=score,
                meta={"kind": "paper"},
            )
            nodes.append(n)

        self.logger.log(
            "InfoBucketArxivCollected",
            {"topic": topic, "count": len(results)},
        )

    async def _collect_wikipedia(
        self,
        topic: str,
        profile: SourceProfile,
        nodes: list[BucketNode],
        edges: list[BucketEdge],
    ) -> None:
        try:
            articles = self.wikipedia.search(topic)
        except Exception as e:
            self.logger.log(
                "InfoBucketWikiError", {"topic": topic, "error": str(e)}
            )
            return

        for idx, art in enumerate(articles[: profile.max_results_wiki]):
            node_id = f"wiki:{art['title']}"
            snippet = art.get("summary", "")
            n = BucketNode(
                id=node_id,
                source_type="wikipedia",
                title=art["title"],
                snippet=snippet,
                url=art.get("url"),
                score=0.8,
                meta={"kind": "wiki"},
            )
            nodes.append(n)

        self.logger.log(
            "InfoBucketWikiCollected",
            {"topic": topic, "count": len(articles)},
        )

    async def _collect_web(
        self,
        topic: str,
        profile: SourceProfile,
        nodes: list[BucketNode],
        edges: list[BucketEdge],
    ) -> None:
        try:
            results = await self.web_search.async_search(topic)
        except Exception as e:
            self.logger.log(
                "InfoBucketWebError", {"topic": topic, "error": str(e)}
            )
            return

        for idx, r in enumerate(results[: profile.max_results_web]):
            node_id = f"web:{idx}"
            title = r.get("title") or "web result"
            snippet = (r.get("summary") or r.get("text") or "")[:1500]
            n = BucketNode(
                id=node_id,
                source_type="web",
                title=title,
                snippet=snippet,
                url=r.get("url"),
                score=0.6,
                meta={"kind": "web"},
            )
            nodes.append(n)

        self.logger.log(
            "InfoBucketWebCollected",
            {"topic": topic, "count": len(nodes)},
        )

    async def _collect_internal_memcubes(
        self,
        topic: str,
        profile: SourceProfile,
        nodes: list[BucketNode],
        edges: list[BucketEdge],
    ) -> None:
        """
        Very simple: search memcubes by topic substring in content/title/domain tag.
        You can later replace with embedding search.
        """
        # This is intentionally crude; refine later.
        maybe_cubes = self.memcube_store.search_by_text(
            query=topic, limit=profile.max_vector_hits
        )

        for cube in maybe_cubes:
            data = cube.to_dict(include_extra=True)
            node_id = f"memcube:{cube.id}"
            title = data.get("extra_data", {}).get("topic") or "MemCube"
            snippet = (cube.content or "")[:1500]
            n = BucketNode(
                id=node_id,
                source_type="memcube",
                title=title,
                snippet=snippet,
                url=None,
                score=0.7,
                doc_id=cube.id,
                meta={
                    "kind": "memcube",
                    "dimension": cube.dimension,
                    "tags": (cube.extra_data or {}).get("tags", []),
                },
            )
            nodes.append(n)

        self.logger.log(
            "InfoBucketMemcubesCollected",
            {
                "topic": topic,
                "count": len(maybe_cubes),
            },
        )
