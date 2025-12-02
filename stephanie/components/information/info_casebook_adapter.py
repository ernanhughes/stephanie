# stephanie/components/information/info_casebook_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.components.information.models import Bucket, BucketNode
from stephanie.memory.casebook_store import CaseBookStore
from stephanie.models.goal import GoalORM  # adjust import if needed


@dataclass
class InfoCaseBookArtifacts:
    goal_id: int
    casebook_id: int
    case_ids: List[int]


class InfoCaseBookAdapter:
    """
    Takes a Bucket and turns it into a CaseBook with Cases.

    Each Case ≈ one section in the final Information MemCube.
    """

    def __init__(self, casebook_store: CaseBookStore, logger, llm_client) -> None:
        """
        llm_client: callable with .generate(prompt: str, **kwargs) -> str
        """
        self.casebook_store = casebook_store
        self.logger = logger
        self.llm = llm_client

    async def build_casebook_from_bucket(
        self,
        topic: str,
        bucket: Bucket,
        agent_name: str = "InformationBuilder",
    ) -> InfoCaseBookArtifacts:
        # 1) Ensure Goal exists (very simple for now)
        goal = await self._ensure_goal(topic)

        # 2) Create CaseBook
        cb = self.casebook_store.create_casebook(
            name=f"[info] {topic}",
            description=f"Information builder casebook for topic: {topic}",
            tags=["info", "memcube_target"],
            meta={"topic": topic, "source_profile": bucket.meta.get("source_profile")},
        )

        # 3) Cluster nodes into subtopics → cases
        clusters = await self._cluster_nodes(topic, bucket.nodes)

        case_ids: list[int] = []
        for idx, (section_title, node_ids) in enumerate(clusters):
            # build prompt_text + description from nodes
            selected_nodes = [n for n in bucket.nodes if n.id in node_ids]
            description = await self._summarize_cluster(topic, section_title, selected_nodes)
            prompt_text = self._make_section_prompt(topic, section_title, selected_nodes)

            # Convert nodes to scorables. Simplest: each node becomes one "document" scorable.
            scorables = [
                {
                    "text": n.snippet,
                    "role": "document",
                    "meta": {
                        "source_type": n.source_type,
                        "title": n.title,
                        "url": n.url,
                        "arxiv_id": n.arxiv_id,
                        "doc_id": n.doc_id,
                        "section": n.section,
                    },
                }
                for n in selected_nodes
            ]

            case = self.casebook_store.add_case(
                casebook_id=cb.id,
                goal_id=goal.id,
                agent_name=agent_name,
                goal_text=topic,
                mars_summary=None,
                scores=None,
                scorables=scorables,
                prompt_text=prompt_text,
                meta={
                    "topic": topic,
                    "section_title": section_title,
                    "cluster_index": idx,
                    "source_profile": bucket.meta.get("source_profile"),
                },
                response_texts=None,
            )
            case_ids.append(case.id)

        self.logger.log(
            "InfoCaseBookBuilt",
            {
                "topic": topic,
                "casebook_id": cb.id,
                "goal_id": goal.id,
                "num_cases": len(case_ids),
            },
        )
        return InfoCaseBookArtifacts(
            goal_id=goal.id, casebook_id=cb.id, case_ids=case_ids
        )

    # -------------------- helpers --------------------

    async def _ensure_goal(self, topic: str) -> GoalORM:
        """
        Very simple: check by text, else create.
        You can wire this into your existing GoalStore/ORM.
        """
        # Pseudo-code; adapt to your goal store API
        from stephanie.memory.goal_store import GoalStore  # adjust
        gs = GoalStore(self.casebook_store.session_maker)
        existing = gs.get_by_text(topic)
        if existing:
            return existing
        return gs.create_goal(text=topic, meta={"kind": "information_build"})

    async def _cluster_nodes(
        self, topic: str, nodes: List[BucketNode]
    ) -> List[Tuple[str, List[str]]]:
        """
        Turn BucketNodes into (section_title, [node_ids]) groups.

        For v1 we do a single LLM call that names 4–7 clusters based on titles/snippets.
        You can replace this with real clustering later.
        """
        if not nodes:
            return [("Overview", [])]

        # Build a simple list for the prompt
        lines = []
        for idx, n in enumerate(nodes[:80]):
            lines.append(
                f"- ID={n.id} | source={n.source_type} | title={n.title!r} | snippet={n.snippet[:160].replace('\n', ' ')}"
            )
        listing = "\n".join(lines)

        prompt = f"""
You are organizing information about the topic: {topic}

Here are some snippets, each with an ID:

{listing}

1. Group these into 4–7 coherent sections.
2. For each section, give:
   - A short section title
   - A list of IDs that belong to that section

Respond as JSON with the shape:
[
  {{"title": "Section Title", "ids": ["id1", "id2", ...]}},
  ...
]
"""
        raw = await self.llm.generate(prompt)
        import json

        try:
            data = json.loads(raw)
            clusters = []
            for sec in data:
                sec_title = sec.get("title") or "Section"
                ids = [i for i in sec.get("ids", []) if isinstance(i, str)]
                clusters.append((sec_title, ids))
            if clusters:
                return clusters
        except Exception:
            # fallback: single cluster with all nodes
            pass

        return [("Overview", [n.id for n in nodes])]

    async def _summarize_cluster(
        self, topic: str, section_title: str, nodes: List[BucketNode]
    ) -> str:
        if not nodes:
            return ""
        snippets = "\n\n".join(
            f"- {n.title}: {n.snippet[:200].replace('\n', ' ')}" for n in nodes[:10]
        )
        prompt = f"""
Topic: {topic}
Section: {section_title}

Using the following snippets, write a 1–3 sentence description of what this section is about.
Keep it high-level and non-redundant.

Snippets:
{snippets}
"""
        desc = await self.llm.generate(prompt)
        return desc.strip()

    def _make_section_prompt(
        self, topic: str, section_title: str, nodes: List[BucketNode]
    ) -> str:
        return (
            f"Write a grounded section titled '{section_title}' about '{topic}'. "
            "Use only the attached source snippets and keep it faithful to them."
        )
