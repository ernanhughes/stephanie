
import logging
from typing import Any, Dict, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.services.knowledge_graph_service import KnowledgeGraphService

log = logging.getLogger(__name__)

class ResearchSprintAgent(BaseAgent):
    def __init__(self, cfg: Dict[str, Any], memory, container, logger) -> None:
        super().__init__(cfg, memory, container, logger)

        self.kg: KnowledgeGraphService = container.get("knowledge_graph")
        self.idea_llm = container.get("prompt")          # LLM / internal idea-generator
        self.critic = container.get("critic")              # TinyCritic / HRM / SICQL combo
        self.planner = container.get("planner")            # experiment / follow-up planner

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        papers = context.get(self.input_key, [])
        await self.ingest_papers(papers, context)
        return context

    async def ingest_papers(self, papers: List[Dict[str, Any]], context: Dict[str, Any]):
        # 1) Index entities + claims via existing KG pipeline
        for p in papers:
            await self.kg._handle_index_request({
                "scorable_id": p["id"],
                "scorable_type": "paper",
                "text": p["full_text"],
                "entities": p.get("entities", []),
                "domains": p.get("domains", []),
            })
            self.kg.upsert_paper(
                paper_id=p["id"],
                title=p["title"], 
                abstract=p["abstract"],
                year=p.get("year"),
                domains=p.get("domains", []),
            )

        # 2) Build per-paper knowledge trees, surface gaps
        for p in papers:
            tree = self.kg.build_tree(
                paper_text=p["full_text"],
                paper_id=p["id"],
                chat_corpus=p.get("chat_corpus", []),
                trajectories=p.get("trajectories", []),
                domains=p.get("domains", []),
            )
            for gap in tree.get("knowledge_gaps", []):
                self.kg.upsert_gap(
                    gap_id=f"{p['id']}:{gap['claim_id']}",
                    claim_text=gap["claim_text"],
                    severity=gap["severity"],
                    paper_id=p["id"],
                    concept_ids=[],  # optional: map claim->concepts
                )

    def discover_innovation_frontier(self) -> List[Dict[str, Any]]:
        # 3) Find promising concept pairs
        return self.kg.find_innovation_candidates(
            novelty_min=0.4,
            novelty_max=0.8,
            limit_pairs=200,
        )

    async def generate_ideas(self, candidates: List[Dict[str, Any]], top_n: int = 10):
        ideas = []
        for c in candidates[:top_n * 3]:  # oversample, we'll filter
            concept_a_meta = self.kg._graph.get_metadata(c["concept_a"])
            concept_b_meta = self.kg._graph.get_metadata(c["concept_b"])

            prompt = self._build_idea_prompt(concept_a_meta, concept_b_meta)
            raw_idea = await self.idea_llm.generate(prompt)

            score = self.critic.evaluate(raw_idea, context=[concept_a_meta, concept_b_meta])
            if score["novelty"] > 0.6 and score["feasibility"] > 0.4:
                plan = self.planner.make_plan(raw_idea, context=[concept_a_meta, concept_b_meta])
                ideas.append(
                    {
                        "concept_a": concept_a_meta,
                        "concept_b": concept_b_meta,
                        "idea_text": raw_idea,
                        "scores": score,
                        "plan": plan,
                    }
                )

        # sort by some composite score from critic
        ideas.sort(key=lambda x: -(x["scores"]["novelty"] * x["scores"]["feasibility"]))
        return ideas[:top_n]

    def _build_idea_prompt(self, a_meta, b_meta) -> str:
        return (
            f"You are an AI research collaborator.\n"
            f"Concept A: {a_meta.get('name') or a_meta.get('text')}\n"
            f"Summary A: {a_meta.get('summary','')}\n\n"
            f"Concept B: {b_meta.get('name') or b_meta.get('text')}\n"
            f"Summary B: {b_meta.get('summary','')}\n\n"
            f"Propose a **concrete research idea** that connects A and B in a novel but plausible way, "
            f"including hypothesis, method sketch, and expected challenges."
        )
