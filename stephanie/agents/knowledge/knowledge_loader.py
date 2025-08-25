# stephanie/agents/knowledge/knowledge_loader.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL


class KnowledgeLoaderAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.domain_seeds = cfg.get("domain_seeds", {})
        self.top_k = cfg.get("top_k", 3)
        self.threshold = cfg.get("domain_threshold", 0.0)
        self.include_full_text = cfg.get("include_full_text", False)
        self.prefer_sections = cfg.get("prefer_sections", True)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        goal_text = goal.get("goal_text", "")
        documents = context.get("documents", [])

        if not goal_text or not documents:
            self.report({
                "event": "skipped",
                "step": "KnowledgeLoader",
                "reason": "Missing goal or documents",
            })
            return context

        self.report({
            "event": "start",
            "step": "KnowledgeLoader",
            "goal_snippet": goal_text[:120],
            "doc_count": len(documents),
        })

        # Step 1: Assign domain to the goal
        goal_vector = self.memory.embedding.get_or_create(goal_text)
        domain_vectors = {
            domain: np.mean(
                [self.memory.embedding.get_or_create(ex) for ex in examples], axis=0
            )
            for domain, examples in self.domain_seeds.items()
        }

        goal_domain, goal_domain_score = None, -1
        domain_scores = []

        for domain, vec in domain_vectors.items():
            score = float(cosine_similarity([goal_vector], [vec])[0][0])
            domain_scores.append((domain, score))
            if score > goal_domain_score:
                goal_domain, goal_domain_score = domain, score

        context["goal_domain"] = goal_domain
        context["goal_domain_score"] = goal_domain_score

        self.report({
            "event": "goal_domain_assigned",
            "step": "KnowledgeLoader",
            "goal_domain": goal_domain,
            "score": round(goal_domain_score, 4),
            "all_scores": {d: round(s, 4) for d, s in domain_scores},
        })

        # Step 2: Filter documents and/or sections
        filtered = []

        for doc in documents:
            doc_id = doc["id"]
            doc_summary = doc.get("summary", "")
            doc_text = doc.get("text", "")
            doc_title = doc.get("title", "")

            doc_domains = self.memory.document_domains.get_domains(doc_id)

            for dom in doc_domains[: self.top_k]:
                if dom.domain == goal_domain and dom.score >= self.threshold:
                    selected_content = (
                        doc_text if self.include_full_text else doc_summary
                    )
                    filtered.append(
                        {
                            "id": doc_id,
                            "title": doc_title,
                            "domain": dom.domain,
                            "score": dom.score,
                            "content": selected_content,
                            "source": "document",
                        }
                    )
                    self.report({
                        "event": "doc_selected",
                        "step": "KnowledgeLoader",
                        "doc_id": doc_id,
                        "title": doc_title[:80],
                        "domain": dom.domain,
                        "score": round(dom.score, 4),
                    })
                    break

            for section in doc.get("sections", []):
                section_id = section["id"]
                section_name = section.get("section_name", "Unknown")
                section_text = section.get("section_text", "")
                section_summary = section.get("summary", "")

                section_domains = self.memory.section_domains.get_domains(section_id)
                for sec_dom in section_domains[: self.top_k]:
                    if sec_dom.domain == goal_domain and sec_dom.score >= self.threshold:
                        selected_content = (
                            section_text if self.include_full_text else section_summary
                        )
                        filtered.append(
                            {
                                "id": doc_id,
                                "section_id": section_id,
                                "title": f"{doc_title} - {section_name}",
                                "domain": sec_dom.domain,
                                "score": sec_dom.score,
                                "content": selected_content,
                                "source": "section",
                            }
                        )
                        self.report({
                            "event": "section_selected",
                            "step": "KnowledgeLoader",
                            "doc_id": doc_id,
                            "section": section_name,
                            "domain": sec_dom.domain,
                            "score": round(sec_dom.score, 4),
                        })
                        break

        context[self.output_key] = filtered
        context["filtered_document_ids"] = list({doc["id"] for doc in filtered})

        self.report({
            "event": "end",
            "step": "KnowledgeLoader",
            "filtered_count": len(filtered),
            "unique_docs": len(context["filtered_document_ids"]),
        })

        return context
