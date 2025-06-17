# co_ai/agents/knowledge_loader.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from co_ai.agents.base_agent import BaseAgent
from co_ai.constants import GOAL


class KnowledgeLoaderAgent(BaseAgent):

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.domain_seeds = cfg.get("domain_seeds", {})
        self.top_k = cfg.get("top_k", 3)
        self.threshold = cfg.get("domain_threshold", 0.0)
        self.include_full_text = cfg.get("include_full_text", False)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        goal_text = goal.get("goal_text", "")
        documents = context.get("documents", [])

        if not goal_text or not documents:
            self.logger.log("DocumentFilterSkipped", {"reason": "Missing goal or documents"})
            return context

        # Step 1: Assign domain to the goal
        goal_vector = self.memory.embedding.get_or_create(goal_text)
        domain_vectors = {
            domain: np.mean([self.memory.embedding.get_or_create(ex) for ex in examples], axis=0)
            for domain, examples in self.domain_seeds.items()
        }

        goal_domain = None
        goal_domain_score = -1
        domain_scores = []

        for domain, vec in domain_vectors.items():
            score = float(cosine_similarity([goal_vector], [vec])[0][0])
            domain_scores.append((domain, score))
            if score > goal_domain_score:
                goal_domain = domain
                goal_domain_score = score

        context["goal_domain"] = goal_domain
        context["goal_domain_score"] = goal_domain_score
        self.logger.log("GoalDomainAssigned", {"domain": goal_domain, "score": goal_domain_score})

        # Step 2: Filter documents based on domain match
        filtered = []
        for doc in documents:
            doc_domains = self.memory.document_domains.get_domains(doc["id"])
            if not doc_domains:
                continue

            for dom in doc_domains[:self.top_k]:
                if dom.domain == goal_domain and dom.score >= self.threshold:
                    selected_content = doc["text"] if self.include_full_text else doc["summary"]
                    filtered.append({
                        "id": doc["id"],
                        "title": doc["title"],
                        "domain": dom.domain,
                        "score": dom.score,
                        "content": selected_content
                    })
                    break  # stop at first matching domain

        context[self.output_key] = filtered
        context["filtered_document_ids"] = [doc["id"] for doc in filtered]
        self.logger.log("DocumentsFiltered", {"count": len(filtered)})

        return context
