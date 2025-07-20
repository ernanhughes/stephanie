import os
from typing import Any, Dict, List, Optional

import yaml

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL
from stephanie.models.belief_cartridge import BeliefCartridgeORM


class BeliefCartridgeBuilder:
    """
    Converts structured research ideas into reusable cognitive scaffolds called 'Belief Cartridges'.

    Features:
        - Builds cartridges from parsed paper ideas
        - Attaches metadata + embeddings
        - Tags by domain/technique
        - Stores in memory for reuse
    """

    def __init__(
        self,
        cfg: dict,
        memory: Any = None,
        prompt_loader: Any = None,
        logger: Any = None,
        call_llm: Any = None,
    ):
        self.cfg = cfg
        self.memory = memory
        self.prompt_loader = prompt_loader
        self.logger = logger
        self.call_llm = call_llm

        # Load cartridge templates
        self.template_dir = cfg.get("template_dir", "templates/cartridges")
        self.default_template = cfg.get("default_template", "base.yaml")

        # Tagging configuration
        self.tag_heuristics = {
            "reinforcement": ["q-value", "policy", "reward", "rl"],
            "stability": ["penalty", "regularization", "smooth", "consistent"],
            "loss_term": ["loss", "objective", "gradient", "optimize"],
            "representation": ["embedding", "transform", "encode", "feature"],
            "planning": ["tree", "search", "plan", "lookahead"],
        }

    def build(
        self,
        title: str,
        content: str,
        source_type: str = "paper",
        source_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BeliefCartridgeORM:
        """
        Main method to build a belief cartridge from raw idea content.
        """
        try:
            # Step 1: Normalize inputs
            metadata = metadata or {}
            source_paper = metadata.get("source_paper", "unknown_paper")
            abstract = metadata.get("abstract", "")

            # Step 2: Build basic structure
            cartridge_data = {
                "title": title,
                "content": content,
                "source_type": source_type,
                "source_id": source_id,
                "source_paper": source_paper,
                "abstract": abstract,
                "integration_hint": metadata.get("integration_hint", ""),
                "type": metadata.get("type", "general"),
                "tags": self._tag_idea(content),
                "metadata": metadata,
            }

            # Step 3: Generate embedding
            embed_text = f"{title}\n\n{abstract}"
            self.memory.embedding.get_or_create(embed_text)
            embedding_vector_id = self.memory.embedding.get_id_for_text(embed_text)

            cartridge_data["embedding_id"] = embedding_vector_id

            # Step 4: Save to DB
            cartridge = self.memory.belief_cartridges.insert(cartridge_data)
            self.assign_domains_to_cartridge(cartridge)

            # Log success
            self.logger.log(
                "BeliefCartridgeBuilt",
                {
                    "cartridge_id": cartridge.id,
                    "title": title[:60],
                    "tags": cartridge_data["tags"],
                },
            )

            return cartridge

        except Exception as e:
            self.logger.log(
                "BeliefCartridgeBuildFailed",
                {"error": str(e), "title": title}
            )
            raise

    def _tag_idea(self, content: str) -> List[str]:
        """
        Uses heuristics + embeddings to assign tags like 'rl', 'stability', etc.
        """
        if not content:
            return []

        tags = set()

        # Heuristic-based tagging
        content_lower = content.lower()
        for tag, keywords in self.tag_heuristics.items():
            if any(kw in content_lower for kw in keywords):
                tags.add(tag)

        # Fallback using similarity
        if not tags:
            fallback_tags = self._infer_tags_via_similarity(content)
            tags.update(fallback_tags)

        return list(tags)

    def _infer_tags_via_similarity(self, content: str) -> List[str]:
        """
        Fall back to embedding-based domain classification when no heuristic matches.
        """
        from stephanie.analysis.domain_classifier import DomainClassifier

        classifier = DomainClassifier(
            memory=self.memory,
            logger=self.logger,
            config_path=self.cfg.get("domain_seed_config_path", "config/domain/seeds.yaml"),
        )

        results = classifier.classify(content, top_k=3, min_score=0.5)
        return [domain for domain, score in results]

    def load_from_template(self, template_name: str) -> Dict[str, Any]:
        """
        Loads a cartridge structure from a YAML template.
        """
        template_path = os.path.join(self.template_dir, template_name)
        try:
            with open(template_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.log("TemplateNotFound", {"template": template_path})
            return {}

    def save_to_yaml(self, cartridge: BeliefCartridgeORM, path: str):
        """
        Saves a belief cartridge to disk in YAML format.
        Useful for inspection, sharing, or archiving.
        """
        data = {
            "id": cartridge.id,
            "title": cartridge.title,
            "type": cartridge.type,
            "description": cartridge.content,
            "source_paper": cartridge.source_paper,
            "source_url": cartridge.source_url,
            "abstract": cartridge.abstract,
            "integration_hint": cartridge.integration_hint,
            "tags": cartridge.tags,
            "metadata": cartridge.metadata,
            "scoring": {
                "usefulness": round(cartridge.usefulness_score, 2),
                "novelty": round(cartridge.novelty_score, 2),
                "alignment": round(cartridge.alignment_score, 2),
            },
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        self.logger.log("BeliefCartridgeSaved", {"path": path})

    def bulk_save_all(self, directory: str):
        """
        Exports all stored cartridges to individual YAML files.
        """
        cartridges = self.memory.belief_cartridges.all()
        for c in cartridges:
            filename = f"bc_{c.id}_{c.title.replace(' ', '_')}.yaml"
            self.save_to_yaml(c, os.path.join(directory, filename))

    def attach_scoring_signals(self, cartridge: BeliefCartridgeORM, context: dict):
        """
        Optionally attaches learned scores (e.g., GILD, MRQ) to the cartridge.
        """
        goal = context.get(GOAL, {})
        gild_score = self._compute_gild_score(cartridge, goal)
        mrq_score = self._compute_mrq_score(cartridge, goal)

        cartridge.usefulness_score = gild_score * 0.5 + mrq_score * 0.5
        cartridge.novelty_score = self._compute_novelty_score(cartridge)
        cartridge.alignment_score = mrq_score

        self.memory.session.commit()

    def _compute_gild_score(self, cartridge: BeliefCartridgeORM, goal: dict) -> float:
        """
        Placeholder for actual GILD-style signal integration.
        """
        return 0.75  # Dummy score; replace with real model later

    def _compute_mrq_score(self, cartridge: BeliefCartridgeORM, goal: dict) -> float:
        """
        Placeholder for MRQ scoring logic.
        """
        return 0.82

    def _compute_novelty_score(self, cartridge: BeliefCartridgeORM) -> float:
        """
        Measures how novel this idea is compared to existing ones.
        """
        similar = self.memory.belief_cartridges.find_similar(
            cartridge.embedding_id, top_k=5
        )
        if not similar:
            return 1.0  # Completely new
        avg_similarity = sum(s.score for s in similar) / len(similar)
        return round(1.0 - avg_similarity, 2)
    

    def assign_domains_to_cartridge(self, cartridge: BeliefCartridgeORM):
        """
        Uses DomainClassifier to assign domains to a belief cartridge.
        """
        content = cartridge.description or cartridge.content
        if not content:
            self.logger.log("CartridgeNoContent", {"cartridge_id": cartridge.id})
            return

        results = self.domain_classifier.classify(content, self.top_k_domains, self.min_classification_score)

        for domain, score in results:
            self.memory.cartridge_domains.insert(
                {
                    "cartridge_id": cartridge.id,
                    "domain": domain,
                    "score": float(score),
                }
            )
            self.logger.log(
                "CartridgeDomainAssigned",
                {
                    "title": cartridge.title[:60],
                    "domain": domain,
                    "score": score,
                },
            )