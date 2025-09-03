# stephanie/analysis/scorable_classifier.py

from typing import Dict, List, Optional, Tuple

import yaml
from sklearn.metrics.pairwise import cosine_similarity


class ScorableClassifier:
    def __init__(self, memory, logger, config_path="config/domain/seeds.yaml"):
        self.memory = memory
        self.logger = logger
        self.logger.log("DomainClassifierInit", {"config_path": config_path})

        # Load static seed domains
        with open(config_path, "r") as f:
            self.domain_config = yaml.safe_load(f)

        self.domains = self.domain_config.get("domains", {})
        self.logger.log("DomainConfigLoaded", {"num_domains": len(self.domains)})
        self.domain_keys = ["tags", "domains", "keywords", "attributes", "concepts"]

        # Precompute seed embeddings
        self._prepare_seed_embeddings()

    def _prepare_seed_embeddings(self):
        """Precompute embeddings for all seed phrases in the YAML config."""
        self.seed_embeddings = []
        self.seed_labels = []
        total_seeds = 0

        for domain, details in self.domains.items():
            seeds = details.get("seeds", [])
            total_seeds += len(seeds)
            for seed in seeds:
                embedding = self.memory.embedding.get_or_create(seed)
                self.seed_embeddings.append(embedding)
                self.seed_labels.append(domain)

        self.logger.log(
            "SeedEmbeddingsPrepared",
            {"total_seeds": total_seeds, "domains": list(self.domains.keys())},
        )

    def classify(
        self,
        text: str,
        top_k: int = 3,
        min_value: float = 0.7,
        context: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        """
        Classify text into domains.
        If `context` is provided and has goal-specific domains/tags, include them in scoring.

        Args:
            text: The text to classify
            top_k: Number of top domains to return
            min_value: Minimum confidence to avoid low-match warnings
            context: Optional dict, e.g. {"goal": {"name": "...", "tags": ["control", "robotics", ...]}}

        Returns:
            List of (domain, score) tuples sorted by score descending
        """
        # Get embedding for input text
        text_embedding = self.memory.embedding.get_or_create(text)
        scores = []

        # === 1. Score against static seed domains ===
        for domain, seed_emb in zip(self.seed_labels, self.seed_embeddings):
            score = float(cosine_similarity([text_embedding], [seed_emb])[0][0])
            scores.append((domain, score))

        # === 2. Score against dynamic goal domains (if provided in context) ===
        if context:
            goal_tags = self._extract_domains(context)
            for tag in goal_tags:
                tag_embedding = self.memory.embedding.get_or_create(tag)
                score = float(cosine_similarity([text_embedding], [tag_embedding])[0][0])
                scores.append((tag, score))  # Use the tag itself as the "domain"

        # Sort and deduplicate (keep highest score for any domain)
        scored_dict = {}
        for domain, score in scores:
            if domain not in scored_dict or score > scored_dict[domain]:
                scored_dict[domain] = score

        sorted_scores = sorted(scored_dict.items(), key=lambda x: x[1], reverse=True)

        top_matches = sorted_scores[:top_k]

        # Log if all scores are low
        if all(score < min_value for _, score in top_matches):
            self.logger.log(
                "LowDomainScore",
                {
                    "text_snippet": text[:100],
                    "top_scores": top_matches,
                    "context_has_goal": bool(context),
                },
            )

        return top_matches


    def _extract_domains(self, context: Dict) -> List[str]:
        """
        Extract domain tags from the dictionary in context.
        We can pass in a goal document anything with a set of domains.
        """
        if isinstance(context, dict):
            # Look for tags, domains, keywords, etc.
            for key in self.domain_keys:
                if key in context and isinstance(context[key], (list, tuple)):
                    return [str(t).strip() for t in context[key] if t]
I