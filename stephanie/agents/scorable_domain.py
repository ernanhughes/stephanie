# agents/domain/multi_source_domain_agent.py

import traceback
from typing import List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.scoring.scorable import Scorable


class ScorableDomainAgent(BaseAgent):
    """
    Enriches scorables with domains from multiple sources:
    - Seed (controlled ontology)
    - Metadata (source-provided, e.g. arXiv categories)
    - Goal (pipeline-specific, ephemeral)
    - Emergent (cluster-based, auto-labeled)

    Each domain is tagged with source and confidence.
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)

        # 1. Seed classifier (YAML-based controlled ontology)
        self.seed_classifier = ScorableClassifier(
            memory,
            logger, 
            config_path=cfg.get("seed_config", "config/domain/seeds.yaml"),
        )

        # 2. Goal classifier (optional, adaptive per goal)
        self.goal_classifier = ScorableClassifier(
            memory,
            logger,
            config_path=cfg.get("goal_config", "config/domain/goal_prompt.yaml"),
        )

        # 3. Embedding + clustering backend
        self.embedding_backend = memory.embedding

        # 5. Config flags
        self.report_enabled = cfg.get("report", True)
        self.min_confidence = cfg.get("min_confidence", 0.1)
        self.max_domains_per_source = cfg.get("max_domains_per_source", 3)

    async def run(self, context: dict) -> dict:
        """
        Enrich all scorables in context with multi-source domains.
        """
        try:
            scorables = self.get_scorables(context)
            enriched = []

            pipeline_id = context.get("pipeline_id", "unknown")
            goal = context.get("goal")

            for sc in scorables:
                if not isinstance(sc, Scorable):
                    sc = Scorable(
                        id=str(sc.get("id")),
                        text=sc.get("text"),
                        target_type=sc.get("target_type", "document"),
                        metadata=sc.get("metadata", {}),
                    )

                # 1. Ensure embedding exists
                emb_id = self.memory.scorable_embeddings.get_or_create(sc)

                # 2. Collect domains from all 4 tiers
                domains = await self._enrich_with_multi_source_domains(
                    sc, goal=goal, pipeline_id=pipeline_id
                )
                filtered_domains = [
                    (d, s) for d, s in domains if s >= self.min_confidence
                ][: self.max_domains_per_source * 4]

                # 3. Save to memory
                self.memory.scorable_domains.set_domains(sc.id, filtered_domains)

                record = {
                    "id": sc.id,
                    "target_type": sc.target_type,
                    "domains": filtered_domains,
                    "embedding_id": emb_id,
                }
                enriched.append(record)

                # ✅ Log enrichment event
                self.logger.log("ScorableEnriched", record)

            context["enriched_scorables"] = enriched

            # ✅ Push report into SIS
            if self.report_enabled:
                self.report(
                    {
                        "event": "multi_source_domain_enrichment",
                        "step": "MultiSourceDomainAgent",
                        "details": f"{len(enriched)} scorables enriched with multi-source domains.",
                        "sources_used": ["seed", "metadata", "goal", "emergent"],
                    }
                )

            return context

        except Exception as e:
            self.logger.log(
                "MultiSourceDomainAgentError",
                {"error": str(e), "trace": traceback.format_exc()},
            )
            return context

    async def _enrich_with_multi_source_domains(
        self, sc: Scorable, goal: Optional[str], pipeline_id: str
    ) -> List[Tuple[str, float]]:
        """
        Returns list of (domain, score) tuples with source tagging.
        """
        all_domains = []

        # === 1. Seed Domains (stable, controlled) ===
        seed_domains = self.seed_classifier.classify(
            sc.text, top_k=self.max_domains_per_source
        )
        all_domains.extend([("seed:" + d, s) for d, s in seed_domains])

        # === 2. Source Metadata Domains ===
        meta_domains = self._extract_metadata_domains(sc)
        all_domains.extend([("meta:" + d, 1.0) for d in meta_domains])

        # === 3. Goal-Specific Domains (ephemeral) ===
        if goal:
            goal_domains = self.goal_classifier.classify(
                sc.text, context=goal, top_k=self.max_domains_per_source
            )
            # Tag with pipeline_id as TTL handle
            all_domains.extend([
                (f"goal:{d}|ttl:{pipeline_id}", s) for d, s in goal_domains
            ])

        # === 4. Emergent Domains (clustering fallback) ===
        emergent_domains = self.clustering_backend.classify(
            sc.text, exclude_labels=[d for d, _ in all_domains]
        )
        labeled_domains = await self._label_emergent_clusters(emergent_domains)
        all_domains.extend([("emergent:" + d, s) for d, s in labeled_domains])

        return sorted(all_domains, key=lambda x: x[1], reverse=True)

    def _extract_metadata_domains(self, sc: Scorable) -> List[str]:
        """
        Extract domain-like tags from scorable's metadata.
        Customize per source: arXiv, PubMed, GitHub, etc.
        """
        meta = sc.metadata or {}
        domains = []

        # Arxiv
        if "arxiv_primary_category" in meta:
            domains.append(meta["arxiv_primary_category"])
        if "arxiv_secondary_categories" in meta:
            domains.extend(meta["arxiv_secondary_categories"])

        # PubMed
        if "mesh_terms" in meta:
            domains.extend(meta["mesh_terms"][:3])

        # GitHub
        if "repo_topics" in meta:
            domains.extend([t.replace(" ", "_") for t in meta["repo_topics"][:3]])
        if "repo_language" in meta:
            domains.append(f"lang:{meta['repo_language']}")

        # Web / General
        if "keywords" in meta:
            domains.extend([k.replace(" ", "_") for k in meta["keywords"][:2]])

        # Dedup
        return list(dict.fromkeys(domains))  # preserve order

    async def _label_emergent_clusters(self, clusters: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Convert raw cluster IDs into human-readable labels.
        Uses LLM if available, else falls back to keyword extraction.
        """
        if not clusters:
            return []

        labeled = []

        for cluster_id, score in clusters:
            if self.llm:
                try:
                    prompt = (
                        "Given a cluster of similar scientific/technical documents, "
                        "suggest a short descriptive label (1-3 words) for this group:\n\n"
                        f"Cluster ID: {cluster_id}\n\n"
                        "Label:"
                    )
                    label = (await self.llm.complete(prompt)).strip().replace(" ", "_")
                    labeled.append((label, score))
                except Exception as e:
                    self.logger.log("LLMLabelError", {"error": str(e)})
                    labeled.append((f"cluster_{cluster_id[:6]}", score))
            else:
                labeled.append((f"cluster_{cluster_id[:6]}", score))

        return labeled