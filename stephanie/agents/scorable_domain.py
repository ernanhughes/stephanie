import traceback
from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable
from stephanie.analysis.scorable_classifier import ScorableClassifier


class ScorableDomainAgent(BaseAgent):
    """
    Ensures that every scorable object has domains + embeddings attached.
    Reports enrichment events to SIS.
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.domain_classifier = ScorableClassifier(
            memory,
            logger,
            config_path=cfg.get("domain_config", "config/domain/seeds.yaml"),
        )
        self.embedding_backend = memory.embedding
        self.report_enabled = cfg.get("report", True)

    async def run(self, context: dict) -> dict:
        """
        Find all new scorables in context and enrich them with embeddings + domains.
        """
        try:
            scorables = self.get_scorables(context)
            enriched = []

            for sc in scorables:
                if not isinstance(sc, Scorable):
                    sc = Scorable(
                        id=str(sc.get("id")),
                        text=sc.get("text"),
                        target_type=sc.get("target_type", "document"),
                    )

                # 1. Ensure embedding exists
                emb_id, _ = self.embedding_backend.get_or_create(sc.text)
                self.memory.scorable_embeddings.get_or_create(
                    document_id=sc.id,
                    document_type=sc.target_type,
                    embedding_id=emb_id,
                    embedding_type=self.embedding_backend.type,
                )

                # 2. Ensure domains exist
                domains = self.domain_classifier.classify(sc.text, top_k=3)
                self.memory.scorable_domains.set_domains(sc.id, domains)

                record = {
                    "id": sc.id,
                    "target_type": sc.target_type,
                    "domains": domains,
                    "embedding_id": emb_id,
                }
                enriched.append(record)

                # ✅ Log enrichment event
                self.logger.log("ScorableEnriched", record)

            context["enriched_scorables"] = enriched

            # ✅ Push report into SIS (if enabled)
            self.report(
                {
                    "event": "scorable_domain",
                    "step": "ScorableDomain",
                    "details": f"{len(enriched)} scorables enriched with embeddings + domains.",
                }
            )

            return context

        except Exception as e:
            self.logger.log(
                "ScorableDomainAgentError",
                {"error": str(e), "trace": traceback.format_exc()},
            )
            return context
