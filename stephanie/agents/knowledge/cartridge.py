# stephanie/agents/knowledge/cartridge.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.scoring_mixin import ScoringMixin
from stephanie.analysis.domain_classifier import DomainClassifier
from stephanie.builders.cartridge_builder import CartridgeBuilder
from stephanie.builders.theorem_extractor import TheoremExtractor
from stephanie.builders.triplet_extractor import TripletExtractor
from stephanie.models.theorem import CartridgeORM
from stephanie.scoring.scorable_factory import TargetType
from stephanie.scoring.scoring_engine import ScoringEngine


class CartridgeAgent(ScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

        self.input_key = cfg.get("input_key", "documents")
        self.score_cartridges = cfg.get("score_cartridges", True)
        self.score_triplets = cfg.get("score_triplets", True)
        self.top_k_domains = cfg.get("top_k_domains", 3)
        self.min_classification_score = cfg.get("min_classification_score", 0.6)

        self.domain_classifier = DomainClassifier(
            memory=self.memory,
            logger=self.logger,
            config_path=cfg.get(
                "domain_seed_config_path", "config/domain/cartridges.yaml"
            ),
        )

        self.builder = CartridgeBuilder(
            cfg,
            memory=self.memory,
            prompt_loader=self.prompt_loader,
            logger=self.logger,
            call_llm=self.call_llm,
        )
        self.triplet_extractor = TripletExtractor(
            cfg=cfg,
            prompt_loader=self.prompt_loader,
            memory=self.memory,
            logger=self.logger,
            call_llm=self.call_llm,
        )
        self.theorem_extractor = TheoremExtractor(
            cfg=cfg,
            prompt_loader=self.prompt_loader,
            memory=self.memory,
            logger=self.logger,
            call_llm=self.call_llm,
        )

        self.scoring_engine = ScoringEngine(
            cfg=self.cfg,
            memory=self.memory,
            prompt_loader=self.prompt_loader,
            logger=self.logger,
            call_llm=self.call_llm,
        )

    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        cartridges = []

        total_docs = len(documents)
        self.logger.log("CartridgeProcessingStarted", {"total_documents": total_docs})

        for idx, doc in enumerate(documents, start=1):
            self.logger.log(
                "CartridgeDocumentProcessingStarted",
                {
                    "current_document": idx,
                    "total_documents": total_docs,
                    "document_id": doc.get("id"),
                },
            )
            try:
                goal = context.get("goal")

                # 1. Build CartridgeORM
                cartridge = self.builder.build(doc, goal=goal)
                if not cartridge:
                    self.logger.log(
                        "CartridgeSkipped",
                        {
                            "reason": "Builder returned None",
                            "document_id": doc.get("id"),
                        },
                    )
                    continue
                self.logger.log("CartridgeBuilt", {"cartridge_id": cartridge.id})

                # 2. Extract and insert triplets
                if self.memory.cartridge_triples.has_triples(cartridge.id):
                    self.logger.log(
                        "TriplesAlreadyExist", {"cartridge_id": cartridge.id}
                    )
                else:
                    triplets = self.triplet_extractor.extract(
                        cartridge.sections, context
                    )
                    total_triplets = len(triplets)
                    self.logger.log(
                        "TripletsExtractionCompleted",
                        {
                            "cartridge_id": cartridge.id,
                            "total_triplets": total_triplets,
                        },
                    )

                    for subj, pred, obj in triplets:
                        triple_orm = self.memory.cartridge_triples.insert(
                            {
                                "cartridge_id": cartridge.id,
                                "subject": subj,
                                "predicate": pred,
                                "object": obj,
                            }
                        )
                        if self.score_triplets:
                            triplet_text = f"({subj}, {pred}, {obj})"
                            score = self.scoring_engine.score(
                                target_id=triple_orm.id,
                                target_type=TargetType.TRIPLE,
                                text=triplet_text,
                                context=context,
                                scoring_profile="cartridge",
                            )
                            context.setdefault("triplet_scores", []).append(score)
                self.logger.log("TripletsInserted", {"cartridge_id": cartridge.id})

                # 3. Extract and insert theorems
                theorems = self.theorem_extractor.extract(cartridge.sections, context)
                total_theorems = len(theorems)
                self.logger.log(
                    "TheoremsExtractionCompleted",
                    {"cartridge_id": cartridge.id, "total_theorems": total_theorems},
                )

                for theorem in theorems:
                    self.memory.embedding.get_or_create(theorem.statement)
                    theorem.embedding_id = self.memory.embedding.get_id_for_text(
                        theorem.statement
                    )
                    theorem.cartridges.append(cartridge)
                    self.memory.session.add(theorem)

                    merged_context = {
                        "theorem": theorem.to_dict(),
                        "cartridge_title": cartridge.title,
                        "cartridge_id": cartridge.id,
                        **context,
                    }

                    # Score theorem immediately
                    theorem_score = self.scoring_engine.score(
                        target_id=theorem.id,
                        target_type=TargetType.THEOREM,
                        text=theorem.statement,
                        context=merged_context,
                        scoring_profile="cartridge",
                    )
                    context.setdefault("theorem_scores", []).append(theorem_score)
                self.memory.session.commit()
                self.logger.log("TheoremsInserted", {"cartridge_id": cartridge.id})

                # 4. Score Cartridge
                if self.score_cartridges:
                    score = self.scoring_engine.score(
                        target_id=cartridge.id,
                        target_type=TargetType.CARTRIDGE,
                        text=cartridge.markdown_content,
                        context=context,
                        scoring_profile="cartridge",
                    )
                    context.setdefault("cartridge_scores", []).append(score)
                    self.logger.log("CartridgeScored", {"cartridge_id": cartridge.id})

                # 5. Assign Domains
                self.assign_domains(cartridge)

                self.logger.log(
                    "CartridgeProcessingCompleted",
                    {
                        "cartridge_id": cartridge.id,
                        "document_number": idx,
                        "total_documents": total_docs,
                    },
                )

                cartridges.append(cartridge.to_dict())

            except Exception as e:
                self.logger.log(
                    "CartridgeProcessingFailed",
                    {
                        "document_id": doc.get("id"),
                        "error": str(e),
                        "document_number": idx,
                        "total_documents": total_docs,
                    },
                )

        self.logger.log(
            "CartridgeProcessingFinished",
            {"processed_documents": len(cartridges), "total_documents": total_docs},
        )

        context[self.output_key] = cartridges
        context["cartridge_ids"] = [c.get("id") for c in cartridges]
        return context

    def assign_domains(self, cartridge: CartridgeORM):
        """Classify and log domains for the cartridge."""
        if not cartridge.markdown_content:
            return

        existing = self.memory.cartridge_domains.get_domains(cartridge.id)
        if existing:
            self.logger.log(
                "DomainAssignmentSkipped",
                {
                    "cartridge_id": cartridge.id,
                    "existing_domains": [e.domain for e in existing],
                },
            )
            return

        results = self.domain_classifier.classify(
            cartridge.markdown_content,
            top_k=self.top_k_domains,
            min_value=self.min_classification_score,
        )
        for domain, score in results:
            self.memory.cartridge_domains.insert(
                {
                    "cartridge_id": cartridge.id,
                    "domain": domain,
                    "score": score,
                }
            )
            self.logger.log(
                "DomainAssigned",
                {"title": cartridge.title[:60], "domain": domain, "score": score},
            )
