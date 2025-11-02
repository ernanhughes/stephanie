# stephanie/agents/knowledge/cartridge.py
from __future__ import annotations

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.builders.cartridge_builder import CartridgeBuilder
from stephanie.builders.theorem_extractor import TheoremExtractor
from stephanie.builders.triplet_extractor import TripletExtractor
from stephanie.models.theorem import CartridgeORM
from stephanie.scoring.scorable import Scorable, ScorableFactory, ScorableType
from stephanie.scoring.scorer.ebt_scorer import EBTScorer
from stephanie.scoring.scorer.mrq_scorer import MRQScorer
from stephanie.scoring.scorer.sicql_scorer import SICQLScorer
from stephanie.scoring.scorer.svm_scorer import SVMScorer


class CartridgeAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger, full_cfg=None):
        super().__init__(cfg, memory, container, logger)

        self.input_key = cfg.get("input_key", "documents")
        self.score_cartridges = cfg.get("score_cartridges", True)
        self.score_triplets = cfg.get("score_triplets", True)
        self.scorer_type = cfg.get("scorer_type", "sicql")  # default
        self.scorer = self._init_scorer(full_cfg, self.scorer_type)
        self.dimensions = cfg.get(
            "dimensions", ["relevance", "acy", "completeness"]
        )
        self.force_rescore = cfg.get("force_rescore", False)
        self.top_k_domains = cfg.get("top_k_domains", 3)
        self.min_classification_score = cfg.get(
            "min_classification_score", 0.6
        )
        self.force_rebuild_cartridges = cfg.get(
            "force_rebuild_cartridges", False
        )

        self.domain_classifier = ScorableClassifier(
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

    def _init_scorer(self, full_cfg, scorer_type):
        if scorer_type == "sicql":
            return SICQLScorer(
                full_cfg.scorer.sicql, memory=self.memory, container=self.container, logger=self.logger
            )
        if scorer_type == "mrq":
            return MRQScorer(
                full_cfg.scorer.mrq, memory=self.memory, container=self.container, logger=self.logger
            )
        if scorer_type == "svm":
            return SVMScorer(
                full_cfg.scorer.svm, memory=self.memory, container=self.container, logger=self.logger
            )
        if scorer_type == "ebt":
            return EBTScorer(
                full_cfg.scorer.ebt, memory=self.memory, container=self.container, logger=self.logger
            )
        raise ValueError(f"Unsupported scorer_type: {scorer_type}")

    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        cartridges = []
        total_docs = len(documents)

        self.report(
            {
                "event": "start",
                "step": "CartridgeAgent",
                "total_documents": total_docs,
            }
        )

        for idx, doc in enumerate(
            tqdm(documents, desc="🔨 Building Cartridges", unit="doc"), start=1
        ):
            try:
                self.report(
                    {
                        "event": "document_start",
                        "step": "CartridgeAgent",
                        "doc_number": idx,
                        "doc_id": doc.get("id"),
                        "title": doc.get("title", "")[:100],
                    }
                )

                goal = context.get("goal")

                existing = self.memory.cartridges.get_by_source_uri(
                    uri=str(doc.get("id")), source_type="document"
                )
                if existing:
                    if context.get("pipeline_run_id"):
                        existing.pipeline_run_id = context["pipeline_run_id"]
                    self.report(
                        {
                            "event": "cartridge_skipped_existing",
                            "step": "CartridgeAgent",
                            "doc_id": doc.get("id"),
                            "cartridge_id": existing.id,
                        }
                    )
                    cartridges.append(existing.to_dict())
                    continue

                # 1. Build CartridgeORM
                cartridge = self.builder.build(doc, goal=goal)
                if not cartridge:
                    self.report(
                        {
                            "event": "cartridge_skipped",
                            "step": "CartridgeAgent",
                            "doc_id": doc.get("id"),
                            "reason": "Builder returned None",
                        }
                    )
                    continue

                # 🔑 Ensure embedding_id is properly registered in document_embeddings
                scorable_text = cartridge.markdown_content or (
                    doc.get("summary") or doc.get("content")
                )
                scorable = Scorable(
                    id=cartridge.id,
                    text=scorable_text,
                    target_type=ScorableType.CARTRIDGE,
                )
                if scorable_text:
                    # Guarantee row exists in document_embeddings
                    doc_embedding_id = (
                        self.memory.scorable_embeddings.get_or_create(scorable)
                    )
                    cartridge.embedding_id = doc_embedding_id

                self.report(
                    {
                        "event": "cartridge_built",
                        "step": "CartridgeAgent",
                        "cartridge_id": cartridge.id,
                        "title": cartridge.title[:80]
                        if cartridge.title
                        else "Untitled",
                    }
                )

                # 2. Extract Triplets
                if not self.memory.cartridge_triples.has_triples(cartridge.id):
                    triplets = self.triplet_extractor.extract(
                        cartridge.sections, context
                    )
                    self.report(
                        {
                            "event": "triplets_extracted",
                            "step": "CartridgeAgent",
                            "cartridge_id": cartridge.id,
                            "triplet_count": len(triplets),
                            "examples": triplets[:3],
                        }
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
                            scorable = ScorableFactory.from_text(
                                f"({subj}, {pred}, {obj})", ScorableType.TRIPLE
                            )
                            score = self.scorer.score(
                                context=context,
                                scorable=scorable,
                                dimensions=self.dimensions,
                            )
                            context.setdefault("triplet_scores", []).append(
                                score
                            )

                # 3. Extract Theorems
                theorems = self.theorem_extractor.extract(
                    cartridge.sections, context
                )
                self.report(
                    {
                        "event": "theorems_extracted",
                        "step": "CartridgeAgent",
                        "cartridge_id": cartridge.id,
                        "theorem_count": len(theorems),
                        "examples": [t.statement[:80] for t in theorems[:2]],
                    }
                )

                for theorem in theorems:
                    # Save theorem first
                    self.memory.theorems.insert(theorem.to_dict())

                    # Create embedding in document_embeddings (doc_id = theorem.id, type = "theorem")
                    scorable = Scorable(
                        id=theorem.id,
                        text=theorem.statement,
                        target_type=ScorableType.THEOREM,
                    )
                    doc_embedding_id = (
                        self.memory.scorable_embeddings.get_or_create(scorable)
                    )

                    # Update theorem with FK to document_embeddings
                    theorem.embedding_id = doc_embedding_id

                    # Link to cartridge
                    theorem.cartridges.append(cartridge)

                    # Score theorem
                    scorable = ScorableFactory.from_text(
                        theorem.statement, ScorableType.THEOREM
                    )
                    theorem_score = self.scorer.score(
                        context={**context, "theorem": theorem.to_dict()},
                        scorable=scorable,
                        dimensions=self.dimensions,
                    )
                    context.setdefault("theorem_scores", []).append(
                        theorem_score
                    )

                # 4. Score Cartridge
                if self.score_cartridges:
                    scorable = ScorableFactory.from_text(
                        cartridge.markdown_content, ScorableType.CARTRIDGE
                    )
                    score = self.scorer.score(
                        context=context,
                        scorable=scorable,
                        dimensions=self.dimensions,
                    )
                    context.setdefault("cartridge_scores", []).append(score)
                    self.report(
                        {
                            "event": "cartridge_scored",
                            "step": "CartridgeAgent",
                            "cartridge_id": cartridge.id,
                            "score": getattr(score, "value", None),
                        }
                    )

                # 5. Assign Domains
                self.assign_domains(cartridge)

                self.report(
                    {
                        "event": "document_done",
                        "step": "CartridgeAgent",
                        "cartridge_id": cartridge.id,
                        "doc_number": idx,
                        "total_documents": total_docs,
                    }
                )

                cartridges.append(cartridge.to_dict())

            except Exception as e:
                self.report(
                    {
                        "event": "error",
                        "step": "CartridgeAgent",
                        "doc_number": idx,
                        "error": str(e),
                    }
                )

        self.report(
            {
                "event": "end",
                "step": "CartridgeAgent",
                "processed_cartridges": len(cartridges),
                "total_documents": total_docs,
            }
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
                {
                    "title": cartridge.title[:60],
                    "domain": domain,
                    "score": score,
                },
            )
