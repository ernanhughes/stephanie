# File: stephanie/agents/knowledge/learnable_idea_extractor.py



from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.domain_classifier import DomainClassifier
from stephanie.builders.belief_cartridge_builder import BeliefCartridgeBuilder
from stephanie.scoring.mrq.mrq_scorer import \
    MRQScorer  # or wherever your scorer lives
from stephanie.utils.idea_parser import IdeaParser


class LearnableIdeaExtractorAgent(BaseAgent):
    """
    Extracts learnable, actionable ideas from research papers and encodes them
    into belief cartridges — structured cognitive scaffolds that Stephanie can use,
    test, and refine over time.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

        # Components from config
        self.idea_parser = IdeaParser(cfg, logger=logger)
        self.idea_parser.prompt_loader = self.prompt_loader
        self.idea_parser.call_llm = self.call_llm
        self.idea_parser.memory = self.memory
        
        self.cartridge_builder = BeliefCartridgeBuilder(cfg, memory=memory, logger=logger)
        self.domain_classifier = DomainClassifier(
            memory=memory,
            logger=logger,
            config_path=cfg.get("domain_seed_config_path", "config/domain/seeds.yaml"),
        )
        self.mrq_scorer = MRQScorer(cfg, memory=memory, logger=logger)  # Replace with your actual scorer

        # Settings
        self.top_k_domains = cfg.get("top_k_domains", 3)
        self.min_classification_score = cfg.get("min_classification_score", 0.6)
        self.max_ideas_per_paper = cfg.get("max_ideas_per_paper", 5)

    async def run(self, context: dict) -> dict:
        """
        Main pipeline:
        1. Load documents from context
        2. Parse each into structured ideas
        3. Classify domains
        4. Encode as belief cartridges
        5. Store results
        """
        documents = context.get(self.input_key, [])
        goal = context.get("goal", {})
        goal_id = goal.get("id")

        cartridges = []

        for doc in documents:
            try:
                doc_id = doc["id"]
                title = doc.get("title")
                summary = doc.get("summary")
                text = doc.get("content", doc.get("text", ""))

                # Skip if already processed
                if self._is_already_processed(doc_id):
                    self.logger.log("DocumentAlreadyProcessed", {"doc_id": doc_id})
                    continue

                # Step 1: Parse paper sections
                parsed_sections = await self._parse_document_sections(text, title, context)
                if not parsed_sections:
                    continue

                # Step 2: Extract candidate ideas
                raw_ideas = self.idea_parser.parse(parsed_sections)
                if not raw_ideas:
                    continue

                # Step 3: Score and rank ideas
                scored_ideas = self._score_ideas(raw_ideas, goal, context)
                top_ideas = sorted(scored_ideas, key=lambda x: x["score"], reverse=True)[
                    : self.max_ideas_per_paper
                ]

                # Step 4: Build cartridges
                for idea in top_ideas:
                    cartridge = self.cartridge_builder.build(
                        title=f"{title}: {idea['title']}",
                        content=idea["description"],
                        source_type="paper",
                        source_id=doc_id,
                        domain_tags=idea.get("tags", []),
                        metadata={
                            "source_paper": title,
                            "source_url": doc.get("url"),
                            "abstract": summary,
                            "scoring": idea.get("scores", {}),
                            "integration_hint": idea.get("hint", ""),
                            "type": idea.get("type", "general"),
                            "goal_id": goal_id,
                        },
                    )
                    if cartridge:
                        cartridges.append(cartridge.to_dict())

                # Step 5: Classify and store domains
                self._classify_and_store_domains(text, doc_id)

            except Exception as e:
                self.logger.log("IdeaExtractionFailed", {"error": str(e), "doc_id": doc_id})

        context[self.output_key] = cartridges
        context["cartridge_ids"] = [c.get("id") for c in cartridges]
        return context

    def _is_already_processed(self, doc_id: int) -> bool:
        """Check if this document has already had ideas extracted."""
        return self.memory.belief_cartridges.exists_by_source(doc_id)

    async def _parse_document_sections(self, text: str, title: str, context: dict) -> dict:
        """Use prompt + parser to extract structured paper content."""
        try:
            # Could also use DocumentProfilerAgent output
            parsed = self.idea_parser.parse(text, title, context)
            return parsed
        except Exception as e:
            self.logger.log("SectionParsingFailed", {"error": str(e)})
            return {}

    def _score_ideas(self, ideas: list, goal: dict, context: dict) -> list:
        """Apply GILD/MRQ-style scoring to prioritize best ideas."""
        scored = []
        for idea in ideas:
            merged = {"idea_description": idea["description"], **context}
            score_bundle = self.mrq_scorer.score(merged)
            idea["score"] = score_bundle.overall_score()
            idea["scores"] = score_bundle.to_dict()
            scored.append(idea)
        return scored

    def _classify_and_store_domains(self, text: str, doc_id: int):
        """Classify the paper and assign domains."""
        results = self.domain_classifier.classify(text, self.top_k_domains, self.min_classification_score)
        for domain, score in results:
            self.memory.document_domains.insert(
                {
                    "document_id": doc_id,
                    "domain": domain,
                    "score": score,
                }
            )
            self.logger.log("DomainAssigned", {"domain": domain, "score": score})