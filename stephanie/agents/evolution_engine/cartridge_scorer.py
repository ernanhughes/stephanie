from stephanie.agents.knowledge.paper_score import PaperScoreAgent
from stephanie.core.knowledge_cartridge import KnowledgeCartridge


class CartridgeScorer:
    def __init__(self, cfg, memory=None, logger=None):
        self.memory = memory
        self.logger = logger
        self.cfg = cfg
        self.scorer = PaperScoreAgent(cfg, memory, logger)

    def evaluate(self, cartridge: KnowledgeCartridge):
        """Evaluate cartridge quality using internal PaperScore logic"""
        document = {
            "id": hash(cartridge.signature),
            "title": f"Cartridge: {cartridge.goal}",
            "text": cartridge.to_markdown()
        }

        result = self.scorer.score_paper(document)

        # Assume dimensions are same as METRICS keys
        scores = {k: max(0.0, min(1.0, v)) for k, v in result.items()}

        # Fallback default weight schema
        weights = {
            "completeness": 0.2,
            "novelty": 0.3,
            "actionability": 0.25,
            "coherence": 0.15,
            "confidence": 0.1
        }

        overall = sum(scores.get(k, 0.0) * weights.get(k, 0.0) for k in weights)

        cartridge.quality_metrics = {
            "overall_score": overall,
            **scores
        }

        return cartridge
