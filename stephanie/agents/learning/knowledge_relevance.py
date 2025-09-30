# stephanie/scoring/scorer/knowledge_relevance_scorer.py
from stephanie.data.score_result import ScoreResult
from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable

class KnowledgeRelevanceAgent(BaseAgent):
    """Verifies if knowledge is RELEVANT to the current task (not just topically similar)"""
    
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
    

    async def run(self, context: dict) -> dict:
        self.logger.log("KnowledgeRelevanceScoringStarted", {})
        
        documents  = context.get("documents", [])
        for document in documents:
            section = context.get("section")
            section = context.get("section")
            merged_context = {**context, } Yeah but I think it was section
            score_result = self.score(context, scorable)
            
            # Store result in context
            context["knowledge_relevance_score"] = {
                "score": score_result.score,
                "rationale": score_result.rationale,
                "source": score_result.source
            }
            
            self.logger.log("KnowledgeRelevanceScoringCompleted", {
                "score": score_result.score,
                "rationale": score_result.rationale
            })
        
        return context

    def score(self, context: dict, scorable: Scorable) -> ScoreResult:
        """Is this knowledge actually relevant to solving the current problem?"""
        prompt = self.prompt_loader.score_prompt(
            "knowledge_relevance", 
            self.cfg, 
            context
        )
        
        # Get LLM judgment on relevance
        response = self.call_llm(prompt, context=context)
        
        # Parse relevance score (0.0-1.0)
        try:
            score_match = re.search(r"relevance_score:\s*([0-9.]+)", response)
            if score_match:
                score = float(score_match.group(1))
                rationale_match = re.search(r"rationale:\s*(.*)", response, re.DOTALL)
                rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided"
                return ScoreResult(
                    dimension="relevance",
                    score=score,
                    rationale=rationale,
                    source="knowledge_relevance_scorer"
                )
        except Exception as e:
            self.logger.error(f"Relevance scoring failed: {str(e)}")
        
        # Fallback: use embedding similarity as proxy
        return ScoreResult(
            dimension="relevance",
            score=context.get("embedding_similarity", 0.0),
            rationale="Used embedding similarity as fallback",
            source="embedding_fallback"
        )