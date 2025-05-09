# co_ai/agents/meta_review.py
from co_ai.agents.base import BaseAgent


class MetaReviewAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, input_data: dict) -> dict:
        """
        Takes evolved hypotheses + reviews and generates:
        - A structured meta-review
        - Feedback for future hypothesis generation and review
        """
        evolved_hypotheses = input_data.get("evolved", [])
        reviews = input_data.get("reviews", [])

        # Extract text from objects if needed
        hypothesis_texts = [
            h.text if hasattr(h, "text") else h for h in evolved_hypotheses
        ]
        review_texts = [
            r.review if hasattr(r, "review") else r for r in reviews
        ]

        self.log("Generating comprehensive meta-review from evolved hypotheses and reviews...")

        # Build the meta-review prompt
        prompt = self._build_meta_review_prompt(hypothesis_texts, review_texts)
        raw_response = self.call_llm(prompt)

        # Log the full response for traceability
        self.logger.log("RawMetaReviewOutput", {
            "raw_output": raw_response[:500] + "...",
            "hypothesis_count": len(hypothesis_texts),
            "review_count": len(review_texts)
        })

        # Store final summary
        self.memory.log_summary(raw_response.strip())

        # Optional: extract key insights or feedback from the response for use in next iteration
        feedback = self._extract_feedback_from_meta_review(raw_response)

        return {
            "meta_review": raw_response,
            "feedback": feedback
        }

    def _build_meta_review_prompt(self, hypotheses, reviews):
        evolved_hypotheses = ''.join(f'- {h}\n' for h in hypotheses)
        full_reviews = ''.join(f'- {r}\n' for r in reviews)
        return f"""
You are an expert in scientific research and meta-analysis.
Your task is to synthesize a comprehensive meta-review of the following information:

**Goal**: {self.cfg.get('goal', 'Unknown research goal')}

**Preferences**: {self.cfg.get('preferences', 'No preferences provided')}

**Instructions**:
1. Generate a structured meta-analysis report of the provided inputs.
2. Focus on identifying recurring critique points and common issues raised across reviews.
3. The generated meta-analysis should provide actionable insights for researchers developing future proposals.
4. Highlight strengths and weaknesses observed in multiple hypotheses.
5. Suggest refinements and future directions.

**Evolved Hypotheses**:
{evolved_hypotheses}

**Reviews of Hypotheses**:
{full_reviews}

Respond in the following format:
# Meta-Analysis Summary
[Summary of overall findings and trends]

# Recurring Critique Points
- Point 1
- Point 2

# Strengths Observed
- Strength 1
- Strength 2

# Recommended Improvements
- Improvement 1
- Improvement 2

# Strategic Research Directions
- Direction 1
- Direction 2
"""

    def _extract_feedback_from_meta_review(self, meta_review_text):
        """
        Parse meta-review to extract feedback that can be used in future iterations.
        This could be injected into prompts for GenerationAgent and ReflectionAgent.
        """
        try:
            sections = meta_review_text.split('# ')[1:]
            feedback = {
                "recurring_critiques": sections[1].split('\n')[1:-1],
                "recommended_improvements": sections[3].split('\n')[1:-1],
                "strategic_directions": sections[4].split('\n')[1:-1]
            }
            return feedback
        except Exception as e:
            self.logger.log("FeedbackExtractionFailed", {"error": str(e)})
            return {}