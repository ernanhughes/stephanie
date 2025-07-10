# stephanie/agents/meta_review.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import (DATABASE_MATCHES, EVOLVED, HYPOTHESES,
                                 PROXIMITY, RANKING, REFLECTION, REVIEW)


class MetaReviewAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        # Load preferences from config or default list
        self.preferences = cfg.get("preferences", ["goal_consistency", "plausibility"])

    async def run(self, context: dict) -> dict:
        """
        Synthesize insights from reviews and evolved hypotheses.

        Takes:Well I'd like to understand this what ok I've got to review the reflection ground
        - Evolved hypotheses
        - Reflections (from ReflectionAgent)
        - Reviews (from ReviewAgent)
        - Rankings (from RankingAgent)
        - Strategic directions (from ProximityAgent)

        Returns enriched context with:
        - meta_review summary
        - extracted feedback for future generations
        """

        # Get inputs from context
        evolved = context.get(EVOLVED, [])
        if len(evolved) == 0:
            evolved = context.get(HYPOTHESES, [])
        review = context.get(REVIEW, [])
        reflection = context.get(REFLECTION, [])
        ranking = context.get(RANKING, [])
        strategic_directions = context.get("strategic_directions", [])
        db_matches = context.get(PROXIMITY, {}).get(DATABASE_MATCHES, [])

        # Extract key themes from DB hypotheses
        db_themes = "\n".join(f"- {h[:100]}" for h in db_matches)

        # Extract text if needed
        hypothesis_texts = [h.text if hasattr(h, "text") else h for h in evolved]
        reflection_texts = [
            r.review if hasattr(r, "reflection") else r for r in reflection
        ]
        reviewed_texts = [r.review if hasattr(r, "text") else r for r in review]

        # Log inputs for traceability
        self.logger.log(
            "MetaReviewInput",
            {
                "hypothesis_count": len(hypothesis_texts),
                "evolved_count": len(evolved),
                "review_count": len(reviewed_texts),
                "ranked_count": len(ranking),
                "reflection_count": len(reflection_texts),
                "strategic_directions_count": len(strategic_directions),
                "strategic_directions": strategic_directions,
            },
        )

        merged = {
            **context,
            **{
                EVOLVED: evolved,
                REVIEW: review,
                RANKING: ranking,
                "db_themes": db_themes,
            },
        }
        prompt = self.prompt_loader.load_prompt(self.cfg, merged)

        raw_response = self.call_llm(prompt, context)

        # Store full response for debugging
        self.logger.log(
            "RawMetaReviewOutput", {"raw_output": raw_response[:500] + "..."}
        )

        # Add to context
        context[self.output_key] = raw_response

        # Extract structured feedback
        feedback = self._extract_feedback_from_meta_review(raw_response)
        context["feedback"] = feedback

        return context

    def _extract_feedback_from_meta_review(self, meta_review_text):
        try:
            sections = {}
            current_section = None

            for line in meta_review_text.split("\n"):
                line = line.strip()
                if line.startswith("# Meta-Analysis Summary"):
                    current_section = "summary"
                    sections[current_section] = []
                elif line.startswith("# Recurring Critique Points"):
                    current_section = "recurrent_critiques"
                    sections[current_section] = []
                elif line.startswith("# Strengths Observed"):
                    current_section = "strengths"
                    sections[current_section] = []
                elif line.startswith("# Recommended Improvements"):
                    current_section = "improvements"
                    sections[current_section] = []
                elif line.startswith("# Strategic Research Directions"):
                    current_section = "strategic_directions"
                    sections[current_section] = []
                elif line.startswith("- "):
                    if current_section not in sections:
                        sections[current_section] = []
                    sections[current_section].append(line[2:].strip())

            return {
                "summary": "\n".join(sections.get("summary", [])),
                "recurring_critiques": sections.get("recurrent_critiques", []),
                "strengths_observed": sections.get("strengths", []),
                "recommended_improvements": sections.get("improvements", []),
                "strategic_directions": sections.get("strategic_directions", []),
            }

        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            self.logger.log("FeedbackExtractionFailed", {"error": str(e)})
            return {}
