# co_ai/agents/meta_review.py

from co_ai.agents.base import BaseAgent
from co_ai.constants import HYPOTHESES, EVOLVED, REVIEWED, STRATEGY, REFLECTION, RANKING


class MetaReviewAgent(BaseAgent):

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        # Load preferences from config or default list
        self.preferences = cfg.get("preferences", ["goal_consistency", "plausibility"])

    async def run(self, context: dict) -> dict:
        """
        Synthesize insights from reviews and evolved hypotheses.

        Takes:
        - Evolved hypotheses
        - Reviews (from ReflectionAgent)
        - Rankings (from RankingAgent)
        - Strategic directions (from ProximityAgent)

        Returns enriched context with:
        - meta_review summary
        - extracted feedback for future generations
        """

        # Get inputs from context
        evolved_hypotheses = context.get(EVOLVED, [])
        if len(evolved_hypotheses) == 0:
            evolved_hypotheses = context.get(HYPOTHESES, [])
        reviewed = context.get(REVIEWED, [])
        reflections = context.get(REFLECTION, [])
        ranked_hypotheses = context.get(RANKING, [])
        strategic_directions = context.get("strategic_directions", [])
        db_matches = context.get("proximity", {}).get("database_matches", [])

        # Extract key themes from DB hypotheses
        db_themes = "\n".join(f"- {h['text'][:100]}" for h in db_matches)

        # Extract text if needed
        hypothesis_texts = [
            h.text if hasattr(h, "text") else h for h in evolved_hypotheses
        ]
        reflection_texts = [
            r.review if hasattr(r, "reflection") else r for r in reflections
        ]
        reviewed_texts = [
            r.review if hasattr(r, "text") else r for r in reviewed
        ]

        # Log inputs for traceability
        self.logger.log(
            "MetaReviewInput",
            {
                "hypothesis_count": len(hypothesis_texts),
                "review_count": len(reviewed_texts),
                "ranked_count": len(ranked_hypotheses),
                "strategic_directions": strategic_directions,
            },
        )

        merged = {
            **context,
            **{
                "evolved_hypotheses": evolved_hypotheses,
                "reviews": reviewed,
                "db_themes": db_themes,
            },
        }
        prompt = self.prompt_loader.load_prompt(self.cfg, merged)

        raw_response = self.call_llm(prompt)

        # Store full response for debugging
        self.logger.log(
            "RawMetaReviewOutput", {"raw_output": raw_response[:500] + "..."}
        )

        # Add to context
        context["meta_review"] = raw_response

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
            self.logger.log("FeedbackExtractionFailed", {"error": str(e)})
            return {}
