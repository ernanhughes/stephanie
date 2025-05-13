# co_ai/agents/meta_review.py

from co_ai.agents.base import BaseAgent


class MetaReviewAgent(BaseAgent):

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.strategy = cfg.get("strategy", "synthesis")
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
        evolved_hypotheses = context.get("evolved", [])
        reviewed = context.get("reviewed", [])
        reflections = context.get("reflections", [])
        ranked_hypotheses = context.get("ranked", [])
        strategic_directions = context.get("strategic_directions", [])
        db_matches = context.get("proximity", {}).get("database_matches", [])

        # Extract key themes from DB hypotheses
        db_themes = "\n".join(f"- {h['text'][:100]}" for h in db_matches)

        # Extract text if needed
        hypothesis_texts = [
            h.text if hasattr(h, "text") else h for h in evolved_hypotheses
        ]
        reflection_texts = [
            r.review if hasattr(r, "review") else r for r in reflections
        ]

        # Log inputs for traceability
        self.logger.log(
            "MetaReviewInput",
            {
                "hypothesis_count": len(hypothesis_texts),
                "review_count": len(reflection_texts),
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

        raw_response = self.call_llm(prompt).strip()

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

    def _build_meta_review_prompt(
        self,
        goal,
        hypotheses: list[str],
        reviews: list[str],
        directions: list[str],
        db_themes: list[str],
    ) -> str:
        """Build prompt using goal, preferences, and input data."""
        preferences = ", ".join(self.preferences)

        evolved_hypotheses = "\n".join(f"- {h}" for h in hypotheses)
        full_reviews = "\n".join(f"- {r}" for r in reviews)
        strategic_directions = "\n".join(f"- {d}" for d in directions)
        return self.prompt_template.format(
            goal=goal,
            preferences=preferences,
            hypotheses=hypotheses,
            evolved_hypotheses=evolved_hypotheses,
            reviews=full_reviews,
            db_themes=db_themes,
            instructions=strategic_directions or "No additional instructions",
        )

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
