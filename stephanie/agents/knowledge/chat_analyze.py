# stephanie/agents/chat_analyze_agent.py
"""
Chat Analysis Agent - LLM-powered evaluation of conversation quality.

This agent processes user-assistant conversation turns and evaluates them across
multiple reasoning dimensions using specialized LLM judges. It serves as the
foundational data preparation step for GAP analysis by creating high-quality
labeled datasets for training HRM and Tiny models.

Key Responsibilities:
- Extract conversation turns from memory or context
- Evaluate each turn across 5 reasoning dimensions using LLM judges
- Parse and validate judge responses with strict formatting
- Persist scores to evaluation store and chat memory
- Generate training targets for model supervision

The agent produces the labeled dataset that enables apples-to-apples
comparison between HRM and Tiny models in the GAP analysis pipeline.
"""
from __future__ import annotations

import logging
import re

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.scorable import Scorable, ScorableType
# stephanie/agents/chat_analyze_agent.py (inside run loop, after you get user_text/assistant_text)
from stephanie.utils.prompt_sanitizer import (clean_text_for_prompt,
                                              is_likely_code_goal)

log = logging.getLogger(__name__)


class ChatAnalyzeAgent(BaseAgent):
    """
    LLM-powered evaluator for conversation turn quality across reasoning dimensions.
    
    Processes user→assistant conversation pairs and scores them using dimension-specific
    LLM judges. Creates standardized evaluation records that serve as training targets
    for both HRM and Tiny models in the GAP analysis pipeline.
    
    Dimensions Evaluated:
    - reasoning: Logical structure and soundness
    - knowledge: Factual accuracy and specificity  
    - clarity: Organization and readability
    - faithfulness: Consistency with context
    - coverage: Completeness across facets
    
    Workflow:
    1. Retrieve conversation turns (from context or memory)
    2. For each turn, run LLM judges for all dimensions
    3. Parse and validate judge responses
    4. Persist scores to evaluation store
    5. Update chat memory with knowledge scores for GUI display
    """
    
    def __init__(self, cfg, memory, container, logger):
        """
        Initialize the chat analysis agent with configuration.
        
        Args:
            cfg: Agent configuration dictionary
            memory: Memory interface for data access
            container: Dependency injection container
            logger: Structured logging interface
            
        Configurable Parameters:
            limit: Maximum number of turns to process (default: 10000)
            dimensions: List of reasoning dimensions to evaluate
            force_rescore: Whether to re-score already evaluated turns
        """
        super().__init__(cfg, memory, container, logger)
        self.limit = cfg.get("limit", 10000)
        self.dimensions = cfg.get("dimensions", [
            "reasoning", "knowledge", "clarity", "faithfulness", "coverage"
        ])
        self.force_rescore = cfg.get("force_rescore", False)
        
        log.info(
            f"ChatAnalyzeAgent initialized with dimensions: {self.dimensions}, "
            f"limit: {self.limit}, force_rescore: {self.force_rescore}"
        )

    async def run(self, context: dict) -> dict:
        """
        Execute chat analysis pipeline across all conversation turns.
        
        Main processing workflow:
        1. Retrieve conversation turns (from context or memory)
        2. For each turn, run dimension-specific LLM judges
        3. Parse responses and validate scores
        4. Persist results to evaluation store
        5. Update chat memory for GUI display
        
        Args:
            context: Pipeline execution context containing:
                - chats: Optional pre-loaded conversation turns
                - pipeline_run_id: Unique identifier for this run
                - Additional pipeline-specific parameters
                
        Returns:
            Updated context with analysis results:
            {
                "analyzed_turns": [
                    {
                        "turn_id": str,
                        "score": float, 
                        "rationale": str
                    }
                ]
            }
            
        Raises:
            ParseError: If LLM response cannot be parsed
            MemoryError: If data persistence fails
            
        Logs:
            - "ChatAnalysisStarted": When analysis begins with turn count
            - "ChatAnalysisCompleted": On successful completion with statistics
            - "ChatAnalysisError": On individual turn processing failures
        """
        log.info("Starting chat analysis pipeline")
        
        # Determine data source: context-provided turns or memory retrieval
        if context.get("chats"):
            batch = context["chats"]
            log.debug(f"Using context-provided chats: {len(batch)} turns")
        else:
            log.debug(f"Retrieving turns from memory with limit: {self.limit}")
            batch = self.memory.chats.list_turns_with_texts(
                min_assistant_len=50,  # Filter very short responses
                limit=self.limit,
                order_desc=False,  # Process in chronological order
            )
            log.info(f"Retrieved {len(batch)} conversation turns from memory")

        # Get required services from container
        scoring = self.container.get("scoring")  # ScoringService for persistence
        prompt_service = self.container.get("prompt")  # PromptService for LLM calls
        pipeline_run_id = context.get("pipeline_run_id")
        
        log.debug(f"Pipeline run ID: {pipeline_run_id}")

        analyzed_turns = []
        processed_count = 0
        error_count = 0
        skip_count = 0

        # Process each conversation turn
        for row in batch:
            turn_id = row.get("id")
            assistant_message_id = row.get("assistant_message_id")
            
            log.debug(f"Processing turn {turn_id}, assistant message {assistant_message_id}")

            # Skip already analyzed turns unless force_rescore is enabled
            if not self.force_rescore and row.get("ai_score") is not None:
                log.debug(f"Skipping already analyzed turn {turn_id}")
                skip_count += 1
                continue

            # Validate turn has required text content
            user_text = row.get("user_text", "").strip()
            assistant_text = row.get("assistant_text", "").strip()
            if not user_text or not assistant_text:
                log.info(f"Skipping incomplete turn {turn_id}: missing user or assistant text")
                skip_count += 1
                continue

            log.debug(f"Turn {turn_id} - User: {user_text[:50]}... Assistant: {assistant_text[:50]}...")

            # Step 1: Create or retrieve goal from user text
            log.debug(f"Creating goal from user text for turn {turn_id}")
            goal = self.memory.goals.get_or_create(
                {
                    "goal_text": user_text,
                    "description": "Goal Created from Chat Analyze Agent",
                    "pipeline_run_id": pipeline_run_id,
                    "meta": {"source": "chat_analyze_agent"},
                }
            )
            log.debug(f"Goal created/retrieved: {goal.id}")

            clean_cfg = self.cfg.get("prompt_cleaning", {})
            code_goal = is_likely_code_goal(context.get("goal", {}).get("goal_text", ""))

            user_text_clean = clean_text_for_prompt(user_text, clean_cfg, is_code_goal=code_goal)
            assistant_text_clean = clean_text_for_prompt(assistant_text, clean_cfg, is_code_goal=code_goal)
            goal_text_clean = clean_text_for_prompt(goal.goal_text, clean_cfg, is_code_goal=code_goal)


            # Merge context for LLM prompt evaluation
            merged_context = {"user_text": user_text_clean, 
                              "assistant_text": assistant_text_clean,
                              "goal_text": goal_text_clean, 
                              **context}

            # Step 2: Evaluate across all reasoning dimensions
            dimension_results = {}
            for dimension in self.dimensions:
                log.debug(f"Evaluating dimension '{dimension}' for turn {turn_id}")
                
                try:
                    # Load dimension-specific prompt template
                    prompt = self.prompt_loader.from_file(f"{dimension}.txt", self.cfg, merged_context)
                    log.debug(f"Loaded prompt for dimension '{dimension}'")
                    
                    # Call LLM judge for this dimension
                    response = await prompt_service.run_prompt(prompt, merged_context)
                    log.debug(f"LLM response for '{dimension}': {response[:100]}...")
                    
                    # Parse and validate judge response
                    score = 0.0
                    rationale = ""
                    try:
                        parsed = parse_knowledge_judge_text(response)
                        score = parsed["score"]
                        rationale = parsed["rationale"]
                        log.debug(f"Parsed {dimension} score: {score}, rationale: {rationale[:50]}...")
                    except ParseError as e:
                        log.error(f"Parse error for turn {assistant_message_id}, dimension {dimension}: {e}")
                        error_count += 1
                        continue

                    # Step 3: Update chat memory with knowledge score for GUI display
                    if dimension == "knowledge":
                        log.debug(f"Updating chat memory with knowledge score for turn {turn_id}")
                        self.memory.chats.set_turn_ai_eval(
                            turn_id=turn_id,
                            score=score,
                            rationale=rationale,
                        )

                    # Create ScoreResult for this dimension
                    score_result = ScoreResult(
                        dimension=dimension,
                        score=score,
                        source="knowledge_llm",
                        rationale=rationale,
                        attributes={"raw_response": response},
                    )
                    dimension_results[dimension] = score_result
                    log.debug(f"Created ScoreResult for {dimension}")

                except Exception as e:
                    log.error(f"Unexpected error processing dimension '{dimension}' for turn {turn_id}: {e}")
                    error_count += 1
                    continue

            # Create ScoreBundle with all dimension results
            bundle = ScoreBundle(results=dimension_results)
            log.debug(f"Created ScoreBundle with {len(dimension_results)} dimensions")

            # Wrap as Scorable for persistence
            scorable = Scorable(
                id=assistant_message_id,
                text=assistant_text,
                target_type=ScorableType.CONVERSATION_TURN,
            )
            log.debug(f"Created Scorable for assistant message {assistant_message_id}")

            # Step 4: Persist evaluation results via ScoringService
            scored_context = {**context, "goal": goal.to_dict()}
            log.debug(f"Persisting ScoreBundle for turn {turn_id}")
            scoring.save_bundle(
                bundle=bundle,
                scorable=scorable,
                context=scored_context,
                cfg=self.cfg,
                agent_name=self.name,
                scorer_name="knowledge_llm",
                source="knowledge_llm",
                model_name="llm"
            )

            # Record successful processing
            analyzed_turns.append({
                "turn_id": assistant_message_id,
                "score": parsed["score"],  # Use last parsed score (knowledge dimension)
                "rationale": parsed["rationale"],
            })
            processed_count += 1
            log.debug(f"Successfully processed turn {turn_id}")

            # Progress logging for large batches
            if processed_count % 100 == 0:
                log.info(f"Processed {processed_count} turns...")

        # Final statistics logging
        log.info(
            f"Chat analysis completed: {processed_count} turns processed, "
            f"{skip_count} skipped, {error_count} errors"
        )

        context["analyzed_turns"] = analyzed_turns
        return context


class ParseError(ValueError):
    """
    Custom exception for LLM response parsing failures.
    
    Raised when the judge response cannot be parsed into the expected
    format containing 'rationale' and 'score' components.
    """
    pass


def parse_knowledge_judge_text(raw: str) -> dict:
    """
    Parse LLM judge response into structured score and rationale.
    
    Expected response format:
    ```
    rationale: <1-3 sentences explaining the score>
    score: <0-100>
    ```
    
    Args:
        raw: Raw LLM response text to parse
        
    Returns:
        Dictionary with parsed components:
        {
            "rationale": str,  # Explanation for the score
            "score": int       # Numeric score between 0-100
        }
        
    Raises:
        ParseError: If response is empty, malformed, or score out of range
        
    Example:
        >>> parse_knowledge_judge_text("rationale: Good logical structure.\\nscore: 85")
        {'rationale': 'Good logical structure.', 'score': 85}
    """
    log.debug(f"Parsing judge response: {raw[:100]}...")
    
    if not raw or not raw.strip():
        log.warning("Empty judge response received")
        raise ParseError("Empty response")

    # Clean response text - remove markdown code blocks if present
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text, flags=re.DOTALL)
        text = re.sub(r"\n```$", "", text, flags=re.DOTALL)
        log.debug("Removed markdown code blocks from response")

    # Normalize line endings
    text = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    log.debug(f"Normalized response text: {text[:100]}...")

    # Primary parsing pattern: "rationale: ... score: ..."
    m = re.search(
        r"(?is)rationale\s*:\s*(.*?)\n\s*score\s*:\s*([0-9]{1,3})\s*$", text
    )
    if m:
        rationale, score = m.group(1).strip(), int(m.group(2))
        log.debug(f"Successfully parsed with primary pattern: score={score}")
    else:
        # Fallback: look for score anywhere in text
        ms = re.search(r"(?im)score\s*:\s*([0-9]{1,3})", text)
        if not ms:
            log.warning(f"Could not find score in response: {text[:100]}...")
            
        score = int(ms.group(1))
        rationale = text[: ms.start()].strip()
        log.debug(f"Used fallback parsing: score={score}")

    # Validate score range
    if not (0 <= score <= 100):
        log.error(f"Score out of range: {score}")
        raise ParseError(f"Score out of range: {score}")

    log.debug(f"Successfully parsed: score={score}, rationale={rationale[:50]}...")
    return {"rationale": rationale, "score": score}