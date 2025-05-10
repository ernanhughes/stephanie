# co_ai/logs/json_logger.py
import json
from datetime import datetime, timezone
from pathlib import Path


class JSONLogger:
    EVENT_ICONS = {
        "PipelineStart": "🔬",             # Start of pipeline execution
        "PipelineSuccess": "✅",           # Pipeline completed successfully
        "PipelineError": "❌",             # Pipeline encountered an error
        "PipelineStageStart": "🚀",        # A specific stage in the pipeline is starting
        "PipelineStageEnd": "🏁",          # A specific stage has completed
        "PipelineStageSkipped": "⏭️",      # A stage was skipped (e.g., disabled)
        "PipelineIterationStart": "🔄",    # Start of a loop iteration
        "PipelineIterationEnd": "🔚",      # End of a loop iteration
        "IterationStart": "🔄",            # Alias for per-agent iteration
        "IterationEnd": "🔚",

        # Generation phase
        "GenerationAgent": "🧪",           # The generation agent runs
        "GeneratedHypotheses": "💡",       # Output of generation (different from the agent)

        # Prompt handling
        "PromptLogged": "🧾",              # Log/save a prompt (📜 also works well)
        
        # Review phase
        "ReflectionAgent": "🪞",           # The reflection agent runs
        "ReviewStored": "💬",              # Review feedback stored (better match than 📥)
        "ReflectedHypotheses": "🔎",       # After reflection logic

        # Ranking
        "RankingAgent": "🏆",              # The ranking agent run s
        "RankedHypotheses": "🏅",          # After ranking

        # Evolution phase
        "EvolutionAgent": "🧬",
        "EvolvingTopHypotheses": "🔄",
        "EvolvedHypotheses": "🌱",         # Represents new/modified hypotheses
        "GraftingPair": "🌿",              # Represents a grafting pair

        # Meta review
        "MetaReviewAgent": "🧠",
        "MetaReviewSummary": "📘",         # Summary output
        "SummaryLogged": "📝",

        # Hypothesis storage
        "HypothesisStored": "📥",          # Store raw hypothesis

        # Other
        "Prompt": "📜",                  # General prompt
        "debug": "🐞"
    }
     
    def __init__(self, log_path="logs/pipeline_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, data: dict):
        icon = self.EVENT_ICONS.get(event_type, "📦")  # Default icon
        print(f"{icon} Logging event: {event_type} with data: {str(data)[:100]}")
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "data": data
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")
        except TypeError as e:
            print(f"[Logger] Skipping non-serializable log: {e}")
            print(f"[Logger] Problematic record: {log_entry}")

