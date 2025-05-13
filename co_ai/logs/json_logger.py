# co_ai/logs/json_logger.py

import json
from datetime import datetime, timezone
from pathlib import Path


class JSONLogger:
    EVENT_ICONS = {
        # Pipeline-level
        "PipelineStart": "🔬",
        "PipelineSuccess": "✅",
        "PipelineError": "❌",
        "PipelineStageStart": "🚀",
        "PipelineStageEnd": "🏁",
        "PipelineStageSkipped": "⏭️",
        "PipelineIterationStart": "🔄",
        "PipelineIterationEnd": "🔚",
        "IterationStart": "🔄",
        "IterationEnd": "🔚",

        # Generation phase
        "GenerationAgent": "🧪",
        "GeneratedHypotheses": "💡",
        "RankingStored": "🗃️",
        "RankingUpdated": "🔁",
        "GeneratedReviews": "🧾",
        "TournamentCompleted": "🏆",
        # Prompt handling
        "PromptLogged": "🧾",

        # Reflection phase
        "ReflectionAgent": "🪞",
        "ReviewStored": "💬",
        "ReflectedHypotheses": "🔎",

        # Ranking phase
        "RankingAgent": "🏆",
        "RankedHypotheses": "🏅",

        # Evolution phase
        "EvolutionAgent": "🧬",
        "EvolvingTopHypotheses": "🔄",
        "EvolvedHypotheses": "🌱",
        "GraftingPair": "🌿",
        "EvolutionCompleted": "🦾",
        "EvolutionError": "⚠️",

        # Meta-review phase
        "MetaReviewAgent": "🧠",
        "MetaReviewSummary": "📘",
        "SummaryLogged": "📝",
        "RawMetaReviewOutput": "📜",

        # Hypothesis storage
        "HypothesisStored": "📥",

        # Reporting
        "ReportGenerated": "📊",

        # General
        "SupervisorInit": "🧑‍🏫",
        "LiteratureAgentInit": "📚",
        "LiteratureSearchSkipped": "⏭️",
        "LiteratureQueryFailed": "❓",
        "NoResultsFromWebSearch": "🚫",
        "DatabaseHypothesesMatched": "🔍",
        "ProximityGraphComputed": "🗺️",
        
        "Prompt": "📜",
        "ContextAfterStage": "🗃️",
        "debug": "🐞"
    }

    def __init__(self, log_path="logs/pipeline_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, data: dict):
        icon = self.EVENT_ICONS.get(event_type, "📦")  # Default icon for unknown types
        print(f"{icon} Logging event: {event_type} | {str(data)[:100]}")

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data
        }

        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                json.dump(log_entry, f, default=str)
                f.write("\n")
        except (TypeError, ValueError) as e:
            print(f"[Logger] ❌ Failed to serialize log entry: {e}")
            print(f"[Logger] 🚨 Problematic log data: {data}")
