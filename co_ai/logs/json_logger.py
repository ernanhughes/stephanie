import json
from datetime import datetime, timezone
from pathlib import Path


class JSONLogger:
    DEFAULT_ICON = "📦"

    EVENT_ICONS = {
        # General System & Supervisor
        "SupervisorInit": "🧑‍🏫",
        "AgentInitialized": "🛠️",
        "StoreRegistered": "✅",
        "ContextSaved": "💾",
        "ContextLoaded": "📂",
        "ContextYAMLDumpSaved": "📄",
        "ContextAfterStage": "🗃️",
        "debug": "🐞",

        # Pipeline execution
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
        "GenerationStart": "✨",
        "GenerationAgent": "🧪",
        "GeneratedHypotheses": "💡",

        # Ranking phase
        "RankingAgent": "🏆",
        "RankedHypotheses": "🏅",
        "RankingStored": "🗃️",
        "RankingUpdated": "🔁",

        # Review and reflection
        "ReviewAgent": "🧑‍⚖️",
        "ReviewStored": "💬",
        "MetaReviewAgent": "🧠",
        "MetaReviewSummary": "📘",
        "MetaReviewInput": "📉",
        "NotEnoughHypothesesForRanking": "⚠️",
        "GeneratedReviews": "🧾",
        "RawMetaReviewOutput": "📜",
        "SummaryLogged": "📝",
        "ReflectionAgent": "🪞",
        "ReflectionStart": "🤔",
        "ReflectionStored": "💾",

        # Evolution phase
        "EvolutionAgent": "🧬",
        "EvolvingTopHypotheses": "🔄",
        "EvolvedHypotheses": "🌱",
        "EvolvedParsedHypotheses": "🧬",
        "GraftingPair": "🌿",
        "EvolutionCompleted": "🦾",
        "EvolutionError": "⚠️",

        # Literature & research
        "LiteratureAgentInit": "📚",
        "LiteratureSearchSkipped": "⏭️",
        "LiteratureQueryFailed": "❓",
        "NoResultsFromWebSearch": "🚫",
        "DatabaseHypothesesMatched": "🔍",
        "ProximityGraphComputed": "🗺️",

        # Prompt handling
        "Prompt": "📜",
        "PromptLogged": "🧾",
        "ReportGenerated": "📊",
    }

    def __init__(self, log_path="logs/pipeline_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, data: dict):
        icon = self.EVENT_ICONS.get(event_type, self.DEFAULT_ICON)
        print(f"{icon} [{event_type}] {str(data)[:100]}")

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                json.dump(log_entry, f, default=str)
                f.write("\n")
        except (TypeError, ValueError) as e:
            print("❌ [Logger] Failed to serialize log entry.")
            print(f"🛠️  Event Type: {event_type}")
            print(f"🪵  Error: {e}")
            print(f"🧱  Data: {repr(data)[:200]}")
