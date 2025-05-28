# co_ai/logs/icons.py

def get_event_icon(event_type: str) -> str:
    """
    Get the icon associated with a specific event type.
    """
    return EVENT_ICONS.get(event_type, "❓")


EVENT_ICONS = {
    # ────────────────────────────────────────────
    # General System & Initialization
    # ────────────────────────────────────────────
    "AgentInitialized": "🛠️",
    "SharpeningResultSaved": "🪓",
    "ContextAfterStage": "🗃️",
    "ContextLoaded": "📂",
    "ContextSaved": "💾",
    "ContextYAMLDumpSaved": "📄",
    "StoreRegistered": "🛍️",
    "SupervisorInit": "🧑‍🏫",
    "debug": "🐞",
    "LiteratureQuery": "📚",
    "SearchQuery": "🔍",
    "SearchingWeb": "🌐",
    # ────────────────────────────────────────────
    # Pipeline Execution
    # ────────────────────────────────────────────
    "IterationEnd": "🔚",
    "IterationStart": "🔄",
    "PipelineError": "❌",
    "PipelineIterationEnd": "🔚",
    "PipelineIterationStart": "🔄",
    "PipelineStageEnd": "🏁",
    "PipelineStageSkipped": "⏭️",
    "PipelineStageStart": "🚀",
    "PipelineStart": "🔬",
    "PipelineSuccess": "✅",
    "GeneratedReflection": "🪞✨",
    "ReflectingOnHypothesis": "🤔💡",
    "JudgeRunStarted": "⚖️🚦",
    "JudgeStrategy": "🧠📐",
    "PairJudged": "⚔️⚖️",
    "JudgeRunCompleted": "✅⚖️",
    "HypothesisRanked": "🏆📊",
    # ────────────────────────────────────────────
    # Prompt Processing & Tuning
    # ────────────────────────────────────────────
    "BatchTunedPromptsComplete": "📊🧬",
    "ComparisonPromptConstructed": "🛠️",
    "ComparisonResponseReceived": "📩",
    "Prompt": "📜",
    "PromptAResponseGenerated": "🅰️",
    "PromptABResponseGenerated": "🅰️",
    "PromptBResponseGenerated": "🅱️",
    "PromptComparisonNoMatch": "❓",
    "PromptComparisonResult": "🏁",
    "PromptEvaluationFailed": "❌",
    "PromptFileNotFound": "🚫",
    "PromptLoadFailed": "❓",
    "PromptLogged": "🧾",
    "LLMCacheHit": "✅",
    "PromptParseFailed": "⚠️",
    "PromptQualityCompareStart": "⚖️",
    "PromptTuningCompleted": "🧪✨",
    "PromptTuningExamples": "📚",
    "PromptTuningSkipped": "⏭️",
    "PromptStored": "🗃🗃️",
    "TunedPromptGenerationFailed": "❌",
    "TunedPromptStored": "🗃️",
    "MRQTrainingStart": "🚀",
    "MRQTrainingEpoch": "📈",
    "MRQTrainingComplete": "🏁",
    "SQLQuery": "🧮",
    # ────────────────────────────────────────────
    # Hypotheses Generation
    # ────────────────────────────────────────────
    "GeneratedHypotheses": "💡",
    "GenerationAgent": "🧪",
    "GenerationStart": "✨",
    "HypothesisStoreFailed": "❌",
    "HypothesisStored": "💾",
    "MRQTraining": "📊🛠️",
    # ────────────────────────────────────────────
    # Hypotheses Evaluation & Ranking
    # ────────────────────────────────────────────
    "NotEnoughHypothesesForRanking": "⚠️",
    "RankedHypotheses": "🏅",
    "RankingAgent": "🏆",
    "AgentInit": "🤖",
    "EvaluatorInit": "📏",
    "RankingStored": "🗃️",
    "RankingUpdated": "🔁",
    "RubricClusteringStarted": "🧩",
    "RubricClusteringCompleted": "✅",
    "AgentRunStarted": "🚀",
    "PromptGenerated": "📝",
    "GenerationStarted": "🎯",
    "GenerationCompleted": "✅",
    "LLMJudgeResult": "⚖️",
    "EvaluationCompleted": "📊",
    "ClassificationStarted": "🔍",
    "RubricClassified": "📌",
    "ClassificationCompleted": "📋",
    "AgentRunCompleted": "🏁",
    "TournamentCompleted": "🏆",
    "SharpenedHypothesisSaved": "🪓💾",
    "SharpenedGoalSaved": "🪓🏆",
    "LiteratureSearchCompleted": "📚✅",
    "AgentRanSuccessfully": "✅",
    # ────────────────────────────────────────────
    # Review & Reflection
    # ────────────────────────────────────────────
    "MetaReviewAgent": "🧠",
    "MetaReviewInput": "📉",
    "SimilarHypothesesFound": "💡🔁",
    "PromptLookup": "🔍📜",
    "IdeaSharpenedAndSaved": "💡🪓💾",
    "MetaReviewSummary": "📘",
    "RawMetaReviewOutput": "📜",
    "HypothesisEloUpdated ": "📈",
    "RefinedSkipped": "⏭️",
    "RefinedUpdated": "🔄",
    "ScoreStored": "💾",
    "HypothesisInserted": "💡📥",
    "GeneralReasoningJudgement": "⚖️",
    "BatchProcessingStart": "📥",
    "RefinerEvaluationPromptGenerated": "💬",
    "RefinerEvaluationResponse": "📊",
    "RefinerError": "❌",
    "RefinerHypothesesExtracted": "🔍",
    "RefinerImprovementPromptLoaded": "📜",
    "RefinerNoHistoryFound": "🚫",
    "RefinerPromptGenerated": "💡",
    "RefinerResponseGenerated": "💬",
    "RefinerStart": "🔄",
    "ReflectionAgent": "🪞",
    "ReflectionStart": "🤔",
    "ReflectionStored": "💾",
    "ReviewAgent": "🧑‍⚖️",
    "ReviewStored": "💬",
    "SummaryLogged": "📝",
    "GeneratedReviews": "🧾",
    # ────────────────────────────────────────────
    # Evolution
    # ────────────────────────────────────────────
    "EvolutionAgent": "🧬",
    "EvolutionCompleted": "🦾",
    "EvolutionError": "⚠️",
    "EvolvedHypotheses": "🌱",
    "EvolvedParsedHypotheses": "🧬",
    "EvolvingTopHypotheses": "🔄",
    "GraftingPair": "🌿",
    # ────────────────────────────────────────────
    # Literature & Research
    # ────────────────────────────────────────────
    "DatabaseHypothesesMatched": "🔍",
    "LiteratureAgentInit": "📚",
    "LiteratureQueryFailed": "📚❌",
    "LiteratureSearchSkipped": "📚⏭️",
    "NoResultsFromWebSearch": "🌐🚫",
    "ProximityGraphComputed": "🗺️",
    "FetchHTMLFailed": "🌐❌",
    "SearchResult": "🔎📄",
    "LLMPromptGenerated_SearchQuery": "🧠🔍",
    "LLMResponseReceived_SearchQuery": "📥🔍",
    "LLMPromptGenerated_Summarize": "🧠📄",
    "LLMResponseReceived_Summarize": "📥📄",
    # ────────────────────────────────────────────
    # Reporting
    # ────────────────────────────────────────────
    "ReportGenerated": "📊",
    "GoalFetchedByText": "📄🔍",
    "GoalExists": "✔️📌",
    "ReflectionDeltaInserted": "🧩📈",
    "ReflectionDeltaLogged": "🔁📝OK ",
    "PipelineRunInserted": "🧱🚀",
}
