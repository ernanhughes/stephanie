# stephanie/logs/icons.py


def get_event_icon(event_type: str) -> str:
    """
    Get the icon associated with a specific event type.
    """
    return EVENT_ICONS.get(event_type, "❓")


EVENT_ICONS = {
    # General System & Initialization
    "AgentInitialized": "🛠️",
    "ContextLoaded": "📂",
    "ContextSaved": "💾",
    "ContextAfterStage": "🗃️",
    "ContextYAMLDumpSaved": "📄",
    "MRQTrainerTrainingComplete": "🏁",
    "MRQTrainerStart": "🚀" ,
    "MRQScoreBoundsUpdated": "📈", 
    "HypothesisJudged": "⚖️",  # Hypothesis judged
    "MRQModelInitializing": "🧠",  # Indicates model initialization
    "MRQDimensionEvaluated": "📏",  # Indicates dimension evaluation
    "HypothesisScored": "🏆",  # Hypothesis scored
    "EditGenerated": "✍️",  # Edit generated for hypothesis
    "TrainingDataProgress": "📊",
    "SymbolicAgentNewKey" : "🔑", 
    "ScoreDelta": "📈",  # Score delta calculated
    "debug": "🐞",
    "DocumentMRQModelMissing": "📄❌",  # Indicates missing model for MRQ scoring
    "DocumentMRQTunedScore": "📄🔧",  # Indicates tuned score for MRQ
    "DocumentMRQTunerMissing": "📄❌",  # Indicates missing tuner for MRQ
    "TunerSaved": "💾🛠️",  # Indicates tuner saved
    "ModelSaved": "💾📦",  # Indicates model saved
    "ArxivSearchStart": "🔍📚",  # Indicates start of Arxiv search
    "RegressionTunerFitted": "📈🛠️",  # Indicates regression tuner fitted
    "RegressionTunerTrainSingle": "🏋️‍♂️",  # Indicates single training of regression tuner
    "DocumentTrainingComplete": "🎉✅",  # Indicates document training completion
    "DocumentPairBuilderComplete": "📑✅",  # Indicates document pair builder completion
    "DocumentMRQTrainerEpoch": "📈",  # Indicates MRQ trainer epoch
    "DocumentMRQTrainingStart": "🚀📊",  # Indicates MRQ training start
    "DocumentTrainingProgress": "🔁📊",  # Indicates document training progress
    "DocumentMRQTrainDimension": "🧩📊",  # Indicates training of a specific dimension
    "DocumentPairBuilderProgress": "📊📄",  # Indicates progress in building document pairs
    "DocumentProfileFailed": "📉",         # Indicates profiling failed or dropped
    "DomainClassifierInit": "🧠",          # Classifier startup — cognitive/init
    "DomainConfigLoaded": "📚",            # YAML or config file loaded
    "SeedEmbeddingsPrepared": "🧬",        # Indicates seeds were embedded — DNA/metaphor
    "DocumentLoadFailed": "❌",            # General failure/loading errortted
    "ScoreSkipped": "📉⏭️",  # Scoring skipped due to existing score
    "GoalDomainAssigned": "🎯📚",  # Domain assigned to goal
    "DocumentsFiltered": "📑🔍",  # Documents filtered based on relevance
    "SurveyAgentSkipped": "📋⏭️",  # Survey step skipped
    "DocumentAlreadyExists": "📄✅",  # Document previously stored
    "DomainAssigned": "🏷️📚",  # Domain label assigned to doc
    "DomainUpserted": "📌🆕",  # Domain inserted or updated
    "ScoringPaper": "📝📊",  # Paper being scored
    "SectionInserted": "📂➕",  # New section added
    "PaperScoreSavedToMemory": "💾📈",  # Score persisted to memory/db
    "NoHypothesesInContext": "🤷‍♂️",
    "SimilarHypothesesFound": "♻️💭",
    "SectionDomainUpserted": "📂🏷️",  # Domain classification for section
    "StageContext": "🔧📝",
    "TrimmingSection": "✂️",
    "NoSymbolicPromptRulesApplied": "⏭️",
    "AgentInit": "🤖",
    "InvalidRuleMutation": "❌🧩",  # Indicates an invalid rule mutation attempt
    "NodeDebug": "🌲🔍",
    "NodeSummary": "🪵📋",
    "CorDimensionEvaluated": "📐✅",
    "PipelineMutationAgentInitialized": "🛠️🤖",  # Indicates a mutation agent is initialized
    "EvaluatorInit": "🧪",
    "RuleApplicationUpdated": "🧩",  # Suggests a symbolic piece being modified
    "MRQScoringComplete": "📈",  # Indicates successful scoring/completion
    "NoSymbolicAgentRulesApplied": "🚫",  # Signifies nothing matched/applied
    "RuleApplicationsScored": "🎯",  # Represents target scoring of rule usage
    "RuleApplicationCount": "🔢",  # Suggests counting or tracking quantity
    # Pipeline Execution
    "StoreRegistered": "🛒",
    "SupervisorInit": "🧑‍🏫",
    "PipelineStart": "🔬",
    "PipelineStageStart": "🚀",
    "PipelineStageEnd": "🏁",
    "PipelineStageSkipped": "⏭️",
    "PipelineIterationStart": "🔄",
    "PipelineIterationEnd": "🔁✅🔚",
    "PipelineRunInserted": "🔁🗃️",
    "PipelineSuccess": "✅",
    "PipelineError": "❌",
    "IterationStart": "🔄",
    "IterationEnd": "🔚",
    "AgentRunStarted": "🚀",
    "AgentRunCompleted": "🏁",
    "AgentRanSuccessfully": "✅",
    "TrainingEpoch": "🏋️‍♂️",
    "EarlyStopping": "⏹️⏳",
    "TrainingComplete": "🎉✅",
    "SymbolicAgentOverride": "🛠️",
    "RuleApplicationLogged": "⚖️📜",
    "ScoreParsed": "📊",
    "SymbolicRulesFound": "🧩",
    "DuplicateSymbolicRuleSkipped": "♻️",
    "SymbolicAgentRulesFound": "🔎",
    "PromptLookup": "📚",
    "PipelineJudgeAgentStart": "⚖️🚦",
    "HypothesesReceived": "🧠📥",
    "PromptLoaded": "📝",
    "JudgementReceived": "🗣️",
    "ScoreSaved": "💾",
    "SectionUpdated": "📝✨",
    "DocumentProfiled": "📄📊",
    "PipelineJudgeAgentEnd": "🛑⚖️",
    "PipelineScoreSummary": "📈🧮",
    "SymbolicPipelineSuggestion": "🧠💡",
    # Prompt Processing & Tuning
    "Prompt": "📜",
    "PromptGenerated": "📝",
    "PromptStored": "🗃🗃️",
    "PromptLogged": "🧾",
    "PromptFileNotFound": "🚫",
    "PromptLoadFailed": "❓",
    "PromptParseFailed": "⚠️",
    "PromptEvaluationFailed": "❌",
    "PromptComparisonResult": "🏁",
    "PromptComparisonNoMatch": "🧪📄❌",
    "PromptAResponseGenerated": "🅰️",
    "PromptBResponseGenerated": "🅱️",
    "PromptABResponseGenerated": "🅰️",
    "PromptQualityCompareStart": "⚖️",
    "PromptTuningCompleted": "🧪✨",
    "PromptTuningSkipped": "⏭️",
    "PromptTuningExamples": "📚",
    "TunedPromptStored": "🗃️",
    "TunedPromptGenerationFailed": "❌",
    "ComparisonPromptConstructed": "🛠️",
    "ComparisonResponseReceived": "📩",
    "LLMCacheHit": "✅",
    "MRQTrainingStart": "🚀",
    "MRQTrainingEpoch": "📈",
    "MRQTrainingComplete": "🏁",
    "MRQTraining": "📊🛠️",
    "MRQTrainingDataLoaded": "🧠📥",
    "MRQPipelineSuggested": "🧠🛤️",
    # goals
    "GoalCreated": "🎯💾",
    # Hypotheses Generation
    "GenerationAgent": "🧪",
    "GeneratedHypotheses": "💡",
    "GenerationStart": "✨",
    "GenerationStarted": "🎯",
    "DatasetLoading": "⏳📦",
    "DatasetLoaded": "✅📂",
    "DPOGenerated": "🔁🧠",
    "TrainingStarted": "🚀📊",
    "AdaptiveReasoningResponse": "🤖🪄",
    "GenerationCompleted": "✅",
    "HypothesisStored": "💾",
    "HypothesisStoreFailed": "❌",
    "HypothesisInserted": "💡📥",
    # Hypotheses Evaluation & Ranking
    "RankingAgent": "🏆",
    "RankedHypotheses": "🏅",
    "RankingStored": "🗃️",
    "RankingUpdated": "🔁",
    "GoalContextOverride": "🎯",
    "DimensionEvaluated": "📏",
    "ScoreLinkedToRuleApplications": "🔗",
    "ScoreSavedToMemory": "💾",
    "HypothesisScoreComputed": "🧮",
    "NotEnoughHypothesesForRanking": "⚠️",
    "LLMJudgeResult": "⚖️",
    "EvaluationCompleted": "📊",
    "ScoreComputed": "🧮📊✅",
    "ReviewScoreComputed": "🧑‍⚖️📊",
    "ReflectionScoreComputed": "🪞📊✅",
    "ScoreStored": "💾",
    # Evolution
    "EvolutionAgent": "🧬",
    "EvolvingTopHypotheses": "🔄",
    "EvolvedHypotheses": "🌱",
    "EvolvedParsedHypotheses": "🧬",
    "EvolutionCompleted": "🦾",
    "EvolutionError": "⚠️",
    "AdaptiveModeDecision": "🧠⚖️",
    "GraftingPair": "🌿",
    # Review & Reflection
    "ReflectionAgent": "🪞",
    "ReflectionStart": "🤔",
    "ReflectionStored": "💾",
    "ReflectionDeltaInserted": "🧩📈",
    "ReflectionDeltaLogged": "🔁📝",
    "MetaReviewAgent": "🧠",
    "MetaReviewInput": "📉",
    "MetaReviewSummary": "📘",
    "RawMetaReviewOutput": "📜",
    "GeneratedReviews": "🧾",
    "ReviewStored": "💬",
    "SharpenedHypothesisSaved": "🪓💾",
    "SharpenedGoalSaved": "🪓🏆",
    "IdeaSharpenedAndSaved": "💡🪓💾",
    "SummaryLogged": "📝",
    "RefinedSkipped": "⏭️",
    "RefinedUpdated": "🔄",
    "CoTGenerated": "🧠🔗📝",
    # Refiner Agent
    "RefinerStart": "🔄",
    "RefinerPromptGenerated": "💡",
    "RefinerEvaluationPromptGenerated": "💬",
    "RefinerResponseGenerated": "💬",
    "RefinerEvaluationResponse": "📊",
    "RefinerHypothesesExtracted": "🔍",
    "RefinerImprovementPromptLoaded": "📜",
    "RefinerNoHistoryFound": "🚫",
    "RefinerError": "❌",
    # Literature & Research
    "LiteratureAgentInit": "📚",
    "LiteratureQuery": "📚",
    "LiteratureQueryFailed": "📚❌",
    "LiteratureSearchCompleted": "📚✅",
    "LiteratureSearchSkipped": "📚⏭️",
    "NoResultsFromWebSearch": "🌐🚫",
    "ProximityGraphComputed": "🗺️",
    "SearchQuery": "🔍",
    "SearchingWeb": "🌐",
    "DatabaseHypothesesMatched": "🔍",
    "SearchResult": "🔎📄",
    "LLMPromptGenerated_SearchQuery": "🧠🔍",
    "LLMResponseReceived_SearchQuery": "📥🔍",
    "LLMPromptGenerated_Summarize": "🧠📄",
    "LLMResponseReceived_Summarize": "📥📄",
    # Reporting
    "ReportGenerated": "📊",
    "GoalFetchedByText": "📄🔍",
    "GoalExists": "✔️📌",
    "BatchProcessingStart": "📥",
    # Rubric Patterns
    "RubricPatternsStored": "📚🧩💾",
    "PatternStatsStored": "📊🧩💾",
    "RubricClassified": "📌",
    "PromptFileLoading": "🗂️📥",
    "PromptFileLoaded": "✅📄",
    "ProximityAnalysisScored": "🗺️📊",
    "DifficultySummary": "📋🧩",
    "SampleByDifficulty": "🧪📚",
    "PreferencePairSaveError": "❌💾",
    "TrainingError": "🔧💥",
    "ClassificationStarted": "🔍",
    "ClassificationCompleted": "📋",
    # SQL
    # ────────────────────────────────────────────
    "SQLQuery": "🧮",
}
