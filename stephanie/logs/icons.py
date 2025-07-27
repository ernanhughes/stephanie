# stephanie/logs/icons.py
def get_event_icon(event_type: str) -> str:
    """Get the icon associated with a specific event type."""
    return EVENT_ICONS.get(event_type, "❓")  # Default: question mark


# ========================
# SYSTEM & INITIALIZATION
# ========================
SYSTEM_INIT = {
    "UncertaintyEstimated": "🔍",  # Uncertainty estimation
    "EBTEnergyCalculated": "⚡",  # EBT energy calculation
    "ScoringPolicyCompleted": "✅",  # Scoring policy completed
    "AllEBTModelsLoaded": "📦✅",  # All EBT models loaded

    "SupervisorInit": "👨‍🏫",  # Supervisor initialization
    "DocumentLLMInferenceCompleted": "📄✅",  # Document LLM inference completed
    "DocumentEmbeddingsBackfilled": "📄🌱",  # Document embeddings backfilled
    "AgentInitialized": "ᯓ★",  # Agent initialization
    "AgentInit": "🤖",  # Agent startup
    "ContextLoaded": "📂",  # Context loaded
    "ContextSaved": "💾",  # Context saved
    "SupervisorComponentsRegistered": "👨‍🏫",  # Supervisor registration
    "DomainClassifierInit": "🏷️🧠",  # Domain classifier init
    "DomainConfigLoaded": "🏷️📋",  # Domain config loaded
    "SeedEmbeddingsPrepared": "🌱🧬",  # Seed embeddings prepared
}

AGENT_EVENTS = {
    "GILDTrainerAgentInitialized": "📊🤖",  # GILD trainer agent initialized
    "MRQInferenceAgentInitialized": "📊🤖",  # MRQ inference agent initialized
    "AgentInitialized": "ᯓ★",  # Agent initialization
    "AgentInit": "🤖",  # Agent startup
    "MRQTrainerAgentInitialized": "📊🤖",  # MRQ trainer agent initialized
}

# =================
# KNOWLEDGE STORAGE
# =================
KNOWLEDGE_OPS = {
    "GoalContextOverride": "🎯🔄",  # Goal context override
    "MgrScoreParseError": "📊❌",  # Scoring parse error
    "SymbolicRulesFound": "🧩🔍",  # Symbolic rules found
    "MRQTrainingDataLoaded": "📊📥",  # MRQ training data loaded
    "DuplicateSymbolicRuleSkipped": "🚫🧩",  # Duplicate symbolic rule skipped
    "EvolvedParsedHypotheses": "🌱💡",  # Evolved hypotheses parsed
    "EvolutionCompleted": "🌱✅",  # Evolution completed
    "GenerationStart": "📝▶️",  # Generation started
    "MetaReviewInput": "📝📋",  # Meta-review input
    "RawMetaReviewOutput": "📝📄",  # Raw meta-review output
    "NotEnoughHypothesesForRanking": "❌💡",  # Not enough hypotheses for ranking
    "PromptLookup": "🔍📝",  # Prompt lookup
    "RubricClassified": "🏷️📄",  # Rubric classified
    "PromptGenerated": "📝✨",  # Prompt generated
    "PatternStatsStored": "📊💾",  # Pattern stats stored
    "GenerationCompleted": "✅",  # Generation completed
    "LLMJudgeResults": "📝⚖️",  # LLM judge results
    "EvaluationCompleted": "✅",  # Evaluation completed
    "RubricPatternsStored": "📊💾",  # Rubric patterns stored
    "GenerationStarted": "📝▶️",  # Generation started
    "EvaluatorInit": "🧑‍🏫",  # Evaluator initialization
    "MRQInferenceAgentInitialized": "📊🤖",  # MRQ inference agent initialized
    "EBTBufferLoaded": "🧪📦",  # EBT buffer loaded
    "EBTInferenceCompleted": "🧪✅",  # EBT inference complete I know what just before I refuse this I just want to kick it off just in case it's not being kicked off seems like it's gone d
    "MemCubeSaved": "💾📦✅",  # MemCube saved
    "DocumentRefinedWithEBT": "📄🔄",  # Document refined with EBT
    "EBTExampleAdded": "🧪➕",  # EBT example added
    "MRQScoresCalculated": "📊✅",  # MRQ scores calculated
    "ScoringEvent": "📊",  # Scoring event
    "DocumentEBTTrainingStart": "🧪▶️ I",  # Training started for a dimension
    "DocumentEBTEpoch": "📊🔁",  # Epoch completed during training
    "DocumentEBTModelSaved": "💾✅",  # Model saved after training
    "DocumentEBTInferenceAgentInitialized": "🧠🚦",  # Inference agent initialized
    "LoadingEBTModel": "📥📦",  # Loading EBT model from disk
    "EBTScoringStarted": "📝⚙️",  # Scoring started for a document
    "EBTScoreComputed": "📈📍",  # Score computed for a dimension
    "EBTScoringFinished": "🏁📘",  # Scoring completed for a document
    "TripletsRetrievedByDomain": "🔗🏷️",  # Triplets retrieved by domain
    "DomainAssigned": "🏷️✅",  # Domain assigned
    "MRQTunedScore": "🧠📊",  # MRQ tuned score
    "CartridgeCreated": "💾📦",  # Cartridge created
    "CartridgeAlreadyExists": "💾✅",  # Cartridge exists check
    "TriplesAlreadyExist": "🔗✅",  # Triples exist check
    "DimensionEvaluated": "📏✅",  # Dimension evaluated All right thanks Dan Dance Engineer the dance
    "CartridgeDomainInserted": "💾🏷️",  # Cartridge domain added
    "TripleInserted": "🔗",  # Triple inserted
    "SectionInserted": "📂➕",  # Section inserted
    "TripletScored": "🔗📊",  # Triplet scored
    "SectionDomainInserted": "📂🏷️",  # Section domain added
    "SectionDomainUpserted": "📂🔄",  # Section domain updated
    "DocumentAlreadyExists": "📄✅",  # Document exists check
    "DomainUpserted": "🏷️🔄",  # Domain updated
    "ContextYAMLDumpSaved": "📄💾",  # YAML context saved
    "CartridgeProcessingStarted": "💾▶️",  # Cartridge processing started
    "CartridgeDocumentProcessingStarted": "💾📄▶️",  # Document processing started
    "CartridgeBuilt": "💾✅",  # Cartridge built
    "TripletsExtractionCompleted": "🏁",  # Triplets extracted
    "DatabaseHypothesesMatched": "📊✅",  # Hypotheses matched in DB
    "TripletsInserted": "🔗💾",  # Triplets inserted
    "TheoremExtracted": "📜✅",  # Theorem extracted
    "TheoremsExtractionCompleted": "🏁",  # Theorems extracted
    "DocumentProfiled": "📄📋",  # Document profiled
    "MaxSectionsReached": "📄⏭️",  # Max sections reached
    "ItemScored": "📊✅",  # Item scored
    "CartridgeScored": "💾📊",  # Cartridge scored
    "DomainAssignmentSkipped": "🏷️⏭️",  # Domain assignment skipped
    "CartridgeProcessingCompleted": "🏁",  # Cartridge processing completed
    "DocumentAlreadyProfiled": "📄✅",  # Document already profiled
    "StoreRegistered": "🛒",  # Store registered
    "PreferencePairBuilder": "💾▶️",  # Preference pair builder started
}

# =================
# PIPELINE CONTROL
# =================
PIPELINE_FLOW = {
    "PipelineSummaryPrinted": "🖇️📄",  # Pipeline summary printed
    "PipelineStageInserted": "🖇️➕",  # Stage inserted
    "PipelineStart": "🖇️▶️",  # Pipeline started
    "PipelineStageStart": "🖇️⏩",  # Stage started
    "PipelineStageEnd": "🖇️🔚",  # Stage completed
    "PipelineStageSkipped": "🖇️⏭️",  # Stage skipped
    "PipelineIterationStart": "🖇️🔄",  # Iteration started
    "PipelineIterationEnd": "🖇️🔚",  # Iteration completed
    "PipelineSuccess": "🖇️✅",  # Pipeline succeeded
    "PipelineRunInserted": "🖇️💾",  # Pipeline run saved
    "AgentRunStarted": "🤖▶️",  # Agent run started
    "AgentRunCompleted": "🤖⏹️",  # Agent run completed
    "AgentRanSuccessfully": "🤖✅",  # Agent succeeded
    "PipelineJudgeAgentEnd": "⚖️🔚",  # Judge agent completed
    "MRQPipelineSuggested": "🧠💡",  # MRQ pipeline suggested
    "PipelineStageFailed": "🖇️⚠️❌💢",  # Pipeline stage failed
    "PipelineScoreSummary": "🖇️📊🧾",  # Pipeline score summary
    "PipelineError": "🖇️❌",  # Pipeline error
}

# =====================
# SCORING & EVALUATION
# =====================
SCORING = {
    "LLMJudgeResults": "📝⚖️",  # LLM judge results
    "DocumentScoresAlreadyExist": "📄✅",  # Document scores already exist
    "LLMJudgeScorerDimension": "📝📊",  # LLM judge scoring dimension
    "DocumentScored": "📊✅",  # Document scored
    "HypothesisScored": "💡📊",  # Hypothesis scored
    "ScoreComputed": "🧮✅",  # Score computed
    "ScoreParsed": "📝📊",  # Score parsed
    "ScoreSaved": "💾📊",  # Score saved
    "ScoreSavedToMemory": "🧠💾",  # Score saved to memory
    "ScoreSkipped": "⏭️📊",  # Scoring skipped
    "ScoreDelta": "📈",  # Score delta
    "ScoreCacheHit": "💾✅",  # Score cache hit
    "MRQScoreBoundsUpdated": "📈🔄",  # MRQ bounds updated
    "MRQDimensionEvaluated": "📏🧠",  # Dimension evaluated
    "CorDimensionEvaluated": "📐✅",  # COR dimension evaluated
    "MRQScoringComplete": "📊✅",  # MRQ scoring complete
    "MRQScoreComputed": "📐✅",  # MRQ score computed
    "ReportGenerated": "📄✅",  # Report generated
    "MRQScoringFinished": "📊🏁",  # MRQ scoring finished
    "MRQScoringStarted": "📊▶️",  # MRQ scoring started
    "AllMRQModelsLoaded": "📊✅",  # All MRQ models loaded
    "LoadingModelPaths": "📂🔄",  # Model paths loading
    "DocumentMRQInferenceAgentInitialized": "📊🤖",  # Document MRQ inference agent initialized
    "KnowledgeDBLoaded": "📚✅",  # Knowledge database loaded
    "DocumentModelSaved": "📄💾",  # Document model saved
    "ModelSaved": "💾✅",  # Model saved
    "EncoderSaved": "📄💾",  # Encoder saved
    "MRQInferenceCompleted": "📊✅",  # MRQ inference completed
    "SVMScoringFinished": "📊🏁",  # SVM scoring finished
    "SVMScoringStarted": "📊▶️",  # SVM scoring started
    "SVMScoreComputed": "📊✅",  # SVM score computed
    "PolicyAnalysis": "📊🔍",  # Policy analysis
    "NoSICQLDataFound": "🚫📊",  # No SI-CQL data found

}

# =====================
# REASONING & ANALYSIS
# =====================
REASONING = {
    "KeywordsExtracted": "🔑",  # Keywords extracted
    "ProximityAnalysisScored": "📌🗺️",  # Proximity analysis
    "ProximityGraphComputed": "📊🌐",  # Proximity graph
    "HypothesisJudged": "⚖️",  # Hypothesis judged
    "SymbolicAgentRulesFound": "🧩🔍",  # Symbolic rules found
    "SymbolicAgentOverride": "🧠🔄",  # Symbolic override
    "RuleApplicationLogged": "🧾🧩",  # Rule application logged
    "RuleApplicationUpdated": "🔄🧩",  # Rule application updated
    "RuleApplicationCount": "🔢🧩",  # Rule applications counted
    "RuleApplicationsScored": "🎯🧩",  # Rule applications scored
    "NoSymbolicAgentRulesApplied": "🚫🧩",  # No rules applied
    "SymbolicAgentNewKey": "🔑🧠",  # New symbolic key
    "SymbolicPipelineSuggestion": "💡🧩",  # Symbolic pipeline suggestion
}

# =====================
# TRAINING & MODEL OPS
# =====================
TRAINING = {
    "MRQTrainerEpoch": "🏋️",  # MRQ training epoch
    "MRQTrainerStart": "🚀🧠",  # MRQ training started
    "MRQTrainerTrainingComplete": "🎓🧠",  # MRQ training completed
    "MRQModelInitializing": "🧠⚙️",  # MRQ model initializing
    "TrainingEpoch": "🏋️",  # Training epoch
    "TrainingComplete": "🎓✅",  # Training completed
    "TrainingDataProgress": "📈🔄",  # Training data progress
    "RegressionTunerFitted": "📈🔧",  # Regression tuner fitted
    "RegressionTunerTrainSingle": "🔧▶️",  # Tuner training
    "DocumentTrainingComplete": "📄🎓",  # Document training completed
    "DocumentPairBuilderComplete": "📑✅",  # Document pairs built
    "DocumentMRQTrainerEpoch": "📊🏋️",  # Document MRQ epoch
    "DocumentMRQTrainingStart": "🚀📊",  # Document MRQ training start
    "DocumentTrainingProgress": "📈🔄",  # Training progress
    "DocumentMRQTrainDimension": "🧩📊",  # Dimension training
    "DocumentPairBuilderProgress": "📊📑",  # Pair building progress
    "SVMInferenceInitialized": "📊🤖",  # SVM inference agent initialized
    "LoadingSVMModel": "📥📊",  # Loading SVM model
    "SVMInferenceCompleted": "📊✅",  # SVM inference completed
    "EBTBufferCreated": "🧪📦",  # EBT buffer created
    "EBTTrainerEpoch": "🏋️🧪",  # EBT training epoch
    "TrainingCompleted": "🏁🎓",  # Training completed
    "MRQTrainingEpoch": "🏋️🧠",  # MRQ training epoch
    "MRQEarlyStopping": "🛑🏋️",  # MRQ early stopping
    "MRQTrainingComplete": "🎓🧠",  # MRQ training completed
    "MRQTrainerInitialized": "🧠🤖",  # MRQ trainer initialized
    "NoSamplesFound": "🚫🚫🚫",  # No samples found for training
}

PROMPTS = {
    "PromptLoaded": "📄✅",  # Prompt loaded
    "PromptStored": "💾📄",  # Prompt stored
    "PromptExecuted": "💬▶️",  # Prompt executed
    "PromptFileLoading": "📄🔄",  # Prompt file loading
    "PromptFileLoaded": "📄✅",  # Prompt file loaded
}

# ==================
# HYPOTHESIS WORKFLOW
# ==================
HYPOTHESIS_OPS = {
    "GoalCreated": "🎯✨",  # Goal created
    "GoalDomainAssigned": "🎯🏷️",  # Goal domain assigned
    "GeneratedHypotheses": "💡✨",  # Hypotheses generated
    "HypothesisStored": "💾💡",  # Hypothesis stored
    "HypothesisInserted": "📥💡",  # Hypothesis inserted
    "HypothesisStoreFailed": "❌💡",  # Hypothesis store failed
    "EvolvingTopHypotheses": "🔄💡",  # Hypotheses evolving
    "EvolvedHypotheses": "🌱💡",  # Hypotheses evolved
    "GraftingPair": "🌿➕",  # Hypothesis grafting
    "EditGenerated": "✏️",  # Hypothesis edit
    "SimilarHypothesesFound": "🔍💡",  # Similar hypotheses found
    "NoHypothesesInContext": "🚫💡",  # No hypotheses found
}

# =================
# RESEARCH & DATA
# =================
RESEARCH = {
    "ArxivSearchStart": "🔍📚",  # Arxiv search started
    "ArxivSearchComplete": "✅📚",  # Arxiv search completed
    "ArxivQueryFilters": "⚙️🔍",  # Arxiv filters applied
    "DocumentsToJudge": "📄⚖️",  # Documents to judge
    "DocumentsFiltered": "📑🔍",  # Documents filtered
    "LiteratureSearchCompleted": "✅📚",  # Literature search completed
    "LiteratureSearchSkipped": "⏭️📚",  # Literature search skipped
    "SearchingWeb": "🌐🔍",  # Web search in progress
    "SearchResult": "🔎📄",  # Search result found
    "NoResultsFromWebSearch": "🌐🚫",  # No search results
    "DocumentProfiled": "📄📋",  # Document profiled
    "DocumentProfileFailed": "📄❌",  # Document profile failed
    "DocumentsSearched": "📄🔍",  # Documents searched
}

# ===================
# DEBUG & DIAGNOSTICS
# ===================
DEBUGGING = {
    "debug": "🐞",  # Debug message
    "NodeDebug": "🌲🔍",  # Node debugging
    "NodeSummary": "🌲📝",  # Node summary
    "StageContext": "🔧📋",  # Stage context
    "TrimmingSection": "✂️",  # Section trimming
    "ContextAfterStage": "🗃️➡️",  # Post-stage context
    "ClassificationStarted": "🏷️▶️",  # Classification started
    "ClassificationCompleted": "🏷️✅",  # Classification completed
}

# ======================
# ERROR & WARNING STATES
# ======================
ERROR_STATES = {
    "DocumentLoadFailed": "⚠️📄",  # Document load failed
    "LiteratureQueryFailed": "❌📚",  # Literature query failed
    "HypothesisStoreFailed": "❌💾",  # Hypothesis store failed
    "PromptLoadFailed": "❌📝",  # Prompt load failed
    "PromptParseFailed": "❌📝",  # Prompt parse failed
    "PromptEvaluationFailed": "❌📝",  # Prompt evaluation failed
    "TrainingError": "❌🏋️",  # Training error
    "PreferencePairSaveError": "❌💾",  # Preference save error
    "RefinerError": "❌🔄",  # Refiner error
    "DocumentMRQModelMissing": "❌🧠",  # MRQ model missing
    "DocumentMRQTunerMissing": "❌🔧",  # MRQ tuner missing
    "TunedPromptGenerationFailed": "❌🔄📝",  # Tuned prompt failed
    "InvalidRuleMutation": "❌🧬",  # Invalid rule mutation
    "DocumentFilterSkipped": "⏭️📄",  # Document filter skipped
}

# =============
# SPECIAL CASES
# =============
SPECIAL = {
    "SQLQuery": "💾🔍",  # SQL query executed
    "EthicsReviewsGenerated": "⚖️🧾",  # Ethics reviews generated
    "SurveyAgentSkipped": "⏭️📋",  # Survey skipped
    "EarlyStopping": "🛑⏱️",  # Early stopping triggered
    "SharpenedHypothesisSaved": "💎💾",  # Sharpened hypothesis saved
    "CoTGenerated": "⛓️💭",  # Chain-of-Thought generated
    "LLMCacheHit": "💾⚡",  # LLM cache hit
}

MODELS = {
    "SVMModelSaved": "💾📊",  # SVM model saved
    "SVMModelLoaded": "📥📊",  # SVM model load
    "SVMModelTrainingStarted": "🏋️⚖️",
    "SVMTrainingStarted": "🏋️📊",
    "EBTModelLoaded": "📥🧪",  # EBT model loaded
    "SICQLTrainingEpoch": "🏋️📊",  # SICQL training epoch
    "SICQLTrainingComplete": "🎓📊",  # SICQL training completed

}

# Combine all categories into a single dictionary
EVENT_ICONS = {
    **AGENT_EVENTS,
    **SYSTEM_INIT,
    **KNOWLEDGE_OPS,
    **PIPELINE_FLOW,
    **SCORING,
    **REASONING,
    **TRAINING,
    **HYPOTHESIS_OPS,
    **RESEARCH,
    **DEBUGGING,
    **ERROR_STATES,
    **SPECIAL,
    **PROMPTS,
    **MODELS,
}
