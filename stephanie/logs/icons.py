# stephanie/logs/icons.py
def get_event_icon(event_type: str) -> str:
    """Get the icon associated with a specific event type."""
    return EVENT_ICONS.get(event_type, "❓")  # Default: question mark

# ========================
# SYSTEM & INITIALIZATION
# ========================
SYSTEM_INIT = {
    "ContextManagerInitialized": "⚙️",  # Context manager initialized
    "UncertaintyEstimated": "🔍",  # Uncertainty estimation
    "EBTEnergyCalculated": "⚡",  # EBT energy calculation
    "ScoringPolicyCompleted": "✅",  # Scoring policy completed
    "AllEBTModelsLoaded": "📦✅",  # All EBT models loaded
    "SupervisorInit": "👨‍🏫",  # Supervisor initialization
    "DocumentLLMInferenceCompleted": "📄✅",  # Document LLM inference completed
    "DocumentEmbeddingsBackfilled": "📄🌱",  # Document embeddings backfilled
    "AgentInitialized": "🤖",  # Agent initialization
    "ContextLoaded": "📂",  # Context loaded
    "ContextSaved": "💾",  # Context saved
    "SupervisorComponentsRegistered": "👨‍🏫",  # Supervisor registration
    "DomainClassifierInit": "🏷️🧠",  # Domain classifier init
    "DomainConfigLoaded": "🏷️📋",  # Domain config loaded
    "SeedEmbeddingsPrepared": "🌱🧬",  # Seed embeddings prepared
    "KnowledgeDBLoaded": "📚✅",  # Knowledge database loaded
}

# =================
# AGENT OPERATIONS
# =================
AGENT_EVENTS = {
    "AgentInitialized": "🤖",  # Agent initialization
    "AgentRunStarted": "🤖▶️",  # Agent run started
    "AgentRunCompleted": "🤖⏹️",  # Agent run completed
    "AgentRanSuccessfully": "🤖✅",  # Agent succeeded
    "GILDTrainerAgentInitialized": "📊🤖",  # GILD trainer agent initialized
    "MRQInferenceAgentInitialized": "📊🤖",  # MRQ inference agent initialized
    "MRQTrainerAgentInitialized": "📊🤖",  # MRQ trainer agent initialized
    "DocumentMRQInferenceAgentInitialized": "📊🤖",  # Document MRQ inference agent initialized
    "DocumentEBTInferenceAgentInitialized": "🧠🚦",  # Inference agent initialized
    "EpistemicPlanExecutorAgentInitialized": "🪸🤖",  # Epistemic plan executor agent initialized
}

# =================
# KNOWLEDGE STORAGE
# =================
KNOWLEDGE_OPS = {
    "GenerationStart": "🧑‍🧒‍🧒▶️",  # Knowledge generation started
    "GoalContextOverride": "🎯🔄",  # Goal context override
    "MgrScoreParseError": "📊❌",  # Scoring parse error
    "SymbolicRulesFound": "🧩🔍",  # Symbolic rules found
    "MRQTrainingDataLoaded": "📊📥",  # MRQ training data loaded
    "DuplicateSymbolicRuleSkipped": "🚫🧩",  # Duplicate symbolic rule skipped
    "EvolvedParsedHypotheses": "🌱💡",  # Evolved hypotheses parsed
    "EvolutionCompleted": "🌱✅",  # Evolution completed
    "ExecutionStepStored": "📥✅",  # Execution step stored
    "MetaReviewInput": "📝📋",  # Meta-review input
    "RawMetaReviewOutput": "📝📄",  # Raw meta-review output
    "NotEnoughHypothesesForRanking": "❌💡",  # Not enough hypotheses for ranking
    "PromptLookup": "🔍📝",  # Prompt lookup
    "RubricClassified": "🏷️📄",  # Rubric classified
    "PlanTraceStored": "📄💾",  # Plan trace stored
    "PromptGenerated": "📝✨",  # Prompt generated
    "PatternStatsStored": "📊💾",  # Pattern stats stored
    "LLMJudgeResults": "📝⚖️",  # LLM judge results
    "RubricPatternsStored": "📊💾",  # Rubric patterns stored
    "EBTBufferLoaded": "🧪📦",  # EBT buffer loaded
    "EBTInferenceCompleted": "🧪✅",  # EBT inference complete
    "MemCubeSaved": "💾📦✅",  # MemCube saved
    "DocumentRefinedWithEBT": "📄🔄",  # Document refined with EBT
    "EBTExampleAdded": "🧪➕",  # EBT example added
    "MRQScoresCalculated": "📊✅",  # MRQ scores calculated
    "ScoringEvent": "📊",  # Scoring event
    "TripletsRetrievedByDomain": "🔗🏷️",  # Triplets retrieved by domain
    "DomainAssigned": "🏷️✅",  # Domain assigned
    "MRQTunedScore": "🧠📊",  # MRQ tuned score
    "CartridgeCreated": "💾📦",  # Cartridge created
    "CartridgeAlreadyExists": "💾✅",  # Cartridge exists check
    "TriplesAlreadyExist": "🔗✅",  # Triples exist check
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
    "PipelineStageCompleted": "🖇️✅",  # Pipeline stage completed
    "PipelineStageStarted": "🖇️▶️",  # Pipeline stage started
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
    "PipelineJudgeAgentEnd": "⚖️🔚",  # Judge agent completed
    "MRQPipelineSuggested": "🧠💡",  # MRQ pipeline suggested
    "PipelineStageFailed": "🖇️⚠️",  # Pipeline stage failed
    "PipelineScoreSummary": "🖇️📊",  # Pipeline score summary
    "PipelineError": "🖇️❌",  # Pipeline error
}

# =====================
# SCORING & EVALUATION
# =====================
SCORING = {
    "ScoringPaper": "📄⚖️",  # Scoring paper
    "EpistemicPlanExecutorSkipped": "🪸⏭️",  # Epistemic plan executor skipped
    "EpistemicPlanHRMTrainingBatch": "🪸🏋️",  # Training batch
    "EpistemicPlanHRMDataLoaderCreated": "🪸📥",  # Data loader created
    "EpistemicPlanHRMTrainingEpoch": "🪸🏋️",  # Training epoch
    "EpistemicPlanHRMModelSaved": "🪸🧮💾",  # Model saved
    "EpistemicTraceSaved": "🪸💾",  # Epistemic trace saved
    "HRMScorerEvaluated": "🧠⚖️",  # HRM scorer evaluated
    "HRMScorerModelLoaded": "🧠🧮📥",  # HRM scorer model loaded
    "HRMScorerMetaLoaded": "🧠📄",  # HRM scorer meta loaded
    "LATS_StepStarted": "🧠🔄",  # LATS step started
    "LATS_StepCompleted": "🧠✅",  # LATS step completed
    "LargeDataContextComponentDumped": "📂💾",  # Large data context dumped
    "EpistemicPlanExecutorStarted": "🪸📄",  # Epistemic plan executor started
    "EpistemicPlanExecutorCompleted": "🪸✅",  # Epistemic plan executor completed
    "PolicyLogits": "📊⚖️",  # Policy logits computed
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
    "AllMRQModelsLoaded": "📊🧮✅",  # All MRQ models loaded
    "LoadingModelPaths": "📂🧮🔄",  # Model paths loading
    "DocumentModelSaved": "📄💾",  # Document model saved
    "ModelSaved": "💾🧮✅",  # Model saved
    "EncoderSaved": "📄💾",  # Encoder saved
    "MRQInferenceCompleted": "📊✅",  # MRQ inference completed
    "SVMScoringFinished": "📊🏁",  # SVM scoring finished
    "SVMScoringStarted": "📊▶️",  # SVM scoring started
    "SVMScoreComputed": "📊✅",  # SVM score computed
    "PolicyAnalysis": "📊🔍",  # Policy analysis
    "NoSICQLDataFound": "🚫📊",  # No SI-CQL data found
    "DimensionEvaluated": "📏✅",  # Dimension evaluated
}

# =====================
# REASONING & ANALYSIS
# =====================
REASONING = {
    "PlanTraceMonitorDisabled": "📄🔧",  # Plan trace monitoring disabled
    "PlanTraceSavedToFile": "📄💾",  # Plan trace saved to file
    "PlanTraceCompleted": "📄✅",  # Plan trace completed
    "MARSAnalysisCompleted": "📊✅",  # MARS analysis completed
    "PlanTraceScoringCompleted": "📄✅",  # Plan trace scoring completed
    "PlanTraceUpdated": "📄🔄",  # Plan trace updated
    "PlanTraceScored": "📄⚖️",  # Plan trace scored
    "PlanTraceScoringComplete": "📄✅",  # Plan trace scoring completed
    "DocumentScoringProgress": "📄🔄",  # Document scoring progress
    "DocumentScoringCompleted": "📄✅",  # Document scoring completed
    "EpistemicPlanHRMModelInitialized": "🪸🧠",  # Epistemic Plan HRM model initialized
    "EpistemicPlanHRMOptimizerInitialized": "🪸⚙️",  # Epistemic Plan HRM optimizer initialized
    "EpistemicPlanHRMLossInitialized": "🪸📉",  # Epistemic Plan HRM loss initialized
    "EpistemicPlanHRMTrainingNoTraces": "🪸🚫",  # No traces for training
    "EpistemicPlanHRMTrainingStarted": "🪸🚀",  # Epistemic Plan HRM training started
    "EpistemicPlanHRMTrainingDataPrepared": "🪸📊",  # Training data prepared

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
    "TrainingStarted": "🏋️▶️",  # Training started
    "CalibrationCompleted": "✅📊",  # Calibration completed
    "ContrastiveRankerTrainingComplete": "🎓📊",  # Contrastive ranker training completed
    "TrainingEpochsCompleted": "🏋️✅",  # Training epochs completed
    "DimensionTrainingStart": "📏▶️",  # Dimension training started
    "ContrastiveRankerTrainingStarted": "📊🏋️",  # Contrastive ranker training started I
    "SICQLTrainerInitialized": "📊🤖",  # SICQL trainer initialized   
    "SVMTrainingComplete": "🎓📊",  # SVM training completed
    "SVMTrainingCompleted": "🎓📊",  # SVM training completed
    "SVMTrainerInvoked": "📊🤖",  # SVM trainer invoked
    "DimensionTrainingStarted": "📏▶️",  # Dimension training started
    "DimensionTrainingComplete": "📏🎓",  # Dimension training completed
    "TunerMissing": "🔧📄",  # Tuner missing
    "MRQTrainerEpoch": "🏋️",  # MRQ training epoch
    "MRQTrainerStart": "🚀🧠",  # MRQ training started
    "MRQTrainerTrainingComplete": "🎓🧠",  # MRQ training completed
    "MRQModelInitializing": "🧠🧮⚙️",  # MRQ model initializing
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
    "MRQEarlyStopping": "✨🏋️",  # MRQ early stopping
    "MRQTrainerInitialized": "🧠🤖",  # MRQ trainer initialized
    "NoSamplesFound": "🚫🚫",  # No samples found for training
    "SICQLTrainingEpoch": "🏋️📊",  # SICQL training epoch
    "SICQLTrainingComplete": "🎓📊",  # SICQL training completed
}

# =================
# HYPOTHESIS WORKFLOW
# =================
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
    "SharpenedHypothesisSaved": "💎💾",  # Sharpened hypothesis saved
}

# =================
# PROMPT OPERATIONS
# =================
PROMPTS = {
    "PromptLoaded": "📄✅",  # Prompt loaded
    "PromptStored": "💾📄",  # Prompt stored
    "PromptExecuted": "💬▶️",  # Prompt executed
    "PromptFileLoading": "📄🔄",  # Prompt file loading
    "PromptFileLoaded": "📄✅",  # Prompt file loaded
    "CoTGenerated": "⛓️💭",  # Chain-of-Thought generated
    "LLMCacheHit": "💾⚡",  # LLM cache hit
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
    "DocumentProfileFailed": "📄❌",  # Document profile failed
    "DocumentsSearched": "📄🔍",  # Documents searched
    "SurveyAgentSkipped": "⏭️📋",  # Survey skipped
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
# MODEL OPS
# =============
MODELS = {
    "EpistemicPlanHRMScorerModelLoaded": "🪸🧠📥",  # Epistemic Plan HRM scorer model loaded

    "SVMModelSaved": "💾🧮📊",  # SVM model saved
    "SVMModelLoaded": "📥🧮📊",  # SVM model load
    "SVMModelTrainingStarted": "🏋️🧮⚖️",
    "SVMTrainingStarted": "🏋️📊",
    "EBTModelLoaded": "📥🧮🧪",  # EBT model loaded
    "DocumentEBTModelSaved": "💾🧮✅",  # Model saved after training
    "DocumentEBTTrainingStart": "🧪▶️",  # Training started for a dimension
    "DocumentEBTEpoch": "📊🔁",  # Epoch completed during training
}

# =============
# ETHICS & REVIEWS
# =============
SPECIAL = {
    "PlanTraceCreated": "📄📝",  # Plan trace created
    "PlanTraceScorerInitialized": "📊🤖🪸",  # Plan trace scorer initialized
    "PlanTraceMonitorInitialized": "📊🤖🪸",  # Plan trace monitor initialized
    "GILDProcessTraceStarted": "📊▶️",  # GILD process trace started
    "SICQLAdvantageExtracted": "📊📈",  # SICQL advantage extracted
    "SICQLAdvantageWarning": "⚠️📊",  # SICQL advantage warning
    "GILDDataPreparationCompleted": "📊✅",  # GILD data preparation completed
      
    "SQLQuery": "💾🔍",  # SQL query executed
    "EthicsReviewsGenerated": "⚖️🧾",  # Ethics reviews generated
    "EarlyStopping": "✅⏱️",  # Early stopping triggered
}

# Combine all categories into a single dictionary
EVENT_ICONS = {
    **SYSTEM_INIT,
    **AGENT_EVENTS,
    **KNOWLEDGE_OPS,
    **PIPELINE_FLOW,
    **SCORING,
    **REASONING,
    **TRAINING,
    **HYPOTHESIS_OPS,
    **PROMPTS,
    **RESEARCH,
    **DEBUGGING,
    **ERROR_STATES,
    **MODELS,
    **SPECIAL,
}