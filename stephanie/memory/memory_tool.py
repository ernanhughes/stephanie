# stephanie/memory/memory_tool.py
from typing import Any, Optional

import psycopg2
from pgvector.psycopg2 import register_vector
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from stephanie.logging import JSONLogger
from stephanie.memory.belief_cartridge_store import BeliefCartridgeStore
from stephanie.memory.calibration_event_store import CalibrationEventStore
from stephanie.memory.cartridge_domain_store import CartridgeDomainStore
from stephanie.memory.cartridge_store import CartridgeStore
from stephanie.memory.cartridge_triple_store import CartridgeTripleStore
from stephanie.memory.casebook_store import CaseBookStore
from stephanie.memory.chat_store import ChatStore
from stephanie.memory.context_store import ContextStore
from stephanie.memory.document_domain_section_store import \
    DocumentSectionDomainStore
from stephanie.memory.document_section_store import DocumentSectionStore
from stephanie.memory.document_store import DocumentStore
from stephanie.memory.dynamic_scorable_store import DynamicScorableStore
from stephanie.memory.embedding_store import EmbeddingStore
from stephanie.memory.entity_cache_store import EntityCacheStore
from stephanie.memory.evaluation_attribute_store import \
    EvaluationAttributeStore
from stephanie.memory.evaluation_store import EvaluationStore
from stephanie.memory.execution_step_store import ExecutionStepStore
from stephanie.memory.goal_dimensions_store import GoalDimensionsStore
from stephanie.memory.goal_store import GoalStore
from stephanie.memory.hf_embedding_store import HuggingFaceEmbeddingStore
from stephanie.memory.hnet_embedding_store import HNetEmbeddingStore
from stephanie.memory.hypothesis_store import HypothesisStore
from stephanie.memory.idea_store import IdeaStore
from stephanie.memory.lookahead_store import LookaheadStore
from stephanie.memory.mars_conflict_store import MARSConflictStore
from stephanie.memory.mars_result_store import MARSResultStore
from stephanie.memory.memcube_store import MemcubeStore
from stephanie.memory.method_plan_store import MethodPlanStore
from stephanie.memory.mrq_store import MRQStore
from stephanie.memory.pattern_store import PatternStatStore
from stephanie.memory.pipeline_reference_store import PipelineReferenceStore
from stephanie.memory.pipeline_run_store import PipelineRunStore
from stephanie.memory.pipeline_stage_store import PipelineStageStore
from stephanie.memory.plan_trace_store import PlanTraceStore
from stephanie.memory.prompt_program_store import PromptProgramStore
from stephanie.memory.prompt_store import PromptStore
from stephanie.memory.reflection_delta_store import ReflectionDeltaStore
from stephanie.memory.report_store import ReportStore
from stephanie.memory.rule_application_store import RuleApplicationStore
from stephanie.memory.rule_effect_store import RuleEffectStore
from stephanie.memory.scorable_domain_store import ScorableDomainStore
from stephanie.memory.scorable_embedding_store import ScorableEmbeddingStore
from stephanie.memory.scorable_entity_store import ScorableEntityStore
from stephanie.memory.scorable_rank_store import ScorableRankStore
from stephanie.memory.score_store import ScoreStore
from stephanie.memory.scoring_store import ScoringStore
from stephanie.memory.search_result_store import SearchResultStore
from stephanie.memory.sharpening_store import SharpeningStore
from stephanie.memory.symbolic_rule_store import SymbolicRuleStore
from stephanie.memory.theorem_store import TheoremStore
from stephanie.memory.training_event_store import TrainingEventStore
from stephanie.models.base import engine  # From your SQLAlchemy setup
from stephanie.services.bus.hybrid_bus import HybridKnowledgeBus
from stephanie.services.bus.knowledge_bus import KnowledgeBus


class MemoryTool:
    def __init__(self, cfg: dict, logger: Optional[JSONLogger] = None):
        self.cfg = cfg
        self.logger = logger
        self._stores = {}  # name -> Store instance

        # Create a new session
        self.session_maker = sessionmaker(bind=engine)
        self.session: Session = self.session_maker()

        db_cfg = self.cfg.get("db", {})
        # Create connection
        self.conn = psycopg2.connect(
            dbname=db_cfg.get("name"),
            user=db_cfg.get("user"),
            password=db_cfg.get("password"),
            host=db_cfg.get("host"),
            port=db_cfg.get("port"),
        )
        self.conn.autocommit = True
        register_vector(self.conn)  # Register pgvector extension

        # Setup knowledge bus needed before embeddings bvecause of ner
        self.bus = self._setup_knowledge_bus()


        embedding_cfg = self.cfg.get("embeddings", {})
        # Register stores
        mxbai = EmbeddingStore(embedding_cfg, memory=self, logger=logger)
        self.register_store(mxbai)
        hnet = HNetEmbeddingStore(embedding_cfg, memory=self, logger=logger)
        self.register_store(hnet)
        hf = HuggingFaceEmbeddingStore(embedding_cfg, memory=self, logger=logger)
        self.register_store(hf)

        # Choose embedding backend based on config
        selected_backend = embedding_cfg.get("backend", "hnet")
        if selected_backend == "hnet":
            self.embedding = hnet
        elif selected_backend == "huggingface":
            self.embedding = hf
        else:
            self.embedding = mxbai

        # Choose embedding backend based on config
        selected_backend = embedding_cfg.get("backend", "mxbai")
        if selected_backend == "hnet":
            self.embedding = hnet
        elif selected_backend == "huggingface":
            self.embedding = hf
        else:
            self.embedding = mxbai

        self.logger.log(
            "EmbeddingBackendSelected",
            {
                "backend": selected_backend,
                "db_host": db_cfg.get("host"),
                "db_name": db_cfg.get("name"),
                "db_port": db_cfg.get("port"),
                "conn_id": id(self.conn),  # unique Python object ID
            },
        )

        self._meta: dict = {} 

        # Register stores
        self.register_store(GoalStore(self.session, logger))
        self.register_store(HypothesisStore(self.session, logger, self.embedding))
        self.register_store(PromptStore(self.session, logger))
        self.register_store(EvaluationStore(self.session, logger))
        self.register_store(PipelineRunStore(self.session, logger))
        self.register_store(LookaheadStore(self.session, logger))
        self.register_store(ContextStore(self.session, logger))
        self.register_store(ReflectionDeltaStore(self.session, logger))
        self.register_store(PatternStatStore(self.session, logger))
        self.register_store(SearchResultStore(self.session, logger))
        self.register_store(IdeaStore(self.session, logger))
        self.register_store(MethodPlanStore(self.session, logger))
        self.register_store(MRQStore(cfg, self.session, logger))
        self.register_store(SharpeningStore(self.session, logger))
        self.register_store(SymbolicRuleStore(self.session, logger))
        self.register_store(RuleEffectStore(self.session, logger))
        self.register_store(RuleApplicationStore(self.session, logger))
        self.register_store(PromptProgramStore(self.session, logger))
        self.register_store(ScoreStore(self.session, logger))
        self.register_store(DocumentStore(self.session, logger))
        self.register_store(ScorableDomainStore(self.session, logger))
        self.register_store(DocumentSectionStore(self.session, logger))
        self.register_store(DocumentSectionDomainStore(self.session, logger))
        self.register_store(CartridgeDomainStore(self.session, logger))
        self.register_store(CartridgeStore(self.session, logger))
        self.register_store(CartridgeTripleStore(self.session, logger))
        self.register_store(MemcubeStore(self.session, logger))
        self.register_store(BeliefCartridgeStore(self.session, logger))
        self.register_store(GoalDimensionsStore(self.session, logger))
        self.register_store(PipelineStageStore(self.session, logger))
        self.register_store(ScoringStore(self.session, logger))
        self.register_store(EvaluationAttributeStore(self.session, logger))
        self.register_store(ExecutionStepStore(self.session, logger))
        self.register_store(PlanTraceStore(self.session, logger))
        self.register_store(PipelineReferenceStore(self.session, logger))
        self.register_store(ScorableEmbeddingStore(self.session, logger, self.embedding))
        self.register_store(ReportStore(self.session, logger))
        self.register_store(TheoremStore(self.session, logger))
        self.register_store(ScorableRankStore(self.session, logger))
        self.register_store(MARSResultStore(self.session, logger))
        self.register_store(MARSConflictStore(self.session, logger))
        self.register_store(CaseBookStore(self.session, logger))
        self.register_store(ChatStore(self.session, logger))
        self.register_store(ScorableEntityStore(self.session, logger))
        self.register_store(DynamicScorableStore(self.session, logger))
        self.register_store(CalibrationEventStore(self.session, logger))
        self.register_store(EntityCacheStore(self.session, logger))
        self.register_store(TrainingEventStore(self.session, logger))


        # Register extra stores if defined in config
        if cfg.get("extra_stores"):
            for store_class in cfg.get("extra_stores", []):
                self.register_store(store_class(self.session, logger))

        self.logger.log("KnowledgeBusInitialized", {
            "backend": self.cfg.get("bus", {}).get("backend", "inprocess")
        })


    def register_store(self, store):
        store_name = getattr(store, "name", store.__class__.__name__)
        if store_name in self._stores:
            raise ValueError(f"A store named '{store_name}' is already registered.")
        self._stores[store_name] = store

        if self.logger:
            self.logger.log("StoreRegistered", {"store": store_name})

    def get(self, name: str) -> Optional[Any]:
        return self._stores.get(name)

    def __getattr__(self, name: str):
        if name in self._stores:
            return self._stores[name]
        raise AttributeError(f"'MemoryTool' has no attribute '{name}'")

    def commit(self):
        """Commit any pending changes"""
        try:
            self.session.commit()
        except SQLAlchemyError as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("SessionRollback", {"error": str(e)})
            raise

    @property
    def meta(self) -> dict:
        """Lightweight, process-local key/value store for small state."""
        return self._meta

    def close(self):
        """Close session at end of run"""
        try:
            self.session.close()
        except SQLAlchemyError as e:
            if self.logger:
                self.logger.log("SessionCloseFailed", {"error": str(e)})
            self.session = self.session_maker()  # Reopen session on failure

    def begin_nested(self):
        """Start nested transaction (for safe rollback during complex ops)"""
        return self.session.begin_nested()

    def refresh_session(self):
        """Closes current session and creates a fresh one"""
        try:
            self.session.rollback()
            self.session.close()
        finally:
            self.session = self.session_maker()
            if self.logger:
                self.logger.log(
                    "SessionRefreshed", {"new_session_id": id(self.session)}
                )

    def _setup_knowledge_bus(self) -> KnowledgeBus:
        return HybridKnowledgeBus(self.cfg.get("bus", {}), self.logger)
    
