# stephanie/memory/memory_tool.py
from typing import Any, Optional

import psycopg2
from pgvector.psycopg2 import register_vector
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from stephanie.logs import JSONLogger
from stephanie.memory.belief_cartridge_store import BeliefCartridgeStore
from stephanie.memory.cartridge_domain_store import CartridgeDomainStore
from stephanie.memory.cartridge_store import CartridgeStore
from stephanie.memory.cartridge_triple_store import CartridgeTripleStore
from stephanie.memory.context_store import ContextStore
from stephanie.memory.document_domain_section_store import \
    DocumentSectionDomainStore
from stephanie.memory.document_domain_store import DocumentDomainStore
from stephanie.memory.document_section_store import DocumentSectionStore
from stephanie.memory.document_store import DocumentStore
from stephanie.memory.embedding_store import EmbeddingStore
from stephanie.memory.evaluation_attribute_store import \
    EvaluationAttributeStore
from stephanie.memory.evaluation_store import EvaluationStore
from stephanie.memory.goal_dimensions_store import GoalDimensionsStore
from stephanie.memory.goal_store import GoalStore
from stephanie.memory.hf_embedding_store import HuggingFaceEmbeddingStore
from stephanie.memory.hnet_embedding_store import HNetEmbeddingStore
from stephanie.memory.hypothesis_store import HypothesisStore
from stephanie.memory.idea_store import IdeaStore
from stephanie.memory.lookahead_store import LookaheadStore
from stephanie.memory.memcube_store import MemcubeStore
from stephanie.memory.method_plan_store import MethodPlanStore
from stephanie.memory.mrq_store import MRQStore
from stephanie.memory.pattern_store import PatternStatStore
from stephanie.memory.pipeline_run_store import PipelineRunStore
from stephanie.memory.pipeline_stage_store import PipelineStageStore
from stephanie.memory.prompt_program_store import PromptProgramStore
from stephanie.memory.prompt_store import PromptStore
from stephanie.memory.reflection_delta_store import ReflectionDeltaStore
from stephanie.memory.rule_application_store import RuleApplicationStore
from stephanie.memory.rule_effect_store import RuleEffectStore
from stephanie.memory.score_store import ScoreStore
from stephanie.memory.scoring_store import ScoringStore
from stephanie.memory.search_result_store import SearchResultStore
from stephanie.memory.sharpening_store import SharpeningStore
from stephanie.memory.symbolic_rule_store import SymbolicRuleStore
from stephanie.models.base import engine  # From your SQLAlchemy setup


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

        embedding_cfg = self.cfg.get("embeddings", {})
        # Register stores
        mxbai = EmbeddingStore(embedding_cfg, self.conn, self.session, logger)
        self.register_store(mxbai)
        hnet = HNetEmbeddingStore(embedding_cfg, self.conn, self.session, logger)
        self.register_store(hnet)
        hf = HuggingFaceEmbeddingStore(embedding_cfg, self.conn, self.session, logger)
        self.register_store(hf)

        # Choose embedding backend based on config
        selected_backend = embedding_cfg.get("backend", "mxbai")
        if selected_backend == "hnet":
            self.embedding = hnet
        elif selected_backend == "huggingface":
            self.embedding = hf
        else:
            self.embedding = mxbai


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
        self.register_store(DocumentDomainStore(self.session, logger))
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

        # Register extra stores if defined in config
        if cfg.get("extra_stores"):
            for store_class in cfg.get("extra_stores", []):
                self.register_store(store_class(self.session, logger))

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
