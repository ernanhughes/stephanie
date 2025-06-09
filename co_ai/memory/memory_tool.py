from typing import Any, Optional

import psycopg2
from pgvector.psycopg2 import register_vector
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from co_ai.logs import JSONLogger
from co_ai.memory.context_store import ContextStore
from co_ai.memory.embedding_store import EmbeddingStore
from co_ai.memory.evaluation_store import EvaluationStore
from co_ai.memory.goal_store import GoalStore
from co_ai.memory.hypothesis_store import HypothesisStore
from co_ai.memory.idea_store import IdeaStore
from co_ai.memory.lookahead_store import LookaheadStore
from co_ai.memory.method_plan_store import MethodPlanStore
from co_ai.memory.mrq_store import MRQStore
from co_ai.memory.pattern_store import PatternStatStore
from co_ai.memory.pipeline_run_store import PipelineRunStore
from co_ai.memory.prompt_program_store import PromptProgramStore
from co_ai.memory.prompt_store import PromptStore
from co_ai.memory.reflection_delta_store import ReflectionDeltaStore
from co_ai.memory.rule_application_store import RuleApplicationStore
from co_ai.memory.rule_effect_store import RuleEffectStore
from co_ai.memory.score_store import ScoreStore
from co_ai.memory.search_result_store import SearchResultStore
from co_ai.memory.sharpening_store import SharpeningStore
from co_ai.memory.symbolic_rule_store import SymbolicRuleStore
from co_ai.models.base import engine  # From your SQLAlchemy setup


class MemoryTool:
    def __init__(self, cfg, logger: Optional[JSONLogger] = None):
        self.cfg = cfg
        self.logger = logger
        self._stores = {}  # name -> Store instance

        # Create a new session
        self.session_maker = sessionmaker(bind=engine)
        self.session: Session = self.session_maker()

        # Create connection
        conn = psycopg2.connect(
            dbname=self.cfg.get("db").get("name"),
            user=self.cfg.get("db").get("user"),
            password=self.cfg.get("db").get("password"),
            host=self.cfg.get("db").get("host"),
            port=self.cfg.get("db").get("port"),
        )
        conn.autocommit = True
        register_vector(conn)  # Register pgvector extension

        # Register stores
        self.register_store(GoalStore(self.session, logger))
        embedding_store = EmbeddingStore(self.cfg, conn, self.session, logger)
        self.register_store(embedding_store)
        self.register_store(HypothesisStore(self.session, logger, embedding_store))
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
                self.logger.log("SessionRefreshed", {"new_session_id": id(self.session)})