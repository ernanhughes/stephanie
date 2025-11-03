# stephanie/memory/memory_tool.py
from __future__ import annotations

import asyncio
from typing import Any, Optional

import psycopg2
from pgvector.psycopg2 import register_vector
from sqlalchemy.orm import sessionmaker

from stephanie.logging import JSONLogger
from stephanie.memory.belief_cartridge_store import BeliefCartridgeStore
from stephanie.memory.bus_event_store import BusEventStore
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
from stephanie.memory.experiment_store import ExperimentStore
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
from stephanie.memory.models_store import ModelsStore
from stephanie.memory.mrq_store import MRQStore
from stephanie.memory.pattern_store import PatternStatStore
from stephanie.memory.pipeline_reference_store import PipelineReferenceStore
from stephanie.memory.pipeline_run_store import PipelineRunStore
from stephanie.memory.pipeline_stage_store import PipelineStageStore
from stephanie.memory.plan_trace_store import PlanTraceStore
from stephanie.memory.prompt_program_store import PromptProgramStore
from stephanie.memory.prompt_store import PromptStore
from stephanie.memory.reasoning_sample_store import ReasoningSampleStore
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
from stephanie.memory.selfplay_store import SelfPlayStore
from stephanie.memory.sharpening_store import SharpeningStore
from stephanie.memory.sis_card_store import SisCardStore
from stephanie.memory.symbolic_rule_store import SymbolicRuleStore
from stephanie.memory.theorem_store import TheoremStore
from stephanie.memory.training_event_store import TrainingEventStore
from stephanie.memory.training_stats_store import TrainingStatsStore
from stephanie.memory.trajectory_store import AgentTrajectoryStore
from stephanie.memory.vpm_store import VPMStore
from stephanie.models.base import engine  # From your SQLAlchemy setup
from stephanie.services.bus.hybrid_bus import HybridKnowledgeBus
from stephanie.services.bus.knowledge_bus import KnowledgeBus


class MemoryTool:
    def __init__(self, cfg: dict, logger: Optional[JSONLogger] = None):
        self.cfg = cfg
        self.logger = logger
        self._stores = {}

        self.session_maker = sessionmaker(bind=engine, expire_on_commit=False,
                                          autocommit=False, autoflush=False)
        self.session = self.session_maker

        db_cfg = self.cfg.get("db", {})
        self.conn = psycopg2.connect(
            dbname=db_cfg.get("name"),
            user=db_cfg.get("user"),
            password=db_cfg.get("password"),
            host=db_cfg.get("host"),
            port=db_cfg.get("port"),
        )
        self.conn.autocommit = True
        register_vector(self.conn)

        # --- Knowledge bus (do NOT auto-connect here) ---
        self.bus = self._setup_knowledge_bus()
        # Idempotent connection guards
        self._bus_lock: asyncio.Lock = asyncio.Lock()
        self._bus_connected_evt: asyncio.Event = asyncio.Event()

        embedding_cfg = self.cfg.get("embeddings", {})

        # Register embedding stores
        mxbai = EmbeddingStore(embedding_cfg, memory=self, logger=logger)
        hnet = HNetEmbeddingStore(embedding_cfg, memory=self, logger=logger)
        hf   = HuggingFaceEmbeddingStore(embedding_cfg, memory=self, logger=logger)
        self.register_store(mxbai)
        self.register_store(hnet)
        self.register_store(hf)

        # Choose embedding backend (single block!)
        selected_backend = embedding_cfg.get("backend", "hnet")
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
                "conn_id": id(self.conn),
            },
        )

        self._meta: dict = {}

        # Register stores
        self.register_store(GoalStore(self.session_maker, logger))
        self.register_store(
            HypothesisStore(self.session_maker, logger, self.embedding)
        )
        self.register_store(PromptStore(self.session_maker, logger))
        self.register_store(EvaluationStore(self.session_maker, logger))
        self.register_store(PipelineRunStore(self.session_maker, logger))
        self.register_store(LookaheadStore(self.session_maker, logger))
        self.register_store(ContextStore(self.session_maker, logger))
        self.register_store(ReflectionDeltaStore(self.session_maker, logger))
        self.register_store(PatternStatStore(self.session_maker, logger))
        self.register_store(SearchResultStore(self.session_maker, logger))
        self.register_store(IdeaStore(self.session_maker, logger))
        self.register_store(MethodPlanStore(self.session_maker, logger))
        self.register_store(MRQStore(cfg, self.session_maker, logger))
        self.register_store(SharpeningStore(self.session_maker, logger))
        self.register_store(SymbolicRuleStore(self.session_maker, logger))
        self.register_store(RuleEffectStore(self.session_maker, logger))
        self.register_store(RuleApplicationStore(self.session_maker, logger))
        self.register_store(PromptProgramStore(self.session_maker, logger))
        self.register_store(ScoreStore(self.session_maker, logger))
        self.register_store(DocumentStore(self.session_maker, logger))
        self.register_store(ScorableDomainStore(self.session_maker, logger))
        self.register_store(DocumentSectionStore(self.session_maker, logger))
        self.register_store(
            DocumentSectionDomainStore(self.session_maker, logger)
        )
        self.register_store(CartridgeDomainStore(self.session_maker, logger))
        self.register_store(CartridgeStore(self.session_maker, logger))
        self.register_store(CartridgeTripleStore(self.session_maker, logger))
        self.register_store(MemcubeStore(self.session_maker, logger))
        self.register_store(BeliefCartridgeStore(self.session_maker, logger))
        self.register_store(GoalDimensionsStore(self.session_maker, logger))
        self.register_store(PipelineStageStore(self.session_maker, logger))
        self.register_store(ScoringStore(self.session_maker, logger))
        self.register_store(
            EvaluationAttributeStore(self.session_maker, logger)
        )
        self.register_store(ExecutionStepStore(self.session_maker, logger))
        self.register_store(PlanTraceStore(self.session_maker, logger))
        self.register_store(PipelineReferenceStore(self.session_maker, logger))
        self.register_store(
            ScorableEmbeddingStore(self.session_maker, logger, self.embedding)
        )
        self.register_store(ReportStore(self.session_maker, logger))
        self.register_store(TheoremStore(self.session_maker, logger))
        self.register_store(ScorableRankStore(self.session_maker, logger))
        self.register_store(MARSResultStore(self.session_maker, logger))
        self.register_store(MARSConflictStore(self.session_maker, logger))
        self.register_store(CaseBookStore(self.session_maker, logger))
        self.register_store(ChatStore(self.session_maker, logger))
        # self.register_store(ScorableEntityStore(self.session_maker, logger))
        self.register_store(DynamicScorableStore(self.session_maker, logger))
        self.register_store(CalibrationEventStore(self.session_maker, logger))
        self.register_store(EntityCacheStore(self.session_maker, logger))
        self.register_store(TrainingEventStore(self.session_maker, logger))
        self.register_store(TrainingStatsStore(self.session_maker, logger))
        self.register_store(ModelsStore(self.session_maker, logger))
        self.register_store(SelfPlayStore(self.session_maker, logger))
        self.register_store(BusEventStore(self.session_maker, logger))
        self.register_store(ExperimentStore(self.session_maker, logger))
        self.register_store(SisCardStore(self.session_maker, logger))
        self.register_store(AgentTrajectoryStore(self.session_maker, logger))
        self.register_store(ReasoningSampleStore(self.session_maker, logger))
        self.register_store(VPMStore(self.session_maker, logger))

        if cfg.get("extra_stores"):
            for store_class in cfg.get("extra_stores", []):
                self.register_store(store_class(self.session_maker, logger))

    def _setup_knowledge_bus(self) -> KnowledgeBus:
        # Always pass nested {"bus": {...}} to make Hybrid bus happier
        bus_cfg = {"bus": self.cfg.get("bus", {"backend": "nats"})}
        bus = HybridKnowledgeBus(bus_cfg, self.logger)
        return bus

    async def ensure_bus_connected(self) -> None:
        bus_cfg = (self.cfg or {}).get("bus", {})
        required = bool(bus_cfg.get("required", False))

        if self._bus_connected_evt.is_set():
            return

        async with self._bus_lock:
            if self._bus_connected_evt.is_set():
                return
            ok = await self.bus.connect()
            if ok:
                self._bus_connected_evt.set()
                self.logger.info("KnowledgeBusReady", {"backend": self.bus.get_backend()})
                return

        # failed to connect
        self.logger.warning("KnowledgeBusConnectFailed", {"required": required})
        if required:
            raise RuntimeError("Knowledge bus failed to connect")
        # else: continue without a bus (backend = "none")

    def register_store(self, store):
        store_name = getattr(store, "name", store.__class__.__name__)
        if store_name in self._stores:
            raise ValueError(
                f"A store named '{store_name}' is already registered."
            )
        self._stores[store_name] = store

        if self.logger:
            self.logger.log("StoreRegistered", {"store": store_name})

    def get(self, name: str) -> Optional[Any]:
        return self._stores.get(name)

    def __getattr__(self, name: str):
        if name in self._stores:
            return self._stores[name]
        raise AttributeError(f"'MemoryTool' has no attribute '{name}'")

    @property
    def meta(self) -> dict:
        """Lightweight, process-local key/value store for small state."""
        return self._meta

