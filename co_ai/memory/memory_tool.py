import psycopg2
from co_ai.memory import ContextStore, EmbeddingStore, HypothesisStore, PromptLogger, ReportLogger
from co_ai.memory.base_store import BaseStore
from typing import Optional

class MemoryTool:
    def __init__(self, cfg, logger=None):

        db_config = cfg.db  # Load DB config from Hydra
        self.conn = psycopg2.connect(
            dbname=db_config.database,
            user=db_config.user,
            password=db_config.password,
            host=db_config.host,
            port=db_config.port,
        )
        self.conn.autocommit = True
        self.logger = logger
        self.cfg = cfg  # Store cfg if needed later
        self.db = self.conn.cursor()


        self.logger = logger
        self.embeddings = EmbeddingStore(self.db, cfg.embeddings, logger)
        self.hypotheses = HypothesisStore(self.db, self.embeddings, logger)
        self.context = ContextStore(self.db, logger)
        self.prompts = PromptLogger(self.db, logger)
        self.reports = ReportLogger(self.db, logger)

        self.register_store(EmbeddingStore(self.db, cfg.embeddings, logger))
        self.register_store(HypothesisStore(self.db, self.get("embedding"), logger))
        self.register_store(ContextStore(self.db, logger))
        self.register_store(PromptLogger(self.db, logger))
        self.register_store(ReportLogger(self.db, logger))

        # Register extra pluggable stores
        if cfg.extra_stores:
            for store in cfg.extra_stores:
                self.register_store(store)

    def register_store(self, store: BaseStore):
        if store.name in self._stores:
            raise ValueError(f"A store with name '{store.name}' is already registered.")
        self._stores[store.name] = store
        if self.logger:
            self.logger.log("StoreRegistered", {"store": store.name})

    def get(self, name: str) -> Optional[BaseStore]:
        return self._stores.get(name)

    def __getattr__(self, name):
        if name in self._stores:
            return self._stores[name]
        raise AttributeError(f"'MemoryTool' has no store named '{name}'")