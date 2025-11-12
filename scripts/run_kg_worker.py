# scripts/run_kg_worker.py
from stephanie.core.logging.json_logger import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from stephanie.services.workers.kg_indexer import KnowledgeGraphIndexerWorker
import yaml

def load_config(path="/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    cfg = load_config()
    logger = JSONLogger("logs/kg.jsonl")
    memory = MemoryTool(cfg=cfg, logger=logger)
    worker = KnowledgeGraphIndexerWorker(cfg, memory)
    worker.start()