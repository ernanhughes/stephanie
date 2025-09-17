# scripts/reset_hnsw_index.py
import os
import logging
from stephanie.models.ner_retriever import NERRetrieverEmbedder
from stephanie.scoring.scorable_factory import TargetType, ScorableFactory
from stephanie.scoring.scorable import Scorable
from stephanie.memory import init_memory   # adjust if your project imports differ

logger = logging.getLogger("reset_hnsw")
logging.basicConfig(level=logging.INFO)

INDEX_PATH = "data/ner_retriever/index"   # <-- adjust if your config points elsewhere

def reset_index(index_path=INDEX_PATH):
    bin_file = f"{index_path}.bin"
    meta_file = f"{index_path}_metadata.json"

    for f in [bin_file, meta_file]:
        if os.path.exists(f):
            os.remove(f)
            logger.info(f"Deleted {f}")
        else:
            logger.info(f"No file to delete: {f}")

def rebuild_index(memory, index_path=INDEX_PATH):
    logger.info("Rebuilding HNSW index...")

    # init retriever
    retriever = NERRetrieverEmbedder(
        model_name="meta-llama/Llama-3.2-1B-Instruct", 
        layer=17,
        device="cuda",
        embedding_dim=2048,
        index_path=index_path,
        projection_enabled=False,
        logger=logger,
        memory=memory,
    )

    # collect scorables (documents + chat)
    scorables = []
    for doc in memory.documents.all():
        for sec in memory.document_sections.get_by_document(doc.id) or []:
            if sec.section_text and len(sec.section_text) > 20:
                scorables.append(Scorable(id=f"{doc.id}:{sec.section_name}", 
                                          text=sec.section_text, 
                                          target_type=TargetType.DOCUMENT))

    for msg in memory.chats.get_top_conversations(limit=50):
        scorable = ScorableFactory.from_orm(msg[0])
        scorables.append(scorable)

    # index them
    total = retriever.index_scorables(scorables)
    logger.info(f"✅ Indexed {total} entities into fresh HNSW index")

if __name__ == "__main__":
    memory = init_memory()   # depends on your project; adjust if different
    reset_index()
    rebuild_index(memory)
