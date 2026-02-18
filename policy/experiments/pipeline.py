# src/certum/evaluation/pipeline.py

from pathlib import Path
import logging

from policy.embedding.hf_embedder import HFEmbedder
from policy.embedding.sqlite_embedding_backend import SQLiteEmbeddingBackend
from policy.geometry.claim_evidence import ClaimEvidenceGeometry
from policy.geometry.nli_wrapper import EntailmentModel
from policy.geometry.sentence_support import SentenceSupportAnalyzer
from policy.orchestration.summarization_runner import SummarizationRunner


logger = logging.getLogger(__name__)


# =========================================================
# Evaluation Pipeline Builder
# =========================================================

def run_summarization_pipeline(
    *,
    samples: list,
    embedding_model: str,
    embedding_db: Path,
    nli_model: str,
    entailment_db: Path,
    top_k: int,
    geometry_top_k: int,
    rank_r: int,
    out_path: Path,
) -> list:
    """
    Builds full summarization evaluation stack and executes it.

    Returns structured results list.
    """

    # -----------------------------------------------------
    # Embedding Backend
    # -----------------------------------------------------

    backend = SQLiteEmbeddingBackend(str(embedding_db))
    logger.info(f"Using embedding DB: {embedding_db}")

    embedder = HFEmbedder(
        embedding_model,
        backend=backend,
    )

    # -----------------------------------------------------
    # Geometry
    # -----------------------------------------------------

    energy_computer = ClaimEvidenceGeometry(
        top_k=geometry_top_k,
        rank_r=rank_r,
    )

    # -----------------------------------------------------
    # Entailment
    # -----------------------------------------------------

    entailment_model = EntailmentModel(
        model_name=nli_model,
        batch_size=32,
        db_path=str(entailment_db),
    )

    # -----------------------------------------------------
    # Sentence Support Analyzer
    # -----------------------------------------------------

    support_analyzer = SentenceSupportAnalyzer(
        embedder=embedder,
        energy_computer=energy_computer,
        entailment_model=entailment_model,
        top_k=top_k,
    )

    # -----------------------------------------------------
    # Summarization Runner
    # -----------------------------------------------------

    summarization_runner = SummarizationRunner(
        support_analyzer=support_analyzer
    )

    logger.info("Running summarization pipeline...")

    results = summarization_runner.run(
        samples,
        out_path=out_path,
    )

    return results
