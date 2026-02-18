"""
run_policy_benchmark.py

Standalone execution script for PolicyContainer + policy.

This:
- Builds Certum summarization analyzer
- Wraps it in PolicyContainer
- Runs stability benchmark
- Outputs decisions and energy trend

Usage:
    python run_policy_benchmark.py --dataset halueval --limit 200
"""

import argparse
from pathlib import Path
import random
import matplotlib.pyplot as plt

from policy.evaluation.pipeline import run_summarization_pipeline
from policy.embedding.hf_embedder import HFEmbedder
from policy.embedding.sqlite_embedding_backend import SQLiteEmbeddingBackend
from policy.geometry.claim_evidence import ClaimEvidenceGeometry
from policy.geometry.sentence_support import SentenceSupportAnalyzer
from policy.geometry.nli_wrapper import EntailmentModel

from policy.policy_container import PolicyContainer
from policy.base_policy import BasePolicy
from policy.dominance import DominanceEngine
from policy.calibration import EnergyRegimeEstimator
from policy.monitor import EnergySpiralDetector
from policy.adapters.certum_adapter import CertumAdapter
from policy.experiments.benchmark import DynamicStabilityBenchmark


# ============================================================
# Dataset Config
# ============================================================

DATASETS = {
    "halueval": Path("E:/data/halueval_test_v1.jsonl"),
    "scifact": Path("E:/data/scifact_dev_rationale.jsonl"),
    "casehold": Path("E:/data/casehold_pos.jsonl"),
}


# ============================================================
# Helper: Load JSONL
# ============================================================

def load_jsonl(path: Path, limit=None):
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data[:limit] if limit else data


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="halueval")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--embedding_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding_db", type=str,
                        default="E:/data/global_embeddings.db")
    parser.add_argument("--nli_model", type=str,
                        default="MoritzLaurer/deberta-v3-base-mnli-fever-anli")
    args = parser.parse_args()

    dataset_path = DATASETS[args.dataset]

    print(f"Loading dataset: {args.dataset}")
    samples = load_jsonl(dataset_path, args.limit)

    # ============================================================
    # Build Certum Components
    # ============================================================

    embedding_backend = SQLiteEmbeddingBackend(args.embedding_db)
    embedder = HFEmbedder(args.embedding_model, embedding_backend)

    energy_computer = ClaimEvidenceGeometry()
    entailment_model = EntailmentModel(args.nli_model)

    analyzer = SentenceSupportAnalyzer(
        embedder=embedder,
        energy_computer=energy_computer,
        entailment_model=entailment_model,
        top_k=3,
    )

    # ============================================================
    # Define "System" callable
    # ============================================================

    def system(sample):
        support = analyzer.analyze(
            summary_text=sample["claim"],
            evidence_text=sample["evidence"][0],
        )
        sample["support_diagnostics"] = support
        return sample

    # ============================================================
    # Build Policy
    # ============================================================

    dominance_engine = DominanceEngine(
        critical_axes=["energy", "embedding_margin", "alignment"]
    )

    regime_estimator = EnergyRegimeEstimator(window=300)
    spiral_detector = EnergySpiralDetector(window=100)

    policy = BasePolicy(
        dominance_engine=dominance_engine,
        regime_estimator=regime_estimator,
        spiral_detector=spiral_detector,
    )

    adapter = CertumAdapter(analyzer)

    container = PolicyContainer(
        system=system,
        policy=policy,
        signal_adapter=adapter,
    )

    # ============================================================
    # Benchmark Loop
    # ============================================================

    benchmark = DynamicStabilityBenchmark(container, steps=len(samples))

    index = 0

    def generator():
        nonlocal index
        s = samples[index]
        index = (index + 1) % len(samples)
        return s

    print("Running policy benchmark...")
    results = benchmark.run(generator)

    decisions = results["decisions"]
    energies = results["energies"]

    # ============================================================
    # Stats
    # ============================================================

    accept_rate = decisions.count("accept") / len(decisions)
    reject_rate = decisions.count("reject") / len(decisions)
    freeze_rate = decisions.count("freeze") / len(decisions)

    print("\n===== POLICY RESULTS =====")
    print(f"ACCEPT: {accept_rate:.3f}")
    print(f"REJECT: {reject_rate:.3f}")
    print(f"FREEZE: {freeze_rate:.3f}")

    # ============================================================
    # Plot Energy Curve
    # ============================================================

    plt.figure(figsize=(10, 4))
    plt.plot(energies)
    plt.title("Energy Over Time (Policy Controlled)")
    plt.xlabel("Step")
    plt.ylabel("Mean Energy")
    plt.tight_layout()
    plt.savefig("policy_energy_curve.png")
    print("Saved energy plot: policy_energy_curve.png")


if __name__ == "__main__":
    main()
