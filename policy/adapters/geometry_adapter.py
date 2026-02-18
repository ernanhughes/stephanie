# policy/adapters/geometry_adapter.py

from typing import Dict

from policy.geometry.claim_evidence import ClaimEvidenceGeometry
from policy.protocols.embedder import Embedder


class GeometryEnergyAdapter:
    """
    Wraps ClaimEvidenceGeometry
    into a policy-compatible energy function.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.geometry = ClaimEvidenceGeometry()

    def compute_energy(self, output: Dict, context: Dict) -> float:

        claim = output["claim"]
        evidence = output["evidence"]

        claim_vec = self.embedder.embed([claim])[0]
        evidence_vecs = self.embedder.embed(evidence)

        result = self.geometry.compute(claim_vec, evidence_vecs)

        return float(result.energy)
