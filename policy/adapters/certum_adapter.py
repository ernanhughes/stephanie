from policy.governance_signal import GovernanceSignal


class CertumAdapter:

    def __init__(self, support_analyzer):
        self.support_analyzer = support_analyzer

    def __call__(self, result) -> GovernanceSignal:
        sd = result.support_diagnostics

        # Normalize energy (lower raw -> higher normalized)
        normalized_energy = 1.0 - min(max(sd.mean_energy, 0.0), 1.0)

        return GovernanceSignal(
            energy=normalized_energy,
            embedding_margin=float(sd.mean_sim_margin),
            alignment=float(sd.max_entailment),
            metadata=sd.to_dict(),
        )
