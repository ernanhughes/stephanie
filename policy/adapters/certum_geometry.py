
from policy.governance_signal import from_support_diagnostics


class CertumEnergyAdapter:
    """
    Wraps ClaimEvidenceGeometry + EntailmentModel
    and returns GovernanceSignal.
    """

    def __init__(self, support_analyzer):
        self.support_analyzer = support_analyzer

    def compute_signal(self, summary_text, evidence_text):
        sd = self.support_analyzer.analyze(summary_text, evidence_text)
        return from_support_diagnostics(sd)
