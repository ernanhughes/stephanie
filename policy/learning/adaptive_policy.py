# policy/learning/adaptive_policy.py

from policy.policy_container import PolicyContainer


class AdaptivePolicyLearner:
    """
    Updates calibration dynamically
    based on accepted samples.
    """

    def __init__(self, policy: PolicyContainer):
        self.policy = policy

    def update_from_history(self):

        energies = self.policy.recent_energies

        if len(energies) < 50:
            return

        positive = energies[-200:]

        self.policy.recalibrate(
            positive_energies=positive,
            hard_negative_energies=None,
        )
