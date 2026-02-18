
from policy.experiments.runaway_report import RunawayReportHarness
from policy.policy_container import PolicyContainer


# -----------------------
# Dummy AI
# -----------------------

def stephanie_sim(state):
    return {
        "claim": f"quality={state['quality']}",
        "evidence": ["some evidence text"]
    }


# -----------------------
# Dummy Energy Function
# -----------------------

def dummy_energy(output, context):
    q = float(output["claim"].split("=")[1])
    return abs(q - 1.0)


# -----------------------
# Dummy Policy (accept low energy)
# -----------------------

class SimplePolicy:
    def __init__(self, tau=0.2):
        self.tau = tau

    def decide(self, energy):
        return energy < self.tau


policy = PolicyContainer(
    ai_callable=stephanie_sim,
    energy_function=dummy_energy,
    calibrator=None,
    calibration=None,
)

harness = RunawayReportHarness(
    ai_callable=stephanie_sim,
    energy_function=dummy_energy,
    policy_container=policy,
    episodes=1000,
)

report = harness.run()
print(report)
