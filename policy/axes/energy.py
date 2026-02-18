class EnergyAxis:

    def __init__(self, computer):
        self.computer = computer
        self.name = "energy"

    def compute(self, context):
        result = self.computer.compute(
            context["claim_vec"],
            context["evidence_vecs"],
        )

        context["energy_result"] = result
        return result.energy
