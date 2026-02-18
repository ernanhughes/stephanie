class ParticipationRatioAxis:

    name = "participation_ratio"

    def compute(self, context):
        energy_result = context["energy_result"]
        return energy_result.geometry.participation_ratio
