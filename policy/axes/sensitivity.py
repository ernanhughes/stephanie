class SensitivityAxis:

    name = "sensitivity"

    def compute(self, context):
        energy_result = context["energy_result"]
        return energy_result.geometry.sensitivity
