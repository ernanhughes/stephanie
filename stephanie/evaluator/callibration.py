# stephanie/evaluator/callibration.py
# evaluators/calibration.py

import torch


class MRQCalibrator:
    @staticmethod
    def calibrated_preference(value_a: float, value_b: float) -> float:
        """
        Returns a probability (0 to 1) that A is preferred over B.
        """
        return torch.sigmoid(torch.tensor(value_a - value_b)).item()

    @staticmethod
    def predicted_preference(value_a: float, value_b: float) -> str:
        """
        Returns 'a' or 'b' based on which value is higher.
        """
        return "a" if value_a >= value_b else "b"
