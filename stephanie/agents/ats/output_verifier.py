# stephanie/agents/ats/output_verifier.py

from typing import Optional


class OutputVerifier:
    def verify(self, output: str, has_submission_file: bool) -> dict[str, any]:
        is_bug = "Error" in output or "Exception" in output
        is_overfitting = "val_loss increasing" in output.lower()
        metric = self.extract_metric(output)

        return {
            "is_bug": is_bug,
            "is_overfitting": is_overfitting,
            "has_csv_submission": has_submission_file,
            "metric": metric,
            "summary": self.summarize(output)
        }

    def extract_metric(self, output: str) -> Optional[float]:
        # Implement logic to parse metric value from logs
        return 0.85  # Example

    def summarize(self, output: str) -> str:
        # Summarize key findings from output
        return "Model trained successfully."