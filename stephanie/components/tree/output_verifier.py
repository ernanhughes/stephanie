# stephanie/components/tree/output_verifier.py

"""
Evaluates stdout/stderr from executed or generated tasks.
Extracts numeric metrics and detects anomalies (bugs, overfitting, etc.).
"""

from __future__ import annotations
import re
from typing import Any, Dict


class OutputVerifier:
    _METRIC_PATTERNS = [
        r"val[_\s]?accuracy[:=]\s*([0-9]*\.?[0-9]+%?)",
        r"accuracy[:=]\s*([0-9]*\.?[0-9]+%?)",
        r"f1[_\s]?(score)?[:=]\s*([0-9]*\.?[0-9]+%?)",
        r"auc[:=]\s*([0-9]*\.?[0-9]+%?)",
        r"precision[:=]\s*([0-9]*\.?[0-9]+%?)",
        r"recall[:=]\s*([0-9]*\.?[0-9]+%?)",
        r"loss[:=]\s*([0-9]*\.?[0-9]+)",
        r"rmse[:=]\s*([0-9]*\.?[0-9]+)",
        r"mae[:=]\s*([0-9]*\.?[0-9]+)",
        r"mse[:=]\s*([0-9]*\.?[0-9]+)",
        r"score[:=]\s*([0-9]*\.?[0-9]+%?)",
        r"metric[:=]\s*([0-9]*\.?[0-9]+%?)",
    ]

    def __init__(self, prefer_higher: bool = True):
        self.prefer_higher = prefer_higher
        self.name = "OutputVerifier-v1.0"

    def verify(self, stdout: str, stderr: str, has_submission_file: bool) -> Dict[str, Any]:
        merged = self._merge_streams(stdout, stderr)
        lower = merged.lower()

        is_bug = any(kw in merged for kw in ("Traceback", "Exception", "Error", "RuntimeWarning"))
        is_overfitting = any(p in lower for p in ("val_loss increasing", "overfit", "unstable training"))
        metric, all_metrics = self.extract_metrics(merged)
        summary = self.summarize(merged)

        if metric is not None and not self.prefer_higher:
            metric = 1.0 - metric

        return {
            "verifier_name": self.name,
            "is_bug": is_bug,
            "is_overfitting": is_overfitting,
            "has_csv_submission": has_submission_file,
            "metric": metric,
            "metrics_found": all_metrics,
            "summary": summary,
            "merged_output": merged,
        }

    def extract_metrics(self, text: str):
        metrics = []
        for pattern in self._METRIC_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                val = m.groups()[-1]
                try:
                    if val.endswith("%"):
                        num = float(val[:-1]) / 100.0
                    else:
                        num = float(val)
                        if any(k in pattern.lower() for k in ("rmse", "mae", "mse", "loss")):
                            num = 1.0 / (1.0 + num)
                    metrics.append(num)
                except Exception:
                    continue
        return (metrics[-1] if metrics else None, metrics)

    def summarize(self, text: str, tail_lines: int = 8, max_chars: int = 400) -> str:
        lines = [ln.strip() for ln in text.splitlines()[-tail_lines:] if ln.strip()]
        summary = " ".join(lines) or "No output."
        return summary[: max_chars - 3] + "..." if len(summary) > max_chars else summary

    @staticmethod
    def _merge_streams(stdout: str, stderr: str) -> str:
        if not stderr:
            return stdout or ""
        if not stdout:
            return stderr or ""
        return f"{stdout}\n--- STDERR ---\n{stderr}"
