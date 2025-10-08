from __future__ import annotations

from typing import Any, Dict, Optional


class OutputVerifier:
    """
    Modern verifier for structured score dictionaries.
    Validates expected dimensions and enforces thresholds.
    """

    def __init__(self,
                 thresholds: Optional[Dict[str, float]] = None,
                 require_all: bool = True):
        """
        thresholds: e.g. {"alignment": 0.9, "clarity": 0.7}
        require_all: if True, fail verification if any threshold unmet
        """
        self.thresholds = thresholds or {"alignment": 0.9}
        self.require_all = require_all

    # --------------------------------------------------------------
    def verify(self,
               result: Dict[str, Any],
               stderr: str = "",
               has_submission_file: bool = False) -> Dict[str, Any]:
        """
        Validate a structured task result dict.
        Returns a standardized record with flags.
        """

        # --- Safety ---
        if not isinstance(result, dict):
            return {
                "metric": 0.0,
                "summary": "Invalid result type",
                "is_bug": True,
                "is_verified": False,
                "merged_output": str(result)
            }

        metric = float(result.get("metric", 0.0))
        vector = result.get("vector", {}) or {}
        summary = result.get("summary", "")
        merged_output = result.get("merged_output", "")

        # --- Check thresholds ---
        flags = {}
        for dim, thresh in self.thresholds.items():
            score = self._find_score_for(dim, vector)
            ok = (score is not None) and (score >= thresh)
            flags[dim] = {
                "score": score,
                "threshold": thresh,
                "ok": ok
            }

        # Aggregate success/failure
        ok_flags = [f["ok"] for f in flags.values()]
        verified = all(ok_flags) if self.require_all else any(ok_flags)

        # --- Compute verification metric (optional) ---
        # Example: average over verified dimensions
        verified_scores = [f["score"] for f in flags.values() if f["score"] is not None]
        verify_metric = sum(verified_scores) / len(verified_scores) if verified_scores else metric

        return {
            "metric": verify_metric,
            "summary": summary,
            "merged_output": merged_output,
            "is_bug": not verified,
            "is_verified": verified,
            "flags": flags
        }

    # --------------------------------------------------------------
    def _find_score_for(self, dim: str, vector: Dict[str, float]) -> Optional[float]:
        """Find a score key containing the given dimension name."""
        for k, v in vector.items():
            if dim.lower() in k.lower():
                try:
                    return float(v)
                except Exception:
                    pass
        return None
