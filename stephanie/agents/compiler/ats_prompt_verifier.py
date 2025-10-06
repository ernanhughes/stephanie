# stephanie/agents/compiler/ats_prompt_verifier.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional


class CompilerVerifier:
    """
    Verifier for prompt/pipeline executions. Expects `stdout` to be a JSON
    string with a numeric `score` field (0..1). Returns the normalized metric
    and a short summary text for the search UI.
    """

    def verify(self, stdout: str, has_submission_file: bool) -> Dict[str, Any]:
        metric: Optional[float] = None
        summary: str = ""
        try:
            obj = json.loads(stdout) if stdout else {}
            metric = obj.get("score")
            summary = (obj.get("selected_text") or "")[:500]
        except Exception:
            summary = (stdout or "")[:500]

        def clamp01(x: Optional[float]) -> float:
            if x is None:
                return 0.0
            try:
                return max(0.0, min(1.0, float(x)))
            except Exception:
                return 0.0

        return {
            "is_bug": False,
            "is_overfitting": False,
            "has_csv_submission": has_submission_file,
            "metric": clamp01(metric),
            "summary": summary,
            # AgenticTreeSearch expects this key in addition to the fields above
            "merged_output": stdout or "",
        }
