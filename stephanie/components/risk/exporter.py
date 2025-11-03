# stephanie/components/risk/exporter.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


class HighRiskExporter:
    """
    Copies only high-risk visuals into the run's visuals subdir and
    writes a tiny index (risk_index.json).
    """
    def __init__(self, base_dir: str | Path, *, subdir: str = "visuals"):
        self.base = Path(base_dir)
        self.subdir = subdir

    def run_dir(self, run_id: str) -> Path:
        p = self.base / run_id / self.subdir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def maybe_export(
        self,
        *,
        run_id: str,
        record: Dict[str, Any],
        epi_paths: Optional[Dict[str, str]] = None,
        risk_gate: float = 0.60,
        allowed_labels: tuple[str, ...] = ("RISK",),
    ) -> Optional[Path]:
        decision = record.get("decision", "OK")
        risk_val = max(
            record.get("metrics", {}).get("faithfulness_risk01", 0.0),
            record.get("metrics", {}).get("delta_gap01", 0.0),
            record.get("metrics", {}).get("ood_hat01", 0.0),
            1.0 - record.get("metrics", {}).get("confidence01", 1.0),
        )
        if decision not in allowed_labels and risk_val < risk_gate:
            return None

        dst = self.run_dir(run_id)
        idx = {
            "run_id": run_id,
            "decision": decision,
            "metrics": record.get("metrics", {}),
            "paths": {},
        }
        # Copy referenced image files if present
        for k in ("field_path", "strip_path", "legend_path", "badge_path"):
            path = None
            if epi_paths and epi_paths.get(k):
                path = Path(epi_paths[k])
            elif record.get(k):  # in case you add them later to record
                path = Path(record[k])
            if path and path.exists():
                out = dst / path.name
                try:
                    out.write_bytes(path.read_bytes())
                except Exception:
                    pass
                idx["paths"][k] = str(out)

        # index
        (dst / "risk_index.json").write_text(
            __import__("json").dumps(idx, indent=2), encoding="utf-8"
        )
        return dst
