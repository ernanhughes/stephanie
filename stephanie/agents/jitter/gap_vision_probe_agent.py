from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.services.gap_probe_service import (make_barbell,
                                                  make_ring_of_cliques,
                                                  make_sbm, run_default_suite,
                                                  run_probe)
from stephanie.utils.progress_mixin import ProgressMixin


class GapVisionProbeAgent(BaseAgent, ProgressMixin):
    """
    Context:
      {
        "out_dir": "gap_reports/vision_probes",
        "layouts": ["forceatlas2","spectral"],
        "suite": true,                # run default suite
        # OR provide a single graph spec:
        "graph": {"type":"barbell","n1":20,"n2":20,"m":4}
      }

    Output:
      {
        "report_path": "<dir>/probe_metrics.json",
        "results": [...],
        "images": [list of montage PNGs]
      }
    """

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        out_dir = Path(context.get("out_dir", "gap_reports/vision_probes"))
        layouts: List[str] = context.get("layouts", ["forceatlas2", "spectral"])

        images: List[str] = []
        if context.get("suite", True):
            self.logger.info("[GapVisionProbeAgent] Running default probe suite...")
            payload = run_default_suite(out_dir, layouts=layouts)
            images = [r["image"] for r in payload["results"]]
        else:
            self.logger.info("[GapVisionProbeAgent] Running single probe...")
            spec = context.get("graph", {"type": "barbell"})
            gtype = spec.get("type", "barbell")
            if gtype == "sbm":
                _, G, comms = make_sbm(**{k: v for k, v in spec.items() if k != "type"})
            elif gtype == "ring_of_cliques":
                _, G, comms = make_ring_of_cliques(**{k: v for k, v in spec.items() if k != "type"})
            else:
                _, G, comms = make_barbell(**{k: v for k, v in spec.items() if k != "type"})
            G.graph["name"] = gtype
            res = run_probe(G, comms, out_dir, layouts)
            payload = {"results": [res]}
            images = [res["image"]]
            (out_dir / "probe_metrics.json").write_text(self._dumps(payload), encoding="utf-8")

        # Optional: log to MemCube & VPM timeline if available
        try:
            mem = self.container.get("MemCubeService")
            mem.save_artifact(kind="gap_vision_probe", path=str(out_dir / "probe_metrics.json"), metadata={"layouts": layouts})
        except Exception:
            pass

        try:
            vpm_worker = self.container.get("VPMWorkerInline")
            if vpm_worker:
                # simple channel log: per-probe separability on FA2 vs spectral
                for res in payload["results"]:
                    run_id = f"gap_probe::{res['name']}"
                    chans = {}
                    if "forceatlas2" in res["metrics"]:
                        chans["sep.fa2"] = [res["metrics"]["forceatlas2"]["separability"]]
                    if "spectral" in res["metrics"]:
                        chans["sep.spectral"] = [res["metrics"]["spectral"]["separability"]]
                    if chans:
                        vpm_worker.add_channels(run_id, chans, namespace="probe")
        except Exception:
            pass

        return {
            "report_path": str(out_dir / "probe_metrics.json"),
            "results": payload["results"],
            "images": images,
        }


# from __future__ import annotations
# import argparse
# from pathlib import Path
# from stephanie.services.gap_probe_service import run_default_suite

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--out", type=str, default="gap_reports/vision_probes")
#     ap.add_argument("--layouts", type=str, default="forceatlas2,spectral")
#     args = ap.parse_args()

#     out_dir = Path(args.out)
#     layouts = [s.strip() for s in args.layouts.split(",") if s.strip()]
#     payload = run_default_suite(out_dir, layouts=tuple(layouts))
#     print(f"Saved report: {out_dir / 'probe_metrics.json'}")
#     for r in payload["results"]:
#         print(f"- {r['name']}: {r['image']}")

# if __name__ == "__main__":
#     main()
