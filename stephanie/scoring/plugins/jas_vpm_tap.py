# stephanie/scoring/plugins/jas_vpm_tap.py
from __future__ import annotations

import asyncio

from .registry import register


@register("vpm_tap")
class VPMTapPlugin:
    """
    After each tick, write a single VPM row to DB and (optionally) push to ZeroModel timeline.
    """
    def __init__(self, container=None, logger=None, host=None, run_id: str = "jitter", log_to_zero: bool = True):
        self.container = container; self.logger = logger; self.host = host
        self.run_id = run_id; self.log_to_zero = bool(log_to_zero)
        self._timeline_open = False

    def post_tick(self, tick: int, vitals, core, triune):
        vpms = self.container.get("vpm_store")
        zm = self.container.get("zeromodel")

        metrics = ["energy.cognitive","energy.metabolic","energy.reserve","boundary.integrity","health"]
        values  = [
            float(vitals.energy_cognitive), float(vitals.energy_metabolic), float(vitals.energy_reserve),
            float(vitals.boundary_integrity), float(vitals.health_score),
        ]

        # persist VPM row
        vpms.insert_row(
            run_id=self.run_id,
            step=int(tick),
            metric_names=metrics,
            values=values,
            extra={"tick": int(tick)}
        )

        # append to ZeroModel timeline (for GIFs)
        if zm:
            if not self._timeline_open:
                zm.timeline_open(self.run_id, metrics=metrics)
                self._timeline_open = True
            zm.timeline_append_row(self.run_id, metrics_columns=metrics, metrics_values=values)

    def on_shutdown(self, **_):
        try:
            zm = self.container.get("zeromodel-service-v2")
            if self._timeline_open and zm:
                # finalize asynchronously if your runtime allows; otherwise, call without await in sync context
                asyncio.create_task(zm.timeline_finalize(self.run_id, out_path="data/vpms"))
        except Exception:
            pass
