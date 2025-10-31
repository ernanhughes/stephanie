# stephanie/components/ssp/component.py
from __future__ import annotations

import uuid
from typing import Dict, Any, Optional

from stephanie.components.ssp.services.telemetry import SSPTelemetry
# from stephanie.components.ssp.actors import Proposer, Solver, Verifier  # your actual actors

class SSPComponent:
    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        # Telemetry
        tcfg = (self.cfg.get("telemetry") or {}) if isinstance(self.cfg, dict) else {}
        self.telemetry = SSPTelemetry(tcfg, memory, logger, subject_root="ssp")

        # self.proposer = Proposer(cfg, memory, container, logger)
        # self.solver   = Solver(cfg, memory, container, logger)
        # self.verifier = Verifier(cfg, memory, container, logger)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Correlation id for this SSP run
        ssp_run_id = context.get("pipeline_run_id")
        context["ssp_run_id"] = ssp_run_id

        # --- RUN START
        await self.telemetry.publish(
            "ssp.run.start",
            {"cfg": {"telemetry": self.cfg.get("telemetry")}, "note": "SSP run started"},
            context=context,
        )

        try:
            # Example: proposer stage
            async with self.telemetry.span(context={**context, "actor": "proposer"}, name="actor.propose"):
                # proposals = await self.proposer.propose(context)
                proposals = context.get("proposals", [])  # stub
                await self.telemetry.publish(
                    "ssp.actor.propose.result",
                    {"count": len(proposals)},
                    context={**context, "actor": "proposer"},
                )

            # solver stage
            async with self.telemetry.span(context={**context, "actor": "solver"}, name="actor.solve"):
                # solutions = await self.solver.solve(context, proposals)
                solutions = context.get("solutions", [])  # stub
                await self.telemetry.publish(
                    "ssp.actor.solve.result",
                    {"count": len(solutions)},
                    context={**context, "actor": "solver"},
                )

            # verifier stage
            async with self.telemetry.span(context={**context, "actor": "verifier"}, name="actor.verify"):
                # verdicts = await self.verifier.verify(context, solutions)
                verdicts = context.get("verdicts", [])  # stub
                await self.telemetry.publish(
                    "ssp.actor.verify.result",
                    {"count": len(verdicts)},
                    context={**context, "actor": "verifier"},
                )

            # Aggregate + emit a final outcome
            await self.telemetry.publish(
                "ssp.run.result",
                {
                    "summary": {
                        "proposals": len(context.get("proposals", [])),
                        "solutions": len(context.get("solutions", [])),
                        "verdicts": len(context.get("verdicts", [])),
                    }
                },
                context=context,
            )

        except Exception as e:
            await self.telemetry.publish(
                "ssp.run.error",
                {"error": str(e)},
                context=context,
            )
            raise

        finally:
            # --- RUN END
            await self.telemetry.publish(
                "ssp.run.end",
                {"note": "SSP run completed"},
                context=context,
            )

        return context
