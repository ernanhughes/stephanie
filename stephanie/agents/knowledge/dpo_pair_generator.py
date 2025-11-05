# stephanie/agents/knowledge/dpo_pair_generator.py
from __future__ import annotations

import asyncio
import json
import logging
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.knowledge.casebook_store import CaseBookStore
from stephanie.utils.json_sanitize import safe_json

log = logging.getLogger(__name__)

class DPOPairGeneratorAgent(BaseAgent):
    """
    Generates (chosen, rejected) text pairs from scored improvements.
    Designed to feed RL/DPO pipelines.
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, container, logger: logging.Logger):
        super().__init__(cfg, memory, container, logger)
        self.min_improvement_score = cfg.get("dpo_min_improvement", 0.1)
        self.max_pairs_per_run = cfg.get("dpo_max_pairs", 10)
        self.auto_publish = cfg.get("dpo_auto_publish", True)
        self.output_dir = Path(cfg.get("dpo_output_dir", "data/dpo_pairs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log.debug(
            "DPOPairGeneratorAgent initialized | "
            f"output_dir={self.output_dir} | "
            f"min_improvement_score={self.min_improvement_score} | "
            f"auto_publish={self.auto_publish}"
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        generated_pairs = []

        # Pull identifiers from context (fallbacks are last resort)
        casebook_name = context.get("casebook_name") or context.get("casebook") or "default"
        case_id = context.get("case_id")
        if not case_id:
            self.logger.log("DPOGenerationSkipped", {"reason": "missing_case_id", "casebook": casebook_name})
            return context

        try:
            # Retrieve artifacts
            final_text = self._get_final_draft(casebook_name, case_id)
            initial_text = self._get_initial_draft(casebook_name, case_id)
            vpm_row, vpm_meta = self._get_vpm_row(casebook_name, case_id)
            goal_eval = self._get_goal_eval(vpm_meta)  # uses meta.goal if present

            if not final_text or not initial_text:
                self.logger.log("DPOGenerationSkipped", {
                    "reason": "missing_texts",
                    "has_final": bool(final_text),
                    "has_initial": bool(initial_text),
                    "casebook": casebook_name,
                    "case_id": case_id
                })
                return context

            # Compute improvement
            initial_goal_score = float(context.get("initial_goal_score", 0.0))
            final_goal_score = float(goal_eval.get("score", 0.0))
            score_delta = final_goal_score - initial_goal_score

            if score_delta < float(self.min_improvement_score):
                self.logger.log("DPOGenerationSkipped", {
                    "reason": "insufficient_improvement",
                    "delta": score_delta,
                    "threshold": self.min_improvement_score,
                    "initial_goal_score": initial_goal_score,
                    "final_goal_score": final_goal_score
                })
                return context

            # Build DPO pair
            pair_id = f"dpo_{uuid.uuid4().hex[:8]}"
            prompt = (
                context.get("knowledge_plan", {}).get("section_title")
                or context.get("prompt")
                or "Improve this section."
            )

            pair = {
                "id": pair_id,
                "prompt": prompt,
                "chosen": final_text.strip(),
                "rejected": initial_text.strip(),
                "metadata": {
                    "casebook": casebook_name,
                    "case_id": case_id,
                    "improvement_score": score_delta,
                    "final_score": final_goal_score,
                    "initial_score": initial_goal_score,
                    "vpm_row": vpm_row,
                    "goal_template": context.get("goal_template"),
                    "generation_style": context.get("generation_style"),
                    "tags": context.get("tags", []),
                    "source_agent": "TextImproverAgent",
                },
            }

            # Persist
            pair_path = self.output_dir / f"{pair_id}.json"
            with pair_path.open("w", encoding="utf-8") as f:
                json.dump(pair, f, indent=2)

            generated_pairs.append(pair)

            # Publish (works with sync or async bus)
            if self.auto_publish:
                await self._safe_publish("dpo.pair.generated", {
                    "pair_id": pair_id,
                    "path": str(pair_path),
                    "improvement_score": score_delta,
                    "case_id": case_id
                })

            self.logger.log("DPOPairGenerated", {
                "pair_id": pair_id,
                "improvement": score_delta,
                "case_id": case_id
            })

            # Update context
            context["dpo_pair"] = pair
            context["dpo_pair_path"] = str(pair_path)
            return context

        except Exception as e:
            self.logger.log("DPOPairGenerationFailed", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return context

    # -------------------------- helpers --------------------------

    async def _safe_publish(self, subject: str, payload: Dict[str, Any]) -> None:
        """Publish via whichever bus is available; handle sync/async seamlessly."""
        bus = getattr(self, "kb", None) or getattr(self.memory, "bus", None)
        if not bus:
            return
        try:
            res = bus.publish(subject, payload)
            if asyncio.iscoroutine(res):
                await res
        except Exception as e:
            self.logger.log("DPOBusPublishFailed", {"error": str(e), "subject": subject})

    def _get_final_draft(self, casebook_name: str, case_id: int) -> Optional[str]:
        items = self.memory.casebooks.get_by_case(
            casebook_name=casebook_name,
            case_id=case_id,
            role="text",
            meta_filter={"stage": "final"},
            limit=1,
        )
        return items[0].text if items else None

    def _get_initial_draft(self, casebook_name: str, case_id: int) -> Optional[str]:
        # 1) Prefer DB: role="text", meta.stage="initial"
        items = self.memory.casebooks.get_by_case(
            casebook_name=casebook_name,
            case_id=case_id,
            role="text",
            meta_filter={"stage": "initial"},
            limit=1,
        )
        if items:
            return items[0].text

        # 2) Fallback: local file from run_dir (read from case meta or scorable meta)
        run_dir = self._get_run_dir(casebook_name, case_id)
        initial_path = run_dir / "initial_draft.md" if run_dir else None
        if initial_path and initial_path.exists():
            return initial_path.read_text(encoding="utf-8")

        return None

    def _get_vpm_row(self, casebook_name: str, case_id: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Returns (vpm_row_dict, vpm_meta_dict).
        vpm_row is parsed from DynamicScorable.text (JSON).
        vpm_meta is taken from DynamicScorable.meta (should include 'goal').
        """
        items = self.memory.casebooks.get_by_case(
            casebook_name=casebook_name,
            case_id=case_id,
            role="vpm",
            limit=1,
        )
        if not items:
            return {}, {}

        row = items[0]
        vpm_row = {}
        try:
            vpm_row = json.loads(row.text or "{}")
        except Exception:
            vpm_row = {}

        vpm_meta = row.meta or {}
        return vpm_row, vpm_meta

    def _get_goal_eval(self, vpm_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Extract goal evaluation dict from VPM meta (fallback to empty)."""
        if isinstance(vpm_meta, dict):
            goal = vpm_meta.get("goal")
            if isinstance(goal, dict):
                return goal
        return {"score": 0.0}

    def _get_run_dir(self, casebook_name: str, case_id: int) -> Optional[Path]:
        """
        Try, in order:
          1) Case.meta['run_dir']
          2) Any vpm/text scorable meta with 'run_dir'
        """
        # 1) Case meta
        case = self.memory.casebooks.get_case_by_id(case_id)
        if case and getattr(case, "meta", None):
            rd = case.meta.get("run_dir")
            if rd:
                return Path(rd)

        # 2) Hunt in scorables
        for role in ("vpm", "text"):
            items = self.memory.casebooks.get_by_case(
                casebook_name=casebook_name,
                case_id=case_id,
                role=role,
                limit=5,
            )
            for it in items:
                if getattr(it, "meta", None) and it.meta.get("run_dir"):
                    return Path(it.meta["run_dir"])

        return None
