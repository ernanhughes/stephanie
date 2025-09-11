# DraftGeneratorAgent — knowledge-first, VPM-powered draft trajectories from paper + chat
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL
from stephanie.models.casebook import CaseBookORM

from stephanie.agents.paper_improver import TextImprover
from stephanie.agents.paper_improver.vpm_controller import VPMController, VPMRow, Signal, default_controller
from stephanie.agents.paper_improver import GoalScorer

# 🔑 new: knowledge fusion (transient NER + 20 domains from paper + chat)
from stephanie.agents.paper_improver.knowledge_fuser import KnowledgeFuser


class DraftGeneratorAgent(BaseAgent):
    """
    Generates a blog section as an *improvement trajectory*:
      1) Fuse paper text + chat history into a content plan (transient NER + 20 domains)
      2) Iterate drafts via TextImprover → VPM rows
      3) Use VPMController to decide EDIT/RESAMPLE/STOP
      4) Store every step in a CaseBook (for SIS + PACS/CBR)
    """

    def __init__(self, cfg, memory, logger, reporter=None):
        super().__init__(cfg, memory, logger)
        self.max_steps = cfg.get("max_steps", 5)
        self.goal_template = cfg.get("goal_template", "academic_summary")
        self.section_name_fallback = cfg.get("section_name_fallback", "Blog Section")
        self.vpm: VPMController = default_controller()
        self.goals = GoalScorer()
        self.ti = TextImprover(workdir=cfg.get("text_workdir", "./text_runs"))
        self.fuser = KnowledgeFuser(cfg, memory, logger)

        # chat ingestion options
        self.chat_max_messages = cfg.get("chat_max_messages", 200)
        self.chat_from_context_key = cfg.get("chat_context_key", "chat_messages")  # optional injection path

    async def run(self, context: dict) -> dict:
        """
        Expects in context:
          - paper: {"id","title","text" or "section_text"}
          - section: {"section_name","section_text"} (optional; will fallback to paper text)
          - goal: GOAL struct
          - chat_messages (optional): [{"role":"user/assistant","text": "...", "ts":...}, ...]
        Produces:
          - casebook_id, case_id
          - plan (fused)
          - trajectory [steps]
          - champion_draft (text), champion_vpm (dict)
        """
        goal = context.get(GOAL, {})
        paper = context.get("paper", {})
        section = context.get("section", {}) or {}

        # --------- Gather inputs ----------
        section_name = section.get("section_name") or self.section_name_fallback
        paper_text = section.get("section_text") or paper.get("text") or paper.get("content") or ""
        chat_messages = self._collect_chat_messages(context)

        self.report({
            "event": "start",
            "step": "DraftGenerator",
            "details": f"Generating trajectory for '{section_name}'",
            "paper_title": paper.get("title", "Unknown"),
            "chat_messages": len(chat_messages)
        })

        # --------- Fuse knowledge (paper + chat) → plan ----------
        plan = self.fuser.fuse(
            text=paper_text,
            chat_messages=chat_messages,
            section_name=section_name
        )
        # Add explicit goal_template so TextImprover and downstream scorers can use it
        plan["goal_template"] = self.goal_template

        # --------- CaseBook to log the trajectory ----------
        casebook = self._ensure_casebook(paper, section_name, plan)
        context["casebook_id"] = casebook.id

        # --------- Trajectory loop ----------
        trajectory: List[Dict[str, Any]] = []
        last_decision: Optional[Signal] = None
        current_vpm_row: Optional[Dict[str, float]] = None
        last_result: Optional[Dict[str, Any]] = None

        for step in range(self.max_steps):
            # Generate/Improve draft
            result = self.ti.improve(plan)  # TextImprover is self-contained; we rerun with the (improved) plan
            last_result = result

            # Extract VPM dims into controller row
            dims = self._vpm_dims_from_text_improver(result)
            vpm_row = VPMRow(
                unit=f"blog:{section_name}",
                kind="text",
                timestamp=time.time(),
                step_idx=step,
                dims=dims,
                meta={"exemplar_id": None}
            )

            # Controller decision
            decision = self.vpm.add(vpm_row, candidate_exemplars=None)
            last_decision = decision.signal

            # Log step to CaseBook
            draft_path = Path(result["final_draft_path"])
            draft_text = draft_path.read_text() if draft_path.exists() else ""
            traj_rec = self._log_step(casebook, plan, result, step, decision, draft_text)
            trajectory.append(traj_rec)

            # STOP / ESCALATE gates
            if decision.signal in (Signal.STOP, Signal.ESCALATE):
                break

        # --------- Champion selection (best goal score) ----------
        champion = self._select_champion(trajectory)

        # Final champion scorable
        self.memory.casebooks.add_scorable(
            case_id=champion["case_id"],
            scorable_id=str(uuid.uuid4()),
            text=champion["draft_text"],
            role="champion",
            meta={"goal_score": champion["goal_score"], "step": champion["step"]}
        )

        # --------- Reflection ----------
        self._log_reflection(casebook, section_name, trajectory, champion)

        # --------- Emit context ----------
        context.update({
            "plan": plan,
            "trajectory": trajectory,
            "champion_step": champion["step"],
            "champion_draft": champion["draft_text"],
            "champion_vpm": champion["vpm_row"],
            "champion_goal_score": champion["goal_score"],
            "case_id": champion["case_id"],
        })

        self.report({
            "event": "end",
            "step": "DraftGenerator",
            "details": f"Trajectory steps: {len(trajectory)}",
            "champion_step": champion["step"],
            "casebook_id": casebook.id
        })
        return context

    # ------------- helpers -------------

    def _collect_chat_messages(self, context: dict) -> List[Dict[str, Any]]:
        """
        Non-persistent transient chat capture. Priority:
          1) context["chat_messages"] if provided
          2) memory.chat.last_n(self.chat_max_messages) if available
        """
        msgs = context.get(self.chat_from_context_key)
        if isinstance(msgs, list) and msgs:
            return msgs[-self.chat_max_messages:]

        # Optional: pull from your memory adapter if exposed
        try:
            if hasattr(self.memory, "chat") and hasattr(self.memory.chat, "last_n"):
                return self.memory.chat.last_n(self.chat_max_messages)  # type: ignore
        except Exception:
            pass
        return []

    def _ensure_casebook(self, paper: dict, section_name: str, plan: dict) -> CaseBookORM:
        casebook_name = f"blog_{paper.get('id','unknown')}_{section_name}_{int(time.time())}"
        meta = {
            "paper_id": paper.get("id"),
            "paper_title": paper.get("title"),
            "section_name": section_name,
            "domains": plan.get("domains", []),
            "knowledge_hash": plan.get("meta", {}).get("knowledge_hash"),
            "transient": True,   # ⚠️ transient domains/NER; not persisted elsewhere
        }
        return self.memory.casebooks.ensure_casebook(
            name=casebook_name,
            description=f"Draft trajectory for '{section_name}' from fused knowledge",
            tag="draft_generator",
            meta=meta
        )

    def _vpm_dims_from_text_improver(self, result: Dict[str, Any]) -> Dict[str, float]:
        row = result.get("vpm_row", {})
        # Normalize keys used by controller
        return {
            "coverage": float(row.get("coverage_final", row.get("coverage", 0.0))),
            "correctness": float(row.get("correctness", 0.0)),
            "coherence": float(row.get("coherence", 0.0)),
            "citation_support": float(row.get("citation_support", 0.0)),
            "entity_consistency": float(row.get("entity_consistency", 0.0)),
            "readability": float(row.get("readability", 10.0)),  # FKGL band, but controller treats as float
            "novelty": float(row.get("novelty", 0.5)),
            # stickiness lives in TextImprover.scores; fall back to 0.5 if absent
            "stickiness": float(result.get("scores", {}).get("stickiness", 0.5)),
        }

    def _log_step(self, casebook: CaseBookORM, plan: dict, result: dict, step: int, decision, draft_text: str) -> Dict[str, Any]:
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=None,
            prompt_text=json.dumps({"plan_slice": self._plan_slice(plan)}),
            agent_name="draft_generator",
            meta={"step": step, "decision": decision.signal.name, "vpm_row": result["vpm_row"]}
        )
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            scorable_id=str(uuid.uuid4()),
            text=draft_text,
            role="draft",
            meta={"step": step, "vpm_row": result["vpm_row"], "edit_log": result.get("edit_log", [])}
        )
        # goal score snapshot (so SIS can render per-step)
        goal_score = self.goals.score("text", self.goal_template, result["vpm_row"])
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            scorable_id=str(uuid.uuid4()),
            text=json.dumps(goal_score),
            role="goal_score",
            meta={"step": step}
        )
        return {
            "step": step,
            "decision": decision.signal.name,
            "vpm_row": result["vpm_row"],
            "goal_score": goal_score,
            "case_id": case.id,
            "draft_len": len(draft_text or ""),
        }

    def _select_champion(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trajectory:
            return {"step": 0, "draft_text": "", "vpm_row": {}, "goal_score": {"score": 0.0}, "case_id": None}
        best = max(trajectory, key=lambda t: float(t["goal_score"]["score"]))
        # recover draft for case
        drafts = self.memory.casebooks.list_scorables(best["case_id"], role="draft")  # type: ignore
        draft_text = drafts[-1].text if drafts else ""
        return {
            "step": best["step"],
            "vpm_row": best["vpm_row"],
            "goal_score": best["goal_score"],
            "draft_text": draft_text,
            "case_id": best["case_id"],
        }

    def _log_reflection(self, casebook: CaseBookORM, section_name: str, trajectory: List[Dict[str, Any]], champion: Dict[str, Any]):
        # Simple trend vectors (SIS can plot these)
        def series(dim: str) -> List[float]:
            return [float(t["vpm_row"].get(dim, 0.0)) for t in trajectory]

        reflection = {
            "section_name": section_name,
            "steps": len(trajectory),
            "champion_step": champion.get("step"),
            "goal_score_champion": champion.get("goal_score", {}).get("score"),
            "coverage_trend": series("coverage"),
            "citation_trend": series("citation_support"),
            "coherence_trend": series("coherence"),
            "entity_consistency_trend": series("entity_consistency"),
            "novelty_trend": series("novelty"),
            "stickiness_trend": series("stickiness"),
        }
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=None,
            prompt_text=f"Reflection for trajectory: {section_name}",
            agent_name="draft_generator",
            meta={"type": "reflection"}
        )
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            scorable_id=str(uuid.uuid4()),
            text=json.dumps(reflection),
            role="reflection",
            meta={"section": section_name}
        )

    def _plan_slice(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "section_title": plan.get("section_title"),
            "units": [{"id": u.get("claim_id"), "claim": u.get("claim")} for u in plan.get("units", [])],
            "abbr": plan.get("entities", {}).get("ABBR", {}),
            "domains": plan.get("domains", [])[:5],
            "knowledge_hash": plan.get("meta", {}).get("knowledge_hash"),
        }
