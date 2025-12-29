# DraftGeneratorAgent — knowledge-first, VPM-powered draft trajectories from paper + chat
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.knowledge.improver import Improver
from stephanie.agents.knowledge.knowledge_fuser import KnowledgeFuser
from stephanie.agents.paper_improver.goals import GoalScorer
from stephanie.orm.casebook import CaseBookORM
from stephanie.scoring.scorable import ScorableFactory, ScorableType
from stephanie.utils.json_sanitize import safe_json
from stephanie.zeromodel.vpm_controller import (Signal, VPMController, VPMRow,
                                                default_controller)


class DraftGeneratorAgent(BaseAgent):
    """
    Generates a blog section as an *improvement trajectory*:
      1) Fuse paper text + chat history into a content plan (transient NER + 20 domains)
      2) Iterate drafts via TextImprover → VPM rows
      3) Use VPMController to decide EDIT/RESAMPLE/STOP
      4) Store every step in a CaseBook (for SIS + PACS/CBR)
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.max_steps = cfg.get("max_steps", 5)
        self.goal_template = cfg.get("goal_template", "academic_summary")
        self.section_name_fallback = cfg.get("section_name_fallback", "Blog Section")
        self.vpm: VPMController = default_controller()
        self.goals = GoalScorer()
        self.ti = Improver(cfg, memory=memory, logger=logger, workdir=cfg.get("text_workdir", "./text_runs"))
        self.fuser = KnowledgeFuser(cfg, memory, container, logger)

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
        documents = context.get("documents", [])

        paper = documents[0]
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
        plan = await self.fuser.fuse(
            text=paper_text,
            chat_messages=chat_messages,
            section_name=section_name,
            context=context
        )
        # Add explicit goal_template so TextImprover and downstream scorers can use it
        plan["goal_template"] = self.goal_template
        kg_ctx = self.container.get("knowledge_graph").build_context_for_plan(plan, k=5)
        plan["kg"] = kg_ctx  # pass into TextImprover

        # --------- CaseBook to log the trajectory ----------
        casebook = self._ensure_casebook(paper, section_name, plan, context)
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
            traj_rec = self._log_step(casebook, plan, result, step, decision, draft_text, context=context)
            trajectory.append(traj_rec)

            # STOP / ESCALATE gates
            if decision.signal in (Signal.STOP, Signal.ESCALATE):
                break

        # --------- Champion selection (best goal score) ----------
        champion = self._select_champion(trajectory)

        draft_scorable = self.memory.dynamic_scorables.add(
            pipeline_run_id=context.get("pipeline_run_id"),
            scorable_type=ScorableType.DYNAMIC,
            source=self.name,
            text=draft_text,
            meta={
                "step": step,
                "decision": decision.signal.name,
                "vpm_row": result["vpm_row"],
                "edit_log": result.get("edit_log", []),
            },
        )

        scorable = ScorableFactory.from_orm(draft_scorable)
        goal = context.get("goal", {})
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=goal.get("id"),
            prompt_text=json.dumps({"plan_slice": self._plan_slice(plan)}),
            agent_name="draft_generator",
            scorables=[scorable.to_dict()],
            meta={
                "step": step,
                "decision": decision.signal.name,
                "vpm_row": result["vpm_row"],
            },
        )

        # --------- Reflection ----------
        self._log_reflection(casebook, section_name, trajectory, champion, context=context)

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
        Return recent chat messages as list of dicts {role, text, ts, id, conversation_id}.
        Priority:
        1) context["chat_messages"] if provided
        2) messages from top conversations in memory
        """
        msgs = context.get(self.chat_from_context_key)
        if isinstance(msgs, list) and msgs:
            return msgs[-self.chat_max_messages:]

        # Otherwise, pull from top conversations
        conversations = self.memory.chats.get_top_conversations(limit=3, by="messages")
        all_msgs = []
        for conv, _ in conversations:
            conv_msgs = self.memory.chats.get_messages(conv.id)
            for m in conv_msgs:
                all_msgs.append({
                    "id": m.id,
                    "conversation_id": m.conversation_id,
                    "role": m.role,
                    "text": m.text,
                    "ts": getattr(m, "created_at", None)
                })

        # Return most recent N across conversations
        all_msgs = sorted(all_msgs, key=lambda m: m.get("ts") or 0)
        return all_msgs[-self.chat_max_messages:]

    def _ensure_casebook(self, paper: dict, section_name: str, plan: dict, context: dict) -> CaseBookORM:
        casebook_name = f"blog_{paper.get('id','unknown')}_{section_name}_{int(time.time())}"
        meta = {
            "paper_id": paper.get("id"),
            "paper_title": paper.get("title"),
            "section_name": section_name,
            "domains": plan.get("domains", []),
            "knowledge_hash": plan.get("meta", {}).get("knowledge_hash"),
            "transient": True,   # ⚠️ transient domains/NER; not persisted elsewhere
        }
        pipeline_run_id = context.get("pipeline_run_id")
        return self.memory.casebooks.ensure_casebook(
            name=casebook_name,
            pipeline_run_id=pipeline_run_id,
            description=f"Draft trajectory for '{section_name}' from fused knowledge",
            tags=["draft_generator"],
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

    def _log_step(
        self,
        casebook: CaseBookORM,
        plan: dict,
        result: dict,
        step: int,
        decision,
        draft_text: str,
        context: dict,
    ) -> Dict[str, Any]:
        run_id = str(uuid.uuid4())
        pipeline_run_id = context.get("pipeline_run_id")
        goal = context.get("goal", {})

        # --- 1) Create dynamic scorables first ---
        draft_scorable = self.memory.dynamic_scorables.add(
            pipeline_run_id=pipeline_run_id,
            scorable_type=ScorableType.DYNAMIC,
            source=self.name,
            text=draft_text,
            meta={
                "step": step,
                "decision": decision.signal.name,
                "vpm_row": result["vpm_row"],
                "edit_log": result.get("edit_log", []),
            },
        )
        scorable = ScorableFactory.from_orm(draft_scorable)

        # --- 2) Create case linking to scorables ---
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=goal.get("id"),
            prompt_text=json.dumps({"plan_slice": self._plan_slice(plan)}),
            agent_name="draft_generator",
            scorables=[scorable.to_dict()],
            meta={
                "step": step,
                "decision": decision.signal.name,
                "vpm_row": result["vpm_row"],
            },
        )

        # --- 3) Goal scoring with fallback ---
        kind = "text"
        goal_text = goal.get("goal_text", "blog_general")
        try:
            goal_score = self.goals.score(kind, goal_text, result["vpm_row"])
        except KeyError:
            # Dynamically register a new GoalTemplate if missing
            from stephanie.agents.paper_improver.goals import GoalTemplate

            self.logger.log("GoalTemplateMissing", {
                "kind": kind,
                "goal": goal_text,
                "message": "Creating dynamic fallback template"
            })

            self.goals.templates[f"{kind}/{goal_text}"] = GoalTemplate(
                name=goal_text,
                dims=list(result["vpm_row"].keys()),  # use all dims present
                thresholds={d: 0.5 for d in result["vpm_row"].keys()},
            )
            goal_score = self.goals.score(kind, goal_text, result["vpm_row"])

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

    def _log_reflection(self, casebook: CaseBookORM, section_name: str, trajectory: List[Dict[str, Any]], champion: Dict[str, Any], context: dict = {}) -> None:
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
        goal = context.get("goal", {})
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=goal.get("id"),
            prompt_text=f"Reflection for trajectory: {section_name}",
            agent_name="draft_generator",
            meta={"type": "reflection"}
        )
        self.memory.casebooks.add_scorable(
            case.id,
            scorable_type="reflection",
            pipeline_run_id=context.get("pipeline_run_id"),
            text=safe_json(reflection),
            meta={"section": section_name},
            role="reflection"
        )

    def _plan_slice(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "section_title": plan.get("section_title"),
            "units": [{"id": u.get("claim_id"), "claim": u.get("claim")} for u in plan.get("units", [])],
            "abbr": plan.get("entities", {}).get("ABBR", {}),
            "domains": plan.get("domains", [])[:5],
            "knowledge_hash": plan.get("meta", {}).get("knowledge_hash"),
        }
