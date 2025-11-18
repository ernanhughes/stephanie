# stephanie/components/nexus/agents/nexus_improver_resume.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.memory.prompt_job_store import PromptJobStore
from stephanie.prompts.prompt_inbox import PromptInbox
from stephanie.scoring.scorable import Scorable
from stephanie.utils.progress_mixin import ProgressMixin


class NexusImproverResumeAgent(ProgressMixin, BaseAgent):
    """
    Collects completed prompt results (offloaded candidate refinements),
    builds child Scorables, evaluates via async scoring, and finalizes decisions.
    """
    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self._init_progress(container, logger)
        self.store = PromptJobStore(container.get("session_maker"), logger)
        self.scoring = container.get("scoring")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        tickets: List[Dict[str, str]] = list(context.get("pending_prompt_tickets") or [])
        if not tickets:
            context[self.output_key] = {"status": "no_tickets"}
            return context

        inbox = PromptInbox(self.store)
        job_ids = [t["job_id"] for t in tickets]
        ready = inbox.gather_ready(job_ids)  # DB poll; switch to bus listener if you want push

        # group by parent
        by_parent: Dict[str, List[Tuple[Dict[str, str], Dict]]] = {}
        for t in tickets:
            r = ready.get(t["job_id"])
            if not r:
                continue
            by_parent.setdefault(t["parent_id"], []).append((t, r))

        decisions: List[Dict[str, Any]] = []
        use_vpm_phi = bool(context.get("use_vpm_phi", False))

        for parent_id, group in by_parent.items():
            parent = self.memory.scorables.get(parent_id) if hasattr(self.memory, "scorables") else None
            if not parent:
                continue

            children: List[Scorable] = []
            for (_, r) in group:
                txt = r.get("result_text") or ""
                if not txt.strip():
                    continue
                child = Scorable(id=f"prompt:{r['job_id']}", text=txt, target_type=getattr(parent, "target_type", "document"))
                children.append(child)

            # evaluate candidates via async scoring bus (use the mixin if you integrated it)
            try:
                async_eval = await self._eval_state_many_async(children, context, use_vpm_phi=use_vpm_phi)
            except AttributeError:
                # fallback to local
                async_eval = {c.id: self._eval_state(c, context, use_vpm_phi=use_vpm_phi) for c in children}

            parent_eval = self._eval_state(parent, context, use_vpm_phi=use_vpm_phi)
            parent_overall = float(parent_eval.get("overall", 0.0))

            cand_evals: List[Tuple[Scorable, Dict[str, Any]]] = []
            for c in children:
                res = async_eval.get(c.id) or {"overall": 0.0, "dims": {}}
                cand_evals.append((c, res))

            winner, win_eval = self._select_winner(parent, parent_overall, cand_evals, margin=0.02)
            winner_overall = float((win_eval or {}).get("overall", parent_overall))
            lift = float(winner_overall - parent_overall)

            decisions.append({
                "parent_id": parent_id,
                "winner_id": getattr(winner, "id", None) if winner else None,
                "winner_overall": winner_overall,
                "lift": lift,
                "k_ready": len(children),
            })

            # link/promote + training, same helpers as your main agent
            self._safe_link_and_promote(parent, [c.id for c,_ in cand_evals], winner, lift)
            self._safe_emit_training(parent=parent, parent_overall=parent_overall, cand_evals=cand_evals, context=context)

        # remove consumed tickets
        consumed = {t["job_id"] for grp in by_parent.values() for (t, _) in grp}
        context["pending_prompt_tickets"] = [t for t in tickets if t["job_id"] not in consumed]

        context[self.output_key] = {
            "status": "ok",
            "decisions": decisions,
            "remaining_tickets": len(context["pending_prompt_tickets"]),
        }
        return context
