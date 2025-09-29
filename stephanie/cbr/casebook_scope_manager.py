# stephanie/cbr/casebook_scope_manager.py
from typing import Any, Dict, List, Optional

from stephanie.constants import GOAL, PIPELINE_RUN_ID


class DefaultCasebookScopeManager:
    def __init__(self, cfg, memory, container, logger):
        self.cfg, self.memory, self.logger = cfg, memory, container, logger
        self.tag = cfg.get("casebook_tag", "default")
        self.retrieval_mode = cfg.get("retrieval_mode", "fallback")

    def ensure_scope(self, pipeline_run_id: Optional[str], agent: Optional[str], tag: str):
        try:
            return self.memory.casebooks.ensure_casebook_scope(pipeline_run_id, agent, tag)
        except AttributeError:
            name = f"cb:{agent or 'all'}:{pipeline_run_id or 'all'}:{tag}"
            return self.memory.casebooks.ensure_casebook(name, description="Scoped fallback", tags=[tag])

    def home_casebook_id(self, ctx: Dict, agent_name: str, tag: str) -> int:
        cb = self.ensure_scope(ctx[PIPELINE_RUN_ID], agent_name, tag)
        return cb.id

    def get_cases(self, ctx: Dict, retrieval_mode: str, tag: str) -> List[Any]:
        goal_id = ctx[GOAL]["id"]
        pipeline_run_id = ctx[PIPELINE_RUN_ID]
        agent = ctx.get("agent_name") or "UnknownAgent"

        scopes = [(pipeline_run_id, agent, tag)]
        if retrieval_mode in ("fallback", "union"):
            scopes += [(None, agent, tag), (pipeline_run_id, None, tag), (None, None, tag)]

        try:
            if retrieval_mode == "strict":
                cb = self.ensure_scope(pipeline_run_id, agent, tag)
                return self.memory.casebooks.get_cases_for_goal_in_casebook(cb.id, goal_id)
            elif retrieval_mode == "fallback":
                for sc in scopes:
                    cases = self.memory.casebooks.get_cases_for_goal_scoped(goal_id, [sc])
                    if cases: return cases
                return []
            else:
                return self.memory.casebooks.get_cases_for_goal_scoped(goal_id, scopes)
        except AttributeError:
            return self.memory.casebooks.get_cases_for_goal(goal_id)
