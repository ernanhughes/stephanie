# stephanie/cbr/case_selector.py
import random
from typing import List


class DefaultCaseSelector:
    def __init__(self, cfg, memory, container, logger):
        self.cfg, self.memory, self.logger = cfg, memory, container, logger

    @staticmethod 
    def _top_scorables_from_case(case, k=3) -> List[str]:
        out = []
        try:
            outs = [cs for cs in case.scorables if (getattr(cs, "role", "") or "").lower() == "output"]
            outs.sort(key=lambda cs: getattr(cs, "rank", 1_000_000))
            for cs in outs[:k]:
                sid = getattr(cs, "scorable_id", None)
                if sid: 
                    out.append(sid)
        except Exception:
            pass
        return out

    def build_reuse_candidates(self, casebook_id: int, goal_id: str, cases, budget: int, novelty_k: int, exploration_eps: float) -> List[str]:
        ids: List[str] = []

        # Champion-first
        try:
            state = self.memory.casebooks.get_goal_state(casebook_id, goal_id)
            if state and state.champion_case_id:
                champion = next((c for c in cases if c.id == state.champion_case_id), None)
                if champion:
                    ids.extend(self._top_scorables_from_case(champion))
        except AttributeError:
            pass

        # Recent-success
        try:
            recent = self.memory.casebooks.get_recent_cases(casebook_id, goal_id, limit=max(1, budget // 2), only_accepted=True)
            for c in recent: ids.extend(self._top_scorables_from_case(c))
        except AttributeError:
            for c in cases[: max(1, budget // 2)]: ids.extend(self._top_scorables_from_case(c))

        # Diverse-novel
        pool_ids = set(ids)
        novel_pool = []
        try:
            pool = self.memory.casebooks.get_pool_for_goal(casebook_id, goal_id, exclude_ids=[getattr(c, "id", None) for c in cases], limit=200)
            novel_pool = pool or []
        except AttributeError:
            novel_pool = [c for c in cases if c.id not in pool_ids]
        random.shuffle(novel_pool)
        for c in novel_pool[:novelty_k]:
            ids.extend(self._top_scorables_from_case(c))

        # Exploration
        if random.random() < float(exploration_eps):
            extra = (cases[novelty_k : novelty_k + 2]) if len(cases) > novelty_k else []
            for c in extra: ids.extend(self._top_scorables_from_case(c))

        # Dedup + cap
        seen, out = set(), []
        for x in ids:
            if x and x not in seen:
                out.append(x); seen.add(x)
            if len(out) >= budget: break
        return out
