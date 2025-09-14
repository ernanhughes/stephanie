# stephanie/cbr/goal_state_tracker.py
class DefaultGoalStateTracker:
    def __init__(self, cfg, memory, container, logger):
        self.cfg, self.memory, self.logger = cfg, memory, container, logger
        self._mem = {}

    def bump_run_ix(self, casebook_id: int, goal_id: str) -> int:
        try:
            state = self.memory.casebooks.get_goal_state(casebook_id, goal_id)
            if state is None:
                self.memory.casebooks.upsert_goal_state(casebook_id, goal_id, case_id=None, quality=0.0)
                state = self.memory.casebooks.get_goal_state(casebook_id, goal_id)
            ix = getattr(state, "run_ix", 0) + 1
            setattr(state, "run_ix", ix)
            self.memory.casebooks.session.commit()  # type: ignore
            return ix
        except Exception:
            key = f"{casebook_id}:{goal_id}"
            self._mem[key] = self._mem.get(key, 0) + 1
            return self._mem[key]

    def should_run_ab(self, run_ix: int, mode: str, period: int) -> bool:
        if mode == "off": return False
        if mode == "always": return True
        return run_ix % max(1, period) == 0
