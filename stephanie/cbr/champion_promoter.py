# stephanie/cbr/champion_promoter.py
class DefaultChampionPromoter:
    def __init__(self, cfg, memory, logger):
        self.cfg, self.memory, self.logger = cfg, memory, logger

    def maybe_promote(self, casebook_id: int, goal_id: str, retained_case_id: int | None, quality: float) -> None:
        if not retained_case_id: return
        try:
            self.memory.casebooks.upsert_goal_state(casebook_id, goal_id, retained_case_id, float(quality))
        except AttributeError:
            pass
