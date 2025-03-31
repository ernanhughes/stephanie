# ai_co_scientist/agents/ranking.py
from agents.base import BaseAgent
from memory.hypothesis_model import Hypothesis
from dspy import Predict, Signature, InputField, OutputField, Module
from logs.ranking_log import EloRankingLog
import random

class JudgeSignature(Signature):
    hypothesis_a = InputField()
    review_a = InputField()
    hypothesis_b = InputField()
    review_b = InputField()
    decision = OutputField()
    explanation = OutputField()

class RankingModule(Module):
    def __init__(self):
        super().__init__()
        self.judge = Predict(JudgeSignature)

    async def forward(self, a: dict, b: dict):
        return await self.judge.apredict(
            hypothesis_a=a["hypothesis"],
            review_a=a["review"],
            hypothesis_b=b["hypothesis"],
            review_b=b["review"]
        )

class RankingAgent(BaseAgent):
    def __init__(self, memory):
        super().__init__(memory)
        self.ranker = RankingModule()
        self.elo_scores = {}
        self.traces = []

    def _initialize_scores(self, reviewed):
        for item in reviewed:
            hyp = item["hypothesis"]
            self.elo_scores[hyp] = 1000

    def _update_elo(self, winner, loser, k=32):
        r_w = self.elo_scores[winner]
        r_l = self.elo_scores[loser]
        e_w = 1 / (1 + 10 ** ((r_l - r_w) / 400))
        self.elo_scores[winner] += int(k * (1 - e_w))
        self.elo_scores[loser] -= int(k * (1 - e_w))

    async def run(self, input_data: dict) -> dict:
        reviewed = input_data.get("reviewed", [])
        print(f"[RankingAgent] Running with {reviewed} hypotheses.")
        run_id = input_data.get("run_id", "default_run")
        print("Reviewed input looks like:", reviewed)
        self._initialize_scores(reviewed["reviewed"])
        pairs = list(self._generate_pairs(reviewed))

        for a, b in pairs:
            result = await self.ranker.forward(a, b)
            decision = result.decision
            explanation = result.explanation
            winner = a['hypothesis'] if 'A' in decision else b['hypothesis']
            loser = b['hypothesis'] if winner == a['hypothesis'] else a['hypothesis']
            self._update_elo(winner, loser)

            self.traces.append({
                "winner": winner,
                "loser": loser,
                "explanation": explanation
            })

        self.log_scores_to_db(run_id)
        self.log_trace_to_db(run_id)
        sorted_hypos = sorted(self.elo_scores.items(), key=lambda x: x[1], reverse=True)
        return {"ranked": sorted_hypos}

    def _generate_pairs(self, reviewed):
        return random.sample(
            [(a, b) for i, a in enumerate(reviewed) for j, b in enumerate(reviewed) if i < j],
            k=min(10, len(reviewed) * (len(reviewed) - 1) // 2)
        )

    def log_scores_to_db(self, run_id="default_run"):
        with self.memory.conn.cursor() as cur:
            for hyp, score in self.elo_scores.items():
                log_entry = EloRankingLog(hypothesis=hyp, score=score, run_id=run_id)
                cur.execute(
                    """
                    INSERT INTO elo_ranking_log (run_id, hypothesis, score, created_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (log_entry.run_id, log_entry.hypothesis, log_entry.score, log_entry.created_at)
                )

    def log_trace_to_db(self, run_id="default_run"):
        with self.memory.conn.cursor() as cur:
            for trace in self.traces:
                cur.execute(
                    """
                    INSERT INTO ranking_trace (run_id, winner, loser, explanation)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (run_id, trace["winner"], trace["loser"], trace["explanation"])
                )
