import torch
from sklearn.metrics.pairwise import cosine_similarity

from co_ai.agents.base_agent import BaseAgent
from co_ai.agents.master_pupil.master import MasterAgent
from co_ai.agents.master_pupil.pupil import PupilAgent
from co_ai.agents.master_pupil.trainer import TrainerAgent


class EvaluatorAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.master = MasterAgent(cfg, memory, logger)
        self.pupil = PupilAgent(cfg, memory, logger)
        self.trainer = TrainerAgent(
            cfg, memory, logger, master=self.master, pupil=self.pupil
        )

    async def run(self, context: dict) -> dict:
        question = context.get("goal", {}).get("goal_text", "")
        master_answer = context.get("master_answer")[0]
        pupil_answer = context.get("pupil_answer")[0]

        score_before = self.score_alignment(master_answer, pupil_answer)
        self.logger.log(
            "EvaluatorRun",
            {
                "score_before": score_before,
                "question": question,
                "master_answer": master_answer,
                "pupil_answer": pupil_answer,
            },
        )
        aligned_answer = self.trainer.align_response(question, context=context)
        score_after = self.score_alignment(master_answer, aligned_answer)
        self.logger.log(
            "EvaluationResult",
            {
                "Before": score_before,
                "After": score_after,
                "Improvement": score_after - score_before,
            },
        )
        context.setdefault("evaluation_results", []).append(
            {
                "score_before": score_before,
                "score_after": score_after,
                "improvement": score_after - score_before,
            }
        )

        return context

    def score_alignment(self, text1, text2):
        emb1 = self.memory.embedding.get_or_create(text1)
        emb2 = self.memory.embedding.get_or_create(text2)
        sim = cosine_similarity([emb1], [emb2])[0][0]
        return sim

    def evaluate_alignment(
        self, master_output: torch.Tensor, pupil_output: torch.Tensor
    ):
        similarity = (
            cosine_similarity(master_output, pupil_output, dim=-1).mean().item()
        )
        distance = torch.norm(master_output - pupil_output, dim=-1).mean().item()
        return {
            "cosine_similarity": round(similarity, 4),
            "vector_distance": round(distance, 4),
        }
