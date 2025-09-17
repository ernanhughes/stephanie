# stephanie/agents/master_pupil/pupil.py
from stephanie.agents.base_agent import BaseAgent


class PupilAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        question = context.get(
            self.input_key, context.get("goal", {}).get("goal_text", "")
        )
        answer = self.answer(question, context)
        context.setdefault(self.output_key, []).append(answer)
        self.logger.log("PupilAnswerGenerated", f"Answered: {answer[:50]}...")
        return context

    def answer(self, question, context):
        return self.call_llm(question, context, self.cfg.get("pupil_model"))
