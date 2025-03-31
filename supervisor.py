# ai_co_scientist/supervisor.py
import asyncio
from agents.generation import GenerationAgent
from agents.reflection import ReflectionAgent
from agents.ranking import RankingAgent
from agents.evolution import EvolutionAgent
from agents.meta_review import MetaReviewAgent
from memory.vector_store import VectorMemory
from interface.cli import get_user_goal
import dspy

class Supervisor:
    def __init__(self):
        self.memory = VectorMemory()
        self.generation_agent = GenerationAgent(self.memory)
        self.reflection_agent = ReflectionAgent(self.memory)
        self.ranking_agent = RankingAgent(self.memory)
        self.evolution_agent = EvolutionAgent(self.memory)
        self.meta_review_agent = MetaReviewAgent(self.memory)
        lm = dspy.LM('ollama_chat/qwen2.5', api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=lm)


    async def run_once(self, goal, run_id="default_run", use_grafting=True):
        print("[Supervisor] Starting pipeline...")

        initial = await self.generation_agent.run({"goal": goal})
        reviewed = await self.reflection_agent.run({"hypotheses": initial["hypotheses"]})
        ranked = await self.ranking_agent.run({"reviewed": reviewed, "run_id": run_id})
        evolved = await self.evolution_agent.run({"ranked": ranked["ranked"], "use_grafting": use_grafting})
        await self.meta_review_agent.run({"evolved": evolved["evolved"]})

        print("[Supervisor] Pipeline complete.")

    def run_pipeline(self):
        goal = get_user_goal()
        asyncio.run(self.run_once(goal))

    def run_pipeline_config(self, goal, run_id="default_run", use_grafting=True):
        asyncio.run(self.run_once(goal, run_id, use_grafting))
