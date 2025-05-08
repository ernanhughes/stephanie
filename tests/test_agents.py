import pytest
from co_ai.agents import GenerationAgent, ReflectionAgent, RankingAgent, EvolutionAgent, MetaReviewAgent
from co_ai.memory import VectorMemory

@pytest.fixture
def memory():
    return VectorMemory()

def test_generation_agent_runs(memory):
    agent = GenerationAgent(memory)
    result = agent.run({"goal": "Test goal"})
    assert result is not None

def test_reflection_agent_runs(memory):
    agent = ReflectionAgent(memory)
    result = agent.run({"hypotheses": ["Test hypothesis"]})
    assert result is not None
