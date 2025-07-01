import pytest
from unittest.mock import AsyncMock
from stephanie.agents.generation import GenerationAgent

@pytest.mark.asyncio
async def test_generation_agent_returns_hypotheses():
    agent = GenerationAgent(cfg={}, memory=None, logger=None)
    agent.call_llm = AsyncMock(return_value="1. Hypothesis A\n2. Hypothesis B")

    result = await agent.run({"goal": "AI and finance"})

    assert "hypotheses" in result
    assert isinstance(result["hypotheses"], list)
    assert len(result["hypotheses"]) == 2