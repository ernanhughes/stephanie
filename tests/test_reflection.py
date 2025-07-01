import pytest
from unittest.mock import AsyncMock
from stephanie.agents.reflection import ReflectionAgent

@pytest.mark.asyncio
async def test_reflection_agent_returns_reviews():
    agent = ReflectionAgent(cfg={}, memory=None, logger=None)
    agent.call_llm = AsyncMock(return_value="This is a review.")

    result = await agent.run({"hypotheses": ["Hypothesis 1"]})

    assert "reviewed" in result
    assert isinstance(result["reviewed"], list)
    assert len(result["reviewed"]) == 1