import pytest
from co_ai.agents.ranking import RankingAgent

@pytest.mark.asyncio
async def test_ranking_agent_returns_sorted():
    agent = RankingAgent(cfg={}, memory=None, logger=None)

    input_data = {
        "hypotheses": [
            {"hypothesis": "Hypo A", "review": "A review", "persona": "Optimist"},
            {"hypothesis": "Hypo B", "review": "Another review", "persona": "Skeptic"},
        ]
    }
    result = await agent.run(input_data)

    assert "ranked" in result
    assert isinstance(result["ranked"], list)
    assert len(result["ranked"]) >= 1