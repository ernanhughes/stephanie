import pytest
from stephanie.agents.ranking import RankingAgent

@pytest.mark.asyncio
async def test_ranking_agent_returns_sorted():
    agent = RankingAgent(cfg={}, memory=None, logger=None)

    input_data = {
        "hypotheses": [
            {"hypotheses": "Hypo A", "review": "A review", "persona": "Optimist"},
            {"hypotheses": "Hypo B", "review": "Another review", "persona": "Skeptic"},
        ]
    }
    result = await agent.run(input_data)

    assert "ranked" in result
    assert isinstance(result["ranked"], list)
    assert len(result["ranked"]) >= 1