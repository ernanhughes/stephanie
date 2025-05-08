import pytest
from ai_co_scientist.memory import VectorMemory, Hypothesis

def test_store_and_search_hypothesis():
    memory = VectorMemory()
    hyp = Hypothesis(goal="Test", text="Test hypothesis")
    memory.store_hypothesis(hyp)
    results = memory.search_related("Test hypothesis")
    assert results is not None
