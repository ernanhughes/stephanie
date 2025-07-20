
from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable_factory import TargetType
from stephanie.scoring.scorable_factory import ScorableFactory

class HNetAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        results = []

        for doc in documents:
            doc_id = doc["id"]
            scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
            embedding = self.memory.hnet_embeddings.get_or_create(scorable.text)
            print(f"Embedding for hnet document {doc_id} created: {embedding[:10]}...")
            embedding = self.memory.hf_embeddings.get_or_create(scorable.text)
            print(f"Embedding for hf document {doc_id} created: {embedding[:10]}...")

            results.append(scorable.to_dict())

        context[self.output_key] = results
        return context

