# co_ai/signatures/literature_signature.py
from dspy import InputField, OutputField, Signature


class LiteratureQuerySignature(Signature):
    """
    Signature for generating effective scientific literature queries.
    
    From the paper:
    > 'The co-scientist iteratively searches the web, retrieves and reads relevant research articles...'
    """
    goal = InputField(desc="Scientific research objective")
    preferences = InputField(desc="Evaluation criteria (e.g., novelty, feasibility)")

    search_query = OutputField(desc="Effective Google Scholar-style search query")