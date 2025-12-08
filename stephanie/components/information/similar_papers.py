import re
from typing import Any, List

from stephanie.components.information.data import SimilarPaperProvider, SimilarPaperRecord

# from huggingface_hub import InferenceClient  # or gradio client you're using


class HuggingFaceSimilarPaperProvider(SimilarPaperProvider):
    def __init__(self, client: Any) -> None:
        self.client = client  # your HF / Gradio client

    def get_similar_for_arxiv(self, arxiv_id: str) -> List[SimilarPaperRecord]:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        raw = self._call_tool(pdf_url)
        paper_ids = re.findall(r"https://huggingface\.co/papers/(\d+\.\d+)", raw)

        results: List[SimilarPaperRecord] = []
        for pid in paper_ids:
            results.append(
                SimilarPaperRecord(
                    arxiv_id=pid,
                    url=f"https://arxiv.org/pdf/{pid}.pdf",
                    title=pid,
                    summary=None,
                    source="hf_similar",
                    raw=None,
                )
            )
        return results

    def _call_tool(self, pdf_url: str) -> str:
        """
        Wrap your real recommend_similar_papers logic here.
        Returns the raw string response from the HF tool.
        """
        # Example stub, plug in your existing `recommend_similar_papers`:
        # res = self.client.predict(pdf_url, None, False, api_name="/predict")
        # return res
        return ""
