# stephanie/agents/paper_summarizer.py
import re
from typing import Any, Dict, Optional

from dspy import Example, Predict

from stephanie.agents.base_agent import BaseAgent
from stephanie.utils.prompt_loader import PromptLoader


class PaperSummarizer(BaseAgent):
    """
    Summarizes scientific papers into structured knowledge:
    - Problem statement
    - Method overview
    - Key results
    - Hypothesis candidates
    """

    def __init__(
        self,
        cfg: Dict,
        memory: Optional[Any] = None,
        logger=None,
        llm_model_name: str = "qwen3",
    ):
        super().__init__(cfg, memory, logger)
        self.prompt_loader = PromptLoader(cfg)
        self.summarize_prompt_key = cfg.get("summarize_prompt_key", "paper_summarize")
        self.llm = Predict(llm_model_name)

    def summarize(self, paper_content: Dict) -> Dict:
        """
        Takes parsed paper_score content (from DocumentParser) and returns a structured summary.
        """
        full_text = paper_content.get("full_text", "")
        if not full_text:
            self.logger.log("SummarizeError", {"error": "No full_text provided for summarization."})
            return {}

        merged_context = {
            "full_text": full_text,
            "abstract": paper_content["metadata"].get("abstract", ""),
            "title": paper_content["metadata"].get("title", ""),
        }

        prompt = self.prompt_loader.load_prompt(self.cfg, merged_context)
        response = self.call_llm(prompt, merged_context)

        try:
            summary = self._parse_summary(response)
        except Exception as e:
            summary = {"raw_response": response}

        # Save to memory if available
        if self.memory:
            self.memory.summaries.insert(summary)

        return summary

    def _parse_summary(self, response: str) -> Dict:
        """
        Parses the LLM's output into a structured dictionary.
        Assumes JSON-like format or key-based sections.
        """
        result = {}

        result["problem"] = self._extract_section(response, "Problem")
        result["method"] = self._extract_section(response, "Method")
        result["results"] = self._extract_section(response, "Results")
        result["hypotheses"] = self._extract_hypotheses(response)

        return result

    def _extract_section(self, text: str, section_name: str) -> str:
        """
        Helper to extract a named section from LLM output using regex.
        """
        pattern = f"{section_name}:(.*?)(?=(\\n\\w+:)|$)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_hypotheses(self, text: str) -> list[str]:
        """
        Extracts hypothesis suggestions from the LLM output.
        """
        hypo_start = text.find("Hypotheses:")
        if hypo_start == -1:
            return []

        hypo_end = text.find("\n\n", hypo_start)
        hypo_text = text[hypo_start:hypo_end].strip()
        hypotheses = re.split(r"\d+\.\s+", hypo_text)[1:]
        return [h.strip() for h in hypotheses if h.strip()]