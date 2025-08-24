# stephanie/utils/token_counter.py
import tiktoken
from transformers import AutoTokenizer


class TokenCounter:
    """
    Counts tokens for different model families.
    Supports OpenAI tiktoken models and HuggingFace/transformers models.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

        # Choose tokenizer backend
        if model_name.startswith(("gpt", "o1", "o3", "o4")):
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
                self.backend = "tiktoken"
            except Exception:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self.backend = "tiktoken"
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.backend = "hf"
            except Exception:
                raise ValueError(f"Unsupported model: {model_name}")

    def count_tokens(self, text: str) -> int:
        if self.backend == "tiktoken":
            return len(self.tokenizer.encode(text, disallowed_special=()))
        elif self.backend == "hf":
            return len(self.tokenizer.encode(text))
        else:
            raise RuntimeError("Invalid backend")
