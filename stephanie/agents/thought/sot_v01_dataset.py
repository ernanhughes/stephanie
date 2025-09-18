# stephanie/agents/thought/sot_v01_dataset.py
from __future__ import annotations

import json
from torch.utils.data import Dataset
from stephanie.agents.thought.sot_v01_multitask import MOVE2ID

class SoTV01Dataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                ex = json.loads(line)
                # skip empties
                if not ex.get("query") or not ex.get("response"): 
                    continue
                self.examples.append(ex)

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        move = (ex.get("predicted_move") or "VOICE").upper()
        move_label = MOVE2ID.get(move, MOVE2ID["VOICE"])

        prompt = self._prompt(ex["query"], ex.get("retrieved_turns", []), move)
        target = ex["response"]

        return {
            "prompt_text": prompt,
            "target_text": target,
            "move_label": move_label
        }

    def _prompt(self, query: str, retrieved_turns: list, move: str) -> str:
        # Few-shot style like your composer
        parts = [f"[PREDICTED MOVE: {move}]",
                 "[RETRIEVED CONTEXT]"]
        # cap to ~3 examples worth of lines
        lines = []
        for rt in retrieved_turns[:12]:  # 6 user/assistant pairs ~ 3 examples
            u = (rt.get("user") or "").strip()
            a = (rt.get("assistant") or "").strip()
            if u or a:
                lines.append(f"User: {u}\nYou: {a}")
        if lines:
            parts.append("\n\n".join(lines))
        parts.append("[END RETRIEVED CONTEXT]\n")
        parts.append(f"User: {query}\nYou: ")
        return "\n".join(parts)
