from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable

log = logging.getLogger(__name__)

class VisualIntrospectionAgent(BaseAgent):
    """
    GSM8K → Qwen3 → Scorables + JSONL log.

    - Loads a subset of GSM8K from Hugging Face
    - Shuffles and selects N examples per run
    - Builds a prompt from a Jinja template (via BaseAgent.prompt_loader)
    - Calls the configured LLM (intended: Qwen3) via self.call_llm
    - Parses structured tags (<reasoning>, <final_answer>, <confidence>, <self_eval>, <tags>)
    - Builds Scorable objects with correctness + metadata
    - Splits into:
        * context["scorables_targeted"] = correct solutions
        * context["scorables_baseline"] = incorrect solutions
        * context["scorables"] = union of both (for VisiCalcAgent input)
    - Writes a JSONL log file for this run:
        * one record per example with question, gold answer, raw + parsed output, scorable fields

    Configure in Hydra, e.g.:

      visual_introspection:
        _target_: stephanie.agents.visual_introspection.VisualIntrospectionAgent
        name: visual_introspection
        enabled: true
        input_key: "scorables"          # VisiCalcAgent expects this later
        output_key: "visual_introspection"
        strategy: "gsm8k_solve"
        num_examples: 32
        split: "train"
        store_raw: true
        out_dir: "runs/visual_introspection"
        shuffle: true
        # model_name: "qwen3-xxx"       # optional; BaseAgent may also set this

    Then VisiCalcAgent runs afterwards and consumes:
      - context["scorables"]
      - context["scorables_targeted"]
      - context["scorables_baseline"]
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Strategy label for logging / telemetry
        self.strategy: str = cfg.get("strategy", "gsm8k_solve")

        # Pattern to optionally extract a region (not critical here, but kept for compatibility)
        self.extraction_regex: str = cfg.get(
            "extraction_regex",
            r"response:(.*)",  # default compatible with your GenericAgent
        )

        # Dataset knobs
        self.num_examples: int = int(cfg.get("num_examples", 400))
        self.split: str = cfg.get("split", "train")
        self.store_raw: bool = bool(cfg.get("store_raw", True))
        self.shuffle: bool = bool(cfg.get("shuffle", True))

        # Model label (for logs only; actual routing is handled by BaseAgent/LLM config)
        self.model_name: str = cfg.get("model_name", "qwen3")

        # Output logging
        out_root = Path(cfg.get("out_dir", "runs/visual_introspection"))
        # Each run_id is unique, so we naturally get a per-run directory
        self.out_dir: Path = out_root / self.run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # JSONL file for this run (one record per GSM8K example)
        self.jsonl_path: Path = Path(
            cfg.get("jsonl_file", self.out_dir / "gsm8k_qwen3_samples.jsonl")
        )

    # ------------------------------------------------------------------ #
    # Helper: regex-based extraction (kept for compatibility if you want)
    # ------------------------------------------------------------------ #
    def _extract_region(self, text: str) -> str:
        """
        Apply the configured regex extraction to the raw LLM text.
        Falls back to the full text if no match is found.

        NOTE: For GSM8K we primarily rely on tag parsing (parse_qwen_gsm8k_output),
        so this is optional / secondary.
        """
        if not self.extraction_regex:
            return text.strip()

        m = re.search(self.extraction_regex, text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # -----------------------------------------------------------------
            # 1) Load dataset and pick a random subset
            # -----------------------------------------------------------------
            ds = load_dataset("gsm8k", "main", split=self.split)
            num_total = len(ds)
            if num_total == 0:
                raise ValueError(f"GSM8K split '{self.split}' is empty")

            # Shuffle to avoid re-using the same examples every run.
            # Use run_id to derive a unique but deterministic seed for this run.
            if self.shuffle:
                # hash(run_id) is stable per run; mask to 32 bits for HF
                seed = int(hash(self.run_id) & 0xFFFFFFFF)
                ds = ds.shuffle(seed=seed)

            n = min(self.num_examples, num_total)
            ds = ds.select(range(n))

            # -----------------------------------------------------------------
            # 2) Iterate over examples, call LLM, parse, build scorables
            # -----------------------------------------------------------------
            all_scorables: List[Scorable] = []
            good_scorables: List[Scorable] = []
            bad_scorables: List[Scorable] = []
            jsonl_records: List[Dict[str, Any]] = []

            for i, row in enumerate(ds):
                problem_id = row.get("id") or f"gsm8k-{self.split}-{i}"
                question: str = row["question"]
                gold_answer: str = row.get("answer", "")

                # Context that feeds the Jinja prompt
                merged_context = {
                    "goal": {
                        "goal_text": "Solve the math word problem and give a single final answer.",
                    },
                    "preferences": [
                        "Show clear, step-by-step reasoning.",
                        "Use simple language, suitable for a student.",
                        "Keep the final answer short and explicit.",
                    ],
                    "instructions": [
                        "Follow the required output schema exactly.",
                        "Do not add extra commentary outside the tags.",
                    ],
                    "example": {
                        "id": problem_id,
                        "source": "gsm8k",
                        "question": question,
                        "context": "",
                        # You can add "gold_answer": gold_answer here if you want it in the prompt
                    },
                }

                # Build prompt from template + merged context (NOT pipeline context)
                prompt: str = self.prompt_loader.load_prompt(self.cfg, merged_context)

                # Call Qwen3 (or whatever model is wired for this agent)
                raw_text: str = self.call_llm(prompt, context)

                # Parse tagged output
                parsed = parse_qwen_gsm8k_output(raw_text)

                # Build a Scorable with correctness and metadata
                scorable = build_scorable_for_example(
                    problem_id=problem_id,
                    question=question,
                    gold_answer=gold_answer,
                    raw_text=raw_text,
                    source_label=self.model_name,
                )

                all_scorables.append(scorable)
                if scorable.meta.get("is_correct"):
                    log.info(f"✅ Correct: {problem_id}")
                    good_scorables.append(scorable)
                else:
                    log.info(f"❌ Incorrect: {problem_id}")
                    bad_scorables.append(scorable)

                # Prepare JSONL record for this example
                rec: Dict[str, Any] = {
                    "task": "gsm8k",
                    "split": self.split,
                    "problem_id": problem_id,
                    "model_name": self.model_name,
                    "strategy": self.strategy,
                    "question": question,
                    "gold_answer": gold_answer,
                    "prompt": prompt,
                    "raw_response": raw_text if self.store_raw else None,
                    "parsed": parsed,
                    "scorable": {
                        "text": scorable.text,
                        "external_id": scorable.id,
                        "meta": scorable.meta,
                    },
                }
                jsonl_records.append(rec)

            # -----------------------------------------------------------------
            # 3) Write JSONL log file for this run
            # -----------------------------------------------------------------
            print(f"💾 Writing GSM8K {self.model_name} samples to {self.jsonl_path}")
            with self.jsonl_path.open("w", encoding="utf-8") as f:
                for rec in jsonl_records:
                    json.dump(rec, f, ensure_ascii=False)
                    f.write("\n")

            # -----------------------------------------------------------------
            # 4) Populate context for downstream agents (VisiCalcAgent)
            # -----------------------------------------------------------------
            # These scorables will be processed by ScorableProcessor, then VisiCalc.
            context["scorables"] = all_scorables
            context["scorables_targeted"] = good_scorables
            context["scorables_baseline"] = bad_scorables

            # Also fill the agent's own output_key with a summary
            summary = {
                "title": getattr(self, "name", "visual_introspection"),
                "strategy": self.strategy,
                "model_name": self.model_name,
                "num_examples": len(all_scorables),
                "num_correct": len(good_scorables),
                "num_incorrect": len(bad_scorables),
                "jsonl_path": str(self.jsonl_path),
                # A tiny peek at the first parsed record for debugging
                "first_example": jsonl_records[0] if jsonl_records else None,
            }
            context[self.output_key] = summary

            # -----------------------------------------------------------------
            # 5) Log success
            # -----------------------------------------------------------------
            first_answer = (
                jsonl_records[0]["parsed"]["final_answer"]
                if jsonl_records and jsonl_records[0].get("parsed")
                else ""
            )
            self.logger.log(
                "AgentRanSuccessfully",
                {
                    "agent": getattr(self, "name", "visual_introspection"),
                    "input_key": self.input_key,
                    "output_key": self.output_key,
                    "prompt_snippet": jsonl_records[0]["prompt"][:200]
                    if jsonl_records
                    else "",
                    "response_snippet": first_answer[:300],
                    "num_examples": len(all_scorables),
                    "num_correct": len(good_scorables),
                    "num_incorrect": len(bad_scorables),
                    "model_name": self.model_name,
                    "jsonl_path": str(self.jsonl_path),
                },
            )

            return context

        except Exception as e:
            # Make sure failures are visible but don't crash the pipeline
            err_msg = f"{type(e).__name__}: {e}"
            print(f"❌ VisualIntrospectionAgent exception: {err_msg}")
            self.logger.log(
                "AgentFailed",
                {
                    "agent": getattr(self, "name", "visual_introspection"),
                    "error": err_msg,
                    "input_key": self.input_key,
                    "output_key": self.output_key,
                    "context_snapshot": {k: len(str(v)) for k, v in context.items()},
                },
            )
            return context


# ---------------------------------------------------------------------- #
# Parsing / Scorable helpers (unchanged from your version, just grouped)
# ---------------------------------------------------------------------- #

def _extract_tag(text: str, tag: str) -> str:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_qwen_gsm8k_output(raw_text: str) -> Dict[str, Any]:
    reasoning = _extract_tag(raw_text, "reasoning")
    final_answer = _extract_tag(raw_text, "final_answer")
    confidence = _extract_tag(raw_text, "confidence")
    self_eval = _extract_tag(raw_text, "self_eval")
    tags_raw = _extract_tag(raw_text, "tags")

    tags: List[str] = []
    if tags_raw:
        parts = re.split(r"[,;\n]", tags_raw)
        tags = [p.strip() for p in parts if p.strip()]

    try:
        confidence_f = float(confidence) if confidence else None
    except ValueError:
        confidence_f = None

    return {
        "reasoning": reasoning,
        "final_answer": final_answer,
        "confidence": confidence_f,
        "self_eval": self_eval,
        "tags": tags,
        "raw": raw_text,
    }


def extract_number(s: str) -> Optional[str]:
    """
    Extract the last integer or decimal number from a string.

    Returns:
        The last numeric substring (e.g. '72', '-3.5') if found,
        otherwise None.
    """
    # Use a non-capturing group so we get the full match
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else None


def canonicalize_gsm8k_gold(answer: str) -> str:
    """
    GSM8K usually ends with '#### 72'.
    We'll extract the last number after '####' if present,
    otherwise fall back to the last number in the string.
    If we can't find any number at all, we return the stripped tail.
    """
    if "####" in answer:
        tail = answer.split("####")[-1]
    else:
        tail = answer

    num = extract_number(tail)
    return num if num is not None else tail.strip()


def extract_pred_answer(text: str) -> Optional[str]:
    """
    Very forgiving numeric extractor for model predictions.

    Strategy:
      1. Look for a line starting with 'Final answer:' and grab the number there.
      2. If not found, fall back to the last number anywhere in the text
         using extract_number.
      3. If still nothing, return None.
    """
    # 1) Try 'Final answer:' line
    m = re.search(r"Final answer\s*[:\-]\s*(.+)", text, re.IGNORECASE)
    if m:
        line = m.group(1).strip()
        num = extract_number(line)
        if num is not None:
            return num

    # 2) Fallback: last number in the whole text
    num = extract_number(text)
    if num is not None:
        return num

    # 3) Nothing found
    return None


def build_scorable_for_example(
    problem_id: str,
    question: str,
    gold_answer: str,
    raw_text: str,
    *,
    source_label: str = "qwen3_gsm8k",
) -> Scorable:
    """
    Build a Scorable that captures:
      - the question
      - the model's full reasoning + answer (raw_text)
      - simple correctness flag based on numeric match
    """
    gold_canon = canonicalize_gsm8k_gold(gold_answer)
    pred_canon = extract_pred_answer(raw_text)

    is_correct = (pred_canon is not None and gold_canon == pred_canon)

    text = (
        f"Question:\n{question}\n\n"
        f"Model response:\n{raw_text}\n"
    )

    meta: Dict[str, Any] = {
        "task": "gsm8k",
        "problem_id": problem_id,
        "source": source_label,
        "gold_answer_raw": gold_answer,
        "gold_answer_canonical": gold_canon,
        "pred_answer_raw": raw_text,
        "pred_answer_canonical": pred_canon,
        "is_correct": is_correct,
    }

    return Scorable(
        text=text,
        id=problem_id,
        meta=meta,
    )


def canonicalize_pred_answer(final_answer: str) -> str:
    # For now, just strip; you can add number parsing later
    return final_answer.strip()

