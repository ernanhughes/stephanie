# stephanie/components/vibe/datasets/writing_dataset_extractor.py
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from stephanie.components.vibe.agents.writing_scorer import WritingScorerAgent
from stephanie.components.vibe.datasets.writing_example_types import \
    WritingDatasetRow

# You’ll need to adapt these bootstrap imports to your stack:
# from stephanie.runtime.bootstrap import bootstrap_container_and_memory

log = logging.getLogger(__name__)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def looks_like_writing(turn: Dict[str, Any]) -> bool:
    """
    Heuristic filter: keep only assistant messages that look like
    substantial writing (blog / research).

    You can tighten this (e.g. by domain tags from your own pipeline).
    """
    if turn.get("role") != "assistant":
        return False

    text = (turn.get("content") or "").strip()
    if len(text) < 200:
        # too short to be a serious section
        return False

    tags = turn.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]

    # very simple heuristic: if tagged as blog/research, keep
    tagged_writing = any(
        t.lower() in ("blog", "research", "paper", "stephanie_post") for t in tags
    )

    # or if the text clearly looks like markdown / sectioned writing
    has_headings = ("## " in text) or ("### " in text)
    has_bullets = ("- " in text or "* " in text)

    return tagged_writing or has_headings or has_bullets


async def score_turn(
    scorer: WritingScorerAgent,
    turn: Dict[str, Any],
) -> Optional[WritingDatasetRow]:
    text = (turn.get("content") or "").strip()
    if not text:
        return None

    ctx: Dict[str, Any] = {
        "text": text,
        "writing_meta": {
            "source": "chat_history",
            "conversation_id": turn.get("conversation_id"),
            "turn_index": turn.get("turn_index"),
            "tags": turn.get("tags"),
            "title": turn.get("title"),
        },
    }

    ctx = await scorer.run(ctx)
    score = ctx.get("writing_score") or {}

    breakdown = score.get("breakdown") or {}
    return WritingDatasetRow(
        text=text,
        clarity=float(breakdown.get("clarity", 0.0)),
        structure=float(breakdown.get("structure", 0.0)),
        technical_correctness=float(breakdown.get("technical_correctness", 0.0)),
        depth=float(breakdown.get("depth", 0.0)),
        actionability=float(breakdown.get("actionability", 0.0)),
        vibe=float(breakdown.get("vibe", 0.0)),
        overall=float(score.get("overall", 0.0)),
        conversation_id=str(turn.get("conversation_id")),
        turn_index=int(turn.get("turn_index", -1)),
        role=str(turn.get("role")),
        tags=",".join(turn.get("tags") or []),
        extra_meta={
            "title": turn.get("title"),
            "raw_tags": turn.get("tags"),
        },
    )


async def extract_dataset(
    input_path: Path,
    output_path: Path,
    max_examples: Optional[int] = None,
) -> None:
    # 1) Bootstrap your infra (adapt this to your real bootstrap)
    #
    # memory, container, logger = bootstrap_container_and_memory()
    # For now, we’ll just stub memory/container/logger as None/log.
    memory = None
    container = None
    logger = log

    scorer_cfg: Dict[str, Any] = {}
    scorer = WritingScorerAgent(scorer_cfg, memory, container, logger)

    examples: List[WritingDatasetRow] = []

    for turn in read_jsonl(input_path):
        if not looks_like_writing(turn):
            continue

        try:
            row = await score_turn(scorer, turn)
        except NotImplementedError:
            log.error(
                "RubricEvaluatorService is not wired yet. "
                "Please connect WritingScorerAgent to your rubric evaluation stack."
            )
            raise
        except Exception:
            log.exception(
                "Failed to score turn %s:%s",
                turn.get("conversation_id"),
                turn.get("turn_index"),
            )
            continue

        if row is None:
            continue

        examples.append(row)

        if max_examples is not None and len(examples) >= max_examples:
            break

    # 3) Write JSONL output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    log.info(
        "Wrote %s examples to %s",
        len(examples),
        output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract writing-quality dataset from normalized chat history."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL chat file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL dataset file.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of examples.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    import asyncio

    asyncio.run(
        extract_dataset(
            input_path=input_path,
            output_path=output_path,
            max_examples=args.max_examples,
        )
    )


if __name__ == "__main__":
    main()
