from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .paper_graph_abi import PaperGraphABI


class PaperGraphDumper:
    def __init__(self, *, run_dir: str) -> None:
        self.run_dir = Path(run_dir)

    def dump(
        self,
        *,
        arxiv_id: str,
        graph: PaperGraphABI,
        filename: str = "paper_graph.json",
    ) -> str:
        out_dir = self.run_dir / arxiv_id
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / filename
        out_path.write_text(json.dumps(graph.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return str(out_path)
