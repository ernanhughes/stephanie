# stephanie/jobs/nightly_code_analysis.py

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import sys

from stephanie.tools.code_analyzer_tool import CodeAnalyzerTool


OUT_DIR_DEFAULT = Path("./data/runs/code_analysis").resolve()


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


@dataclass
class RepoScorable:
    """
    Minimal duck-typed scorable for tool.apply().
    This avoids depending on the exact Scorable constructor.
    """
    id: str
    target_type: str = "codebase"
    meta: Dict[str, Any] = field(default_factory=dict)


def _get_container() -> Any:
    """
    Mirror overnight.py: it expects an app that exposes app.state.container.
    Adjust this import if your bootstrap module differs.
    """
    from sis.main import app  # same pattern used in overnight.py
    container = getattr(app.state, "container", None)
    if container is None:
        raise RuntimeError("No container found at app.state.container. Check your app bootstrap.")
    return container


async def run_once(
    *,
    base_path: str,
    out_dir: Path,
    enable_llm: bool = False,
    force: bool = False,
) -> Path:
    container = _get_container()

    # Tool config overrides for nightly run
    cfg = {
        "base_path": base_path,
        "force": force,
        "store_to_memory": True,
        "enable_ast_checks": True,
        "enable_ruff": True,
        "enable_llm": enable_llm,
        "cache_dir": str(Path("./.stephanie_cache/code_analysis").resolve()),
        # tighten excludes for repo runs if you want:
        # "exclude_globs": ["**/migrations/**", "**/site-packages/**", "**/.tox/**"],
    }

    tool = CodeAnalyzerTool(cfg, container.memory, container, getattr(container, "logger", None))

    run_id = _ts()
    sc = RepoScorable(id=f"repo:{base_path}")

    await tool.apply(sc, context={"run_id": run_id})

    payload = sc.meta.get("code_analysis", {}).get(tool.name, {})
    payload.setdefault("run_id", run_id)
    payload.setdefault("base_path", base_path)

    out_path = out_dir / f"{run_id}_code_analysis.json"
    _write_json_atomic(out_path, payload)

    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base",
        default=r"C:\Users\ernan\Project\stephanie\stephanie",
        help="Directory to analyze",
    )
    p.add_argument("--out", default=str(OUT_DIR_DEFAULT), help="Output directory")
    p.add_argument("--enable-llm", action="store_true", help="Enable LLM advice on top-K risky files")
    p.add_argument("--force", action="store_true", help="Ignore meta cache and recompute")
    args = p.parse_args()

    out_path = asyncio.run(
        run_once(
            base_path=args.base,
            out_dir=Path(args.out),
            enable_llm=args.enable_llm,
            force=args.force,
        )
    )
    print(f"Wrote: {out_path}")
    inspect_file(out_path, file_path="orm/ner_retriever.py")

def inspect_file(out_path: Path, file_path: str) -> None:
    j = json.load(open(out_path, "r", encoding="utf-8"))
    target = file_path
    f = next(x for x in j["files"] if x["rel_path"] == target)

    print("FILE:", f["rel_path"])
    print("RISK:", f["risk_score"])
    print("METRICS:", f["metrics"])
    print("\nTOP FINDINGS:")
    for it in f["findings"][:30]:
        print(f"- {it.get('severity'):>6} {it.get('kind'):>14}  L{it.get('line')}:{it.get('col')}  {it.get('message')}")
    print("\nTOP RUFF:")
    for it in (f.get("ruff") or [])[:20]:
        if it.get("code"):
            loc = it.get("location", {})
            print(f"- {it['code']} L{loc.get('row')}:{loc.get('column')} {it.get('message')}")


if __name__ == "__main__":
    main()
