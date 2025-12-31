# stephanie/components/information/agents/section_subgraph_forkjoin.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from stephanie.constants import PIPELINE_RUN_ID
from stephanie.services.knowledge_graph.subgraphs.subgraph_builder import (
    SubgraphBuilder, SubgraphConfig)

log = logging.getLogger(__name__)


# -----------------------------
# Helpers
# -----------------------------

_SAFE_CHARS_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_filename(s: str, *, max_len: int = 140) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    s = _SAFE_CHARS_RE.sub("_", s)
    s = s.strip("_")
    return s[:max_len] if len(s) > max_len else s


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _dump_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _get_section_fields(section: Any) -> Tuple[str, str, str, Dict[str, Any], Optional[int]]:
    """
    Normalize a 'section-like' object into fields we can use for query building and artifact naming.
    Supports:
      - dataclass/ORM objects with .id/.title/.text/.meta/.section_index
      - dict-like sections
    """
    if isinstance(section, dict):
        sid = str(section.get("id") or "")
        title = str(section.get("title") or "")
        text = str(section.get("text") or "")
        meta = section.get("meta") or {}
        sidx = section.get("section_index")
        sidx = int(sidx) if isinstance(sidx, int) or (isinstance(sidx, str) and sidx.isdigit()) else None
        return sid, title, text, dict(meta), sidx

    sid = str(getattr(section, "id", "") or "")
    title = str(getattr(section, "title", "") or "")
    text = str(getattr(section, "text", "") or "")
    meta = getattr(section, "meta", None) or {}
    sidx = getattr(section, "section_index", None)
    try:
        sidx = int(sidx) if sidx is not None else None
    except Exception:
        sidx = None
    return sid, title, text, dict(meta), sidx


def _get_spine_elements(spine: Any) -> List[Dict[str, Any]]:
    """
    Try to extract element-like dicts from a spine object, if present.
    We only use lightweight fields to enrich the query (e.g., captions/labels).
    """
    if spine is None:
        return []
    elems = getattr(spine, "elements", None)
    if elems is None and isinstance(spine, dict):
        elems = spine.get("elements")
    if not elems:
        return []
    out: List[Dict[str, Any]] = []
    for e in elems:
        if isinstance(e, dict):
            out.append(e)
        else:
            out.append(
                {
                    "type": getattr(e, "type", None) or getattr(e, "kind", None),
                    "label": getattr(e, "label", None),
                    "caption": getattr(e, "caption", None),
                    "page": getattr(e, "page", None),
                }
            )
    return out


def build_section_query(
    section: Any,
    *,
    spine: Optional[Any] = None,
    max_text_chars: int = 1800,
) -> str:
    """
    Deterministic, embedding-friendly query for SubgraphBuilder:
      title + key meta + (small) body text + element captions/labels + entities.
    """
    sid, title, text, meta, _sidx = _get_section_fields(section)

    summary = str(meta.get("summary") or meta.get("abstract") or "")
    entities = meta.get("entities") or meta.get("ner") or []
    if isinstance(entities, dict):
        flat: List[str] = []
        for _k, vs in entities.items():
            if isinstance(vs, list):
                flat.extend([str(x) for x in vs])
        entities = flat
    if not isinstance(entities, list):
        entities = []

    body = (text or "").strip()
    if len(body) > max_text_chars:
        body = body[:max_text_chars].rstrip() + "…"

    elem_bits: List[str] = []
    for e in _get_spine_elements(spine):
        cap = (e.get("caption") or "").strip()
        lab = (e.get("label") or "").strip()
        typ = (e.get("type") or "").strip()
        if cap:
            elem_bits.append(f"{typ} caption: {cap}" if typ else f"caption: {cap}")
        elif lab:
            elem_bits.append(f"{typ} label: {lab}" if typ else f"label: {lab}")

    ent_bits = ", ".join([str(x) for x in entities[:25] if str(x).strip()])

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Summary: {summary}")
    if ent_bits:
        parts.append(f"Entities: {ent_bits}")
    if elem_bits:
        parts.append("Elements:\n" + "\n".join(elem_bits[:20]))
    if body:
        parts.append("Text:\n" + body)

    q = "\n\n".join([p for p in parts if p.strip()])
    return q.strip() or sid


# -----------------------------
# Evidence pack outputs
# -----------------------------

@dataclass
class SubgraphArtifactRef:
    goal_type: str
    cfg: Dict[str, Any]
    artifact_path: str
    node_count: int = 0
    edge_count: int = 0
    edge_types: Dict[str, int] = field(default_factory=dict)
    evidence_rate: float = 0.0
    confidence_p50: float = 0.0
    confidence_p90: float = 0.0


@dataclass
class SectionPack:
    section_id: str
    section_index: Optional[int]
    title: str
    start_page: Optional[int]
    end_page: Optional[int]
    query_preview: str
    subgraphs: List[SubgraphArtifactRef] = field(default_factory=list)


# -----------------------------
# Agent
# -----------------------------

class SectionSubgraphForkJoinAgent:
    """
    Build goal-conditioned subgraphs per section spine, in parallel, and dump artifacts:

      runs/paper_blogs/{run_id}/{paper_id}/subgraphs/{goal_type}/sec_{idx}_{slug}.json
      runs/paper_blogs/{run_id}/{paper_id}/section_packs.json
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory=None,
        container=None,
        logger=None,
        *,
        subgraph_builder: Optional[SubgraphBuilder] = None,
    ) -> None:
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger
        self.subgraph_builder = subgraph_builder

        self.max_inflight = int(self.cfg.get("max_inflight", 8))

        self.goal_cfgs: Dict[str, SubgraphConfig] = {}
        goals = (self.cfg.get("goals") or {})
        for goal_type, g in goals.items():
            if isinstance(g, SubgraphConfig):
                self.goal_cfgs[str(goal_type)] = g
            else:
                self.goal_cfgs[str(goal_type)] = SubgraphConfig(**(g or {}))

        self.verifier_goal_type = self.cfg.get("verifier_goal_type")
        self.max_query_preview = int(self.cfg.get("max_query_preview", 320))

    def _get_builder(self) -> SubgraphBuilder:
        if self.subgraph_builder is not None:
            return self.subgraph_builder
        b = getattr(self.container, "subgraph_builder", None) if self.container else None
        if b is None:
            raise RuntimeError(
                "SectionSubgraphForkJoinAgent requires a SubgraphBuilder. "
                "Pass subgraph_builder=... or expose container.subgraph_builder."
            )
        return b

    def _resolve_run_dir(self, *, run_id: Any) -> str:
        run_dir = self.cfg.get("run_dir")
        if run_dir:
            return str(run_dir)
        return f"runs/paper_blogs/{run_id}"

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_id = context.get(PIPELINE_RUN_ID)
        arxiv_id = context.get("arxiv_id")
        run_dir = self._resolve_run_dir(run_id=run_id)

        builder = self._get_builder()

        # ✅ FIX 1: define semaphore here (used by build_one)
        sem = asyncio.Semaphore(self.max_inflight)

        spines = context.get("section_spines")
        sections = context.get("paper_sections")

        spine_pairs: List[Tuple[Any, Optional[Any]]] = []

        if spines:
            for sp in spines:
                sec = getattr(sp, "section", None)
                if sec is None and isinstance(sp, dict):
                    sec = sp.get("section") or sp.get("section_obj") or sp.get("section_data")
                spine_pairs.append((sec or sp, sp))
        elif sections:
            for sec in sections:
                spine_pairs.append((sec, None))
        else:
            log.warning("[SectionSubgraphForkJoinAgent] no sections/spines found in context")
            return context

        goal_types = list(self.goal_cfgs.keys())
        if self.verifier_goal_type and self.verifier_goal_type not in goal_types:
            vcfg = self.cfg.get("verifier_goal_cfg")
            if vcfg:
                self.goal_cfgs[self.verifier_goal_type] = SubgraphConfig(**(vcfg or {}))
                goal_types.append(self.verifier_goal_type)

        if not goal_types:
            log.warning("[SectionSubgraphForkJoinAgent] cfg.goals is empty; nothing to build")
            return context

        out_dir = os.path.join(run_dir, str(arxiv_id))

        async def build_one(section_obj: Any, spine_obj: Optional[Any], goal_type: str) -> SubgraphArtifactRef:
            cfg = self.goal_cfgs[goal_type]
            q = build_section_query(section_obj, spine=spine_obj)

            # ✅ FIX 2: builder.build is sync; don't block event loop
            async with sem:
                sg = await asyncio.to_thread(builder.build, query=q, cfg=cfg)

            sid, title, _text, meta, sidx = _get_section_fields(section_obj)

            slug = _safe_filename(title or sid)
            idx_part = f"{int(sidx):03d}_" if sidx is not None else ""
            art_rel = os.path.join("subgraphs", goal_type, f"sec_{idx_part}{slug}.json")
            art_path = os.path.join(out_dir, art_rel)
            _dump_json(art_path, sg)

            m = (sg or {}).get("meta") or {}
            st = (m.get("stats") or {}) if isinstance(m, dict) else {}
            return SubgraphArtifactRef(
                goal_type=goal_type,
                cfg=asdict(cfg),
                artifact_path=art_path,
                node_count=int(st.get("node_count") or len((sg or {}).get("nodes") or [])),
                edge_count=int(st.get("edge_count") or len((sg or {}).get("edges") or [])),
                edge_types=dict(st.get("edge_types") or {}),
                evidence_rate=float(st.get("evidence_rate") or 0.0),
                confidence_p50=float(st.get("confidence_p50") or 0.0),
                confidence_p90=float(st.get("confidence_p90") or 0.0),
            )

        async def build_section_pack(section_obj: Any, spine_obj: Optional[Any]) -> SectionPack:
            sid, title, _text, meta, sidx = _get_section_fields(section_obj)
            start_page = meta.get("start_page")
            end_page = meta.get("end_page")

            query = build_section_query(section_obj, spine=spine_obj)
            preview = query[: self.max_query_preview].replace("\n", " ").strip()

            tasks = [build_one(section_obj, spine_obj, gt) for gt in goal_types]
            refs = await asyncio.gather(*tasks)

            return SectionPack(
                section_id=sid or _safe_filename(title),
                section_index=sidx,
                title=title,
                start_page=int(start_page) if isinstance(start_page, int) or (isinstance(start_page, str) and str(start_page).isdigit()) else None,
                end_page=int(end_page) if isinstance(end_page, int) or (isinstance(end_page, str) and str(end_page).isdigit()) else None,
                query_preview=preview,
                subgraphs=list(refs),
            )

        section_tasks = [build_section_pack(sec, sp) for (sec, sp) in spine_pairs if sec is not None]
        packs = await asyncio.gather(*section_tasks)

        packs_path = os.path.join(out_dir, "section_packs.json")
        _dump_json(packs_path, {"paper_id": arxiv_id, "run_id": run_id, "sections": [asdict(p) for p in packs]})

        log.info("[SectionSubgraphForkJoinAgent] wrote %d section packs → %s", len(packs), packs_path)

        context["section_packs_path"] = packs_path
        context["section_packs"] = [asdict(p) for p in packs]
        return context
