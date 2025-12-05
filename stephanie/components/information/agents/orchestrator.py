# stephanie/components/information/agents/information_orchestrator_agent.py
from __future__ import annotations

from typing import Any, Dict, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.core.context.context_manager import ContextManager

from stephanie.components.information.orchestrator import (
    InformationOrchestrator,
    InformationOrchestratorConfig,
)
from stephanie.components.information.builder.bucket import BucketBuilder
from stephanie.components.information.adapters.info_casebook_adapter import InfoCaseBookAdapter
from stephanie.components.information.builder.memcube import MemCubeBuilder


class InformationOrchestratorAgent(BaseAgent):
    """
    Thin wrapper around InformationOrchestrator so it can be used as a pipeline stage.

    Given a topic (from config or context), it:
      - builds an Information Bucket
      - builds a CaseBook + Cases
      - builds an Information MemCube
      - runs ScorableProcessor
      - writes an 'information_build' block into the ContextManager
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger) -> None:
        super().__init__(cfg, memory, container, logger)

        # --- agent-level config ---
        self.topic_key: str = cfg.get("topic_key", "topic")
        # Where to look for the topic in the context, e.g. context["goal"]["topic"]
        self.topic_path: str = cfg.get("topic_path", "goal.topic")

        # Orchestrator config
        orch_cfg = InformationOrchestratorConfig(
            default_source_profile=cfg.get(
                "default_source_profile", "research_web_mixed"
            ),
            target=cfg.get("target", "memcube_blog_post"),
        )

        # Builders use memory + container.llm
        llm_client = container.get_llm(cfg.get("llm_name", "information_builder"))

        bucket_builder = BucketBuilder(
            memory=memory,
            container=container,
            logger=logger,
            llm_client=llm_client,
        )

        casebook_adapter = InfoCaseBookAdapter(
            casebook_store=memory.casebooks,
            logger=logger,
            llm_client=llm_client,
        )

        memcube_builder = MemCubeBuilder(
            memcube_store=memory.memcubes,
            casebook_store=memory.casebooks,
            logger=logger,
            llm_client=llm_client,
        )

        # ScorableProcessor is usually already registered in the container
        scorable_processor = container.resolve("scorable_processor")

        self.orchestrator = InformationOrchestrator(
            cfg=orch_cfg,
            bucket_builder=bucket_builder,
            casebook_adapter=casebook_adapter,
            memcube_builder=memcube_builder,
            scorable_processor=scorable_processor,
            memory=memory,
            container=container,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Pipeline entrypoint
    # ------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> ContextManager:

        topic = self._resolve_topic(context)
        if not topic:
            self.logger.error("InformationOrchestratorAgent: no topic found; skipping")
            return context

        source_profile = self.cfg.get("source_profile")  # optional override
        target = self.cfg.get("target")  # optional override

        self.logger.log(
            "InformationOrchestratorStart",
            {"topic": topic, "source_profile": source_profile, "target": target},
        )

        build_result = await self.orchestrator.run(
            topic=topic,
            target=target,
            source_profile=source_profile,
            options={"pipeline_run_id": context.get("pipeline_run_id")},
        )

        # Write a compact summary back into the context
        context["information_build"] = {
            "status": "ok",
            "topic": build_result.topic,
            "target": build_result.target,
            "memcube_id": build_result.memcube_id,
            "casebook_id": build_result.casebook_id,
            "preview_markdown": build_result.preview_markdown,
            "attributes": build_result.attributes,
            "extra_data": build_result.extra_data,
        }

        # Convenience shortcuts for downstream stages
        context["memcube_id"] = build_result.memcube_id
        context.setdefault("blog", {})
        context["blog"]["markdown"] = build_result.preview_markdown

        self.logger.log(
            "InformationOrchestratorDone",
            {
                "topic": build_result.topic,
                "memcube_id": build_result.memcube_id,
                "casebook_id": build_result.casebook_id,
            },
        )

        context.data = context
        return context

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_topic(self, ctx: Dict[str, Any]) -> Optional[str]:
        """
        Try multiple places to find a topic string.
        Priority:
          1) explicit cfg["topic"]
          2) context path in cfg["topic_path"], e.g. "goal.topic"
          3) context["documents"][0]["title"], if present
        """
        # 1) explicit override
        if "topic" in self.cfg:
            return self.cfg["topic"]

        # 2) nested path like "goal.topic"
        path = self.topic_path.split(".") if self.topic_path else []
        node: Any = ctx
        try:
            for part in path:
                if not part:
                    continue
                if isinstance(node, dict):
                    node = node.get(part)
                else:
                    node = getattr(node, part, None)
            if isinstance(node, str) and node.strip():
                return node.strip()
        except Exception:
            pass

        # 3) use first document title if available
        docs = ctx.get("documents") or []
        if docs and isinstance(docs, list):
            title = docs[0].get("title") or docs[0].get("name")
            if title:
                return str(title).strip()

        return None
