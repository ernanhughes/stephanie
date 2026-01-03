# stephanie/agents/target_finalize.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from stephanie.tools.target_tool import TargetTool


class TargetFinalizeAgent(BaseAgent):
    """
    End-of-pipeline agent:
      - registers final artifacts as Targets
      - links them to input sources if present

    Expected context keys (you can change these):
      - run_id (or pipeline_run_id)
      - blog_path (optional)
      - report_path (optional)
      - book_path (optional)
      - template_path (optional)
      - source_ids (optional) : list[int] sources that fed the output
      - root_node_type/root_node_id (optional) : anchor to KG node
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.enabled = bool(cfg.get("enabled", True))
        self.run_dir = cfg.get("run_dir", "runs/paper_blogs/${run_id}")

        self.tool = TargetTool(memory=self.memory, logger=self.logger, run_dir_tpl=self.run_dir)

        # map context keys -> target_type
        self.targets_map = cfg.get("targets_map", {
            "blog_path": "blog_post",
            "report_path": "report",
            "book_path": "book",
            "template_path": "template",
        })

    async def run(self, context: dict) -> dict:
        if not self.enabled:
            return context

        run_id = context.get("run_id") or context.get("pipeline_run_id")
        if not run_id:
            return context

        source_ids = context.get("source_ids") or []
        root_node_type = context.get("root_node_type")
        root_node_id = context.get("root_node_id")

        registered = []

        for ctx_key, target_type in (self.targets_map or {}).items():
            path = context.get(ctx_key)
            if not path:
                continue

            rec = self.tool.register_existing_file(
                pipeline_run_id=int(run_id),
                target_type=target_type,
                target_uri=str(path),
                title=context.get("title"),
                root_node_type=root_node_type,
                root_node_id=root_node_id,
                input_source_ids=source_ids,
                input_relation_type="derived_from",
            )
            registered.append(rec.__dict__)

        context["targets_registered"] = registered
        return context
