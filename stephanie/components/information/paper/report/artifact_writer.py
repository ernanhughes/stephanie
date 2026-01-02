# stephanie/components/information/paper/report_artifact_writer.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from stephanie.utils.file_utils import save_to_timestamped_file, write_last_copy
from stephanie.utils.hash_utils import hash_text


@dataclass(frozen=True)
class ReportWriteResult:
    report_path: str
    report_last_path: Optional[str]
    blog_config_path: Optional[str]
    blog_config_last_path: Optional[str]


class ReportArtifactWriter:
    def __init__(self, *, report_dir: str, write_last: bool = True):
        self.report_dir = report_dir
        self.write_last = bool(write_last)

    def write_report_markdown(self, *, arxiv_id: str, md: str) -> ReportWriteResult:
        report_path = save_to_timestamped_file(
            data=md,
            output_dir=self.report_dir,
            file_prefix=f"{arxiv_id}_report",
            file_extension="md",
            last_filename=f"{arxiv_id}_report.last.md" if self.write_last else None,
        )
        return ReportWriteResult(
            report_path=str(report_path),
            report_last_path=str(Path(self.report_dir) / f"{arxiv_id}_report.last.md") if self.write_last else None,
            blog_config_path=None,
            blog_config_last_path=None,
        )

    def write_blog_config_snapshot(self, *, arxiv_id: str, blog_cfg: Dict[str, Any]) -> Dict[str, str]:
        blog_cfg_norm = json.dumps(blog_cfg, sort_keys=True)
        blog_cfg_hash = hash_text(blog_cfg_norm)

        cfg_path = save_to_timestamped_file(
            data=json.dumps(blog_cfg, indent=2, sort_keys=True),
            output_dir=self.report_dir,
            file_prefix=f"{arxiv_id}_blog_config",
            file_extension="json",
            last_filename=f"{arxiv_id}_blog_config.last.json" if self.write_last else None,
        )

        out: Dict[str, str] = {
            "blog_config_hash": blog_cfg_hash,
            "blog_config_path": str(cfg_path),
        }
        if self.write_last:
            out["blog_config_path_last"] = str(Path(self.report_dir) / f"{arxiv_id}_blog_config.last.json")
        return out

    def write_stable_last_artifacts(
        self,
        *,
        report_path: str,
        graph_html_path: Optional[str],
        graph_json_path: Optional[str],
        nexus_tree_json_path: Optional[str],
    ) -> None:
        # Convenience: stable 'last_*' artifacts at the report_dir level
        write_last_copy(source_path=report_path, last_path=f"{self.report_dir}/last_paper_report.md")
        if graph_html_path:
            write_last_copy(source_path=graph_html_path, last_path=f"{self.report_dir}/last_paper_graph.html")
        if graph_json_path:
            write_last_copy(source_path=graph_json_path, last_path=f"{self.report_dir}/last_paper_graph.json")
        if nexus_tree_json_path:
            write_last_copy(source_path=nexus_tree_json_path, last_path=f"{self.report_dir}/last_nexus_tree.json")
