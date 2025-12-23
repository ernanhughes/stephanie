# stephanie/components/information/tasks/paper_import_task.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from stephanie.tools.paper_import_tool import (PaperImportResult,
                                               PaperImportTool)

log = logging.getLogger(__name__)


class PaperImportTask:
    """
    Backwards-compatible wrapper.

    ✅ Existing code can keep doing:
        import_task = PaperImportTask(papers_root=...)
        res = await import_task.run(arxiv_id=..., role=...)

    🚀 New code should prefer constructing PaperImportTool directly with memory/container
    so it can reuse PaperStore + PaperReferenceStore.
    """

    def __init__(
        self,
        cfg,
        memory,
        container,
        logger,
    ):
        self.tool = PaperImportTool(cfg=cfg, memory=memory, container=container, logger=logger or log)

    async def run(
        self,
        *,
        arxiv_id: Optional[str] = None,
        url: Optional[str] = None,
        local_pdf_path: Optional[str | Path] = None,
        role: Optional[str] = None,
        source: str = "arxiv",
        force: bool = False,
        force_references: bool = False,
    ) -> PaperImportResult:
        return await self.tool.import_paper(
            arxiv_id=arxiv_id,
            url=url,
            local_pdf_path=local_pdf_path,
            role=role,
            source=source,
            force=True,
            force_references=force_references,
        )
