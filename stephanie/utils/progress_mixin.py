# stephanie/utils/progress_mixin.py
from __future__ import annotations

class ProgressMixin:
    def _init_progress(self, container, logger=None):
        self._progress = None
        try:
            self._progress = container.get("progress")
        except Exception:
            if logger:
                try:
                    logger.log("ProgressServiceUnavailable", {})
                except Exception:
                    pass

    def pstart(self, task: str, **kw):
        if self._progress:
            self._progress.start(task=task, **kw)

    def pstage(self, task: str, stage: str, **kw):
        if self._progress:
            self._progress.stage(task=task, stage=stage, **kw)

    def ptick(self, task: str, done: int, total: int, **kw):
        # use absolute setter for clarity
        if self._progress:
            self._progress.set(task=task, done=done, total=total, **kw)

    def pdone(self, task: str, **kw):
        if self._progress:
            # compatible with both .end and .done alias
            self._progress.end(task=task, **kw)
