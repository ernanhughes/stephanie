from __future__ import annotations

class ProgressMixin:
    def _init_progress(self, container, logger=None):
        self._progress = None
        self._progress_state = {}  # track done/total per task
        try:
            self._progress = container.get("progress")
        except Exception:
            if logger:
                try:
                    logger.log("ProgressServiceUnavailable", {})
                except Exception:
                    pass

    def pstart(self, task: str, **kw):
        # capture initial totals if provided
        total = kw.get("total")
        self._progress_state[task] = {"done": 0, "total": total}
        if self._progress:
            self._progress.start(task=task, **kw)

    def pstage(self, task: str, stage: str, **kw):
        if self._progress:
            self._progress.stage(task=task, stage=stage, **kw)

    def ptick(self, task: str, done: int, total: int, **kw):
        # absolute setter
        self._progress_state[task] = {"done": done, "total": total}
        if self._progress:
            self._progress.set(task=task, done=done, total=total, **kw)

    def pstep(self, task: str, n: int = 1, **kw):
        state = self._progress_state.get(task, {"done": 0, "total": kw.get("total")})
        state["done"] = (state.get("done") or 0) + int(n)
        # allow caller to update total on the fly
        if "total" in kw and kw["total"] is not None:
            state["total"] = kw["total"]
        self._progress_state[task] = state
        if self._progress:
            self._progress.set(task=task, done=state["done"], total=state.get("total"), **kw)

    def pdone(self, task: str, **kw):
        if self._progress:
            self._progress.end(task=task, **kw)
        # cleanup local state
        self._progress_state.pop(task, None)
