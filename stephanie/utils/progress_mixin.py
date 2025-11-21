# stephanie/utils/progress_mixin.py
from __future__ import annotations


class ProgressMixin:
    """
    Small helper mixin to talk to an optional ProgressService.

    Expected ProgressService interface (minimal):
        start(task: str, **kw)
        stage(task: str, stage: str, **kw)
        set(task: str, done: int, total: int | None = None, **kw)
        end(task: str, **kw)
    """

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

    # ------------------------------------------------------------------ #
    # High-level helpers
    # ------------------------------------------------------------------ #

    def pstart(self, task: str, **kw):
        """
        Start a logical progress task.

        Example:
            self.pstart("scorable_processor", total=len(scorables))
        """
        total = kw.get("total")
        self._progress_state[task] = {"done": 0, "total": total}
        if self._progress:
            self._progress.start(task=task, **kw)

    def pstage(self, task: str, stage: str, **kw):
        """
        Mark a stage change for a task, without touching done/total.
        """
        if self._progress:
            self._progress.stage(task=task, stage=stage, **kw)

    def ptick(self, task: str, done: int, total: int | None, stage: str | None = None, **kw):
        """
        Set the absolute progress for a task.

        If stage is provided, we emit a stage event as well.
        """
        self._progress_state[task] = {"done": done, "total": total}
        if not self._progress:
            return

        if stage is not None:
            # IMPORTANT: stage is sent only to .stage(), not to .set()
            self._progress.stage(task=task, stage=stage)

        self._progress.set(task=task, done=done, total=total, **kw)

    def pstep(self, task: str, n: int = 1, stage: str | None = None, **kw):
        """
        Increment progress for a task by n (default = 1).

        You can optionally pass a new total via kw["total"], and an optional stage:
            self.pstep("scorable_processor", 1, stage="hydrate")
        """
        state = self._progress_state.get(task, {"done": 0, "total": kw.get("total")})
        state_done = int(state.get("done") or 0) + int(n)
        state_total = state.get("total")

        # allow caller to update total on the fly
        if "total" in kw and kw["total"] is not None:
            state_total = kw["total"]

        state["done"] = state_done
        state["total"] = state_total
        self._progress_state[task] = state

        if not self._progress:
            return

        if stage is not None:
            # IMPORTANT: do not leak 'stage' into .set()
            self._progress.stage(task=task, stage=stage)

        self._progress.set(task=task, done=state_done, total=state_total, **{k: v for k, v in kw.items() if k != "total"})

    def pdone(self, task: str, **kw):
        """
        Mark task done and clean up local state.
        """
        if self._progress:
            self._progress.end(task=task, **kw)
        self._progress_state.pop(task, None)
