# stephanie/utils/progress_mixin.py
from __future__ import annotations

from typing import Any, Dict, Optional


class ProgressMixin:
    """
    Small helper mixin to talk to an optional ProgressService.

    Expected ProgressService interface (minimal):
        start(task: str, total: int, meta: Optional[Dict[str, Any]] = None)
        stage(task: str, stage: str, **kw)
        set(task: str, done: int, total: Optional[int] = None,
            substage: Optional[str] = None, extra: Optional[Dict[str, Any]] = None)
        tick(task: str, n: int = 1, substage: Optional[str] = None,
             extra: Optional[Dict[str, Any]] = None, **kw)
        end(task: str, status: str = "ok", extra: Optional[Dict[str, Any]] = None)
    """

    def _init_progress(self, container, logger=None):
        """
        Call this once in your __init__:

            self._init_progress(container, logger)

        so that the mixin can find an optional ProgressService.
        """
        self._progress = None
        self._progress_state: Dict[str, Dict[str, Any]] = {}  # track done/total per task
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

    def pstart(self, task: str, total: Optional[int] = None, **kw):
        """
        Start a logical progress task.

        Example:
            self.pstart("ScorableProcess:8967", total=len(scorables))

        'total' is optional but strongly recommended; if omitted,
        we default to total=1 so ProgressService doesn't explode.
        """
        if total is None:
            total = 1

        self._progress_state[task] = {"done": 0, "total": int(total)}

        if self._progress:
            meta = kw.get("meta")
            try:
                # ProgressService.start(task, total, meta=...)
                self._progress.start(task=task, total=int(total), meta=meta)
            except TypeError:
                # Backward-compat if signature changes; best effort
                self._progress.start(task=task, total=int(total))

    def pstage(self, task: str, stage: str, **kw):
        """
        Mark a stage change for a task, without touching done/total.

        This is for rare "jump" updates like:
            self.pstage("scorable_processor", "finalize")

        NOTE: This will immediately emit a log line (uses ProgressService.stage),
              so it shouldn't be spammed in tight loops.
        """
        if self._progress:
            self._progress.stage(task=task, stage=stage, **kw)

    def ptick(
        self,
        task: str,
        done: int,
        total: Optional[int] = None,
        stage: Optional[str] = None,
        **kw: Any,
    ):
        """
        Set the absolute progress for a task.

        If 'stage' is provided, it is passed as the ProgressService 'substage'
        so that each tick produces **ONE** console line like:

            [progress] SPItem:gsm8k-train-53:  80% (4/5) | scores+vpm

        instead of two separate lines.
        """
        # Update local state
        state = self._progress_state.get(task, {"done": 0, "total": total or 1})
        state_done = int(done)
        state_total = int(total) if total is not None else int(state.get("total") or 1)
        state["done"] = state_done
        state["total"] = state_total
        self._progress_state[task] = state

        if not self._progress:
            return

        # Separate 'total' from other extras to avoid duplicate kwargs
        extra = {k: v for k, v in kw.items() if k != "total"}

        # CRITICAL CHANGE:
        #   - we DO NOT call .stage() here
        #   - we pass 'stage' as 'substage' into .set()
        self._progress.set(
            task=task,
            done=state_done,
            total=state_total,
            substage=stage,
            extra=extra or None,
        )

    def pstep(
        self,
        task: str,
        n: int = 1,
        stage: Optional[str] = None,
        **kw: Any,
    ):
        """
        Increment progress for a task by n (default = 1).

        You can optionally pass:
            - stage="embed" / "domains" / ...
            - total=...  (to adjust total dynamically)

        Example (per-item subprogress):
            self.pstep(f"SPItem:{item_id}", 1, stage="embed")
        """
        state = self._progress_state.get(task, {"done": 0, "total": kw.get("total") or 1})

        state_done = int(state.get("done") or 0) + int(n)
        state_total = int(state.get("total") or 1)

        # allow caller to update total on the fly
        if "total" in kw and kw["total"] is not None:
            state_total = int(kw["total"])

        state["done"] = state_done
        state["total"] = state_total
        self._progress_state[task] = state

        if not self._progress:
            return

        extra = {k: v for k, v in kw.items() if k != "total"}

        # Same pattern as ptick: ONE call, stage → substage
        self._progress.set(
            task=task,
            done=state_done,
            total=state_total,
            substage=stage,
            extra=extra or None,
        )

    def pdone(self, task: str, status: str = "ok", **kw):
        """
        Mark task done and clean up local state.
        """
        if self._progress:
            extra = kw.get("extra")
            self._progress.end(task=task, status=status, extra=extra)
        self._progress_state.pop(task, None)
