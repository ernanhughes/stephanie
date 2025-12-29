# stephanie/scoring/scorer/model_health.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from stephanie.utils.hash_utils import hash_dict

log = logging.getLogger(__name__)


@dataclass
class LoadAudit:
    missing: Sequence[str]
    unexpected: Sequence[str]

    @property
    def ok(self) -> bool:
        return (not self.missing) and (not self.unexpected)


def audit_load_state_dict(
    model, state_dict: Dict[str, Any], *, strict: bool = False
) -> LoadAudit:
    """
    Loads state_dict and returns missing/unexpected keys.
    Always uses strict=False by default, but audits it.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return LoadAudit(missing=missing, unexpected=unexpected)


def run_self_test_if_available(
    model, *, device: str = "cpu", n_trials: int = 8
):
    fn = getattr(model, "self_test", None)
    if not callable(fn):
        return None

    try:
        out = fn(device=device, n_trials=n_trials)
        if isinstance(out, dict) and "ok" in out:
            return out
        return {"ok": bool(out), "summary": str(out)}
    except Exception as e:
        # Important: don't break scorer init; treat as a failed self-test.
        return {
            "ok": False,
            "summary": f"self_test raised: {type(e).__name__}: {e}",
        }


class ModelHealthLogger:
    """
    Centralized logging policy:
      - error on load missing/unexpected
      - warning only when self_test ok=False OR load audit not ok
      - silent on success
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        self_test_enabled: bool = True,
        self_test_device: str = "cpu",
        self_test_trials: int = 8,
        warn_on_load_issues: bool = True,
        warn_on_selftest_fail: bool = True,
    ) -> None:
        self.enabled = enabled
        self.self_test_enabled = self_test_enabled
        self.self_test_device = self_test_device
        self.self_test_trials = self_test_trials
        self.warn_on_load_issues = warn_on_load_issues
        self.warn_on_selftest_fail = warn_on_selftest_fail

        # Optional: prevent repeated logs per scorer instance
        self._seen: set[tuple[str, str]] = set()  # (scorer_name, dimension)

    def check_and_log(
        self,
        *,
        scorer_name: str,
        dimension: str,
        model_name: str,
        model,
        load_audit: Optional[LoadAudit] = None,
    ) -> None:
        if not self.enabled:
            return

        key = (scorer_name, dimension)
        if key in self._seen:
            return
        self._seen.add(key)

        # 1) Load audit logs (always actionable)
        if load_audit is not None and not load_audit.ok:
            # error with small examples
            if load_audit.missing:
                log.error(
                    "%s model load missing keys dim=%s model=%s count=%d example=%s",
                    scorer_name,
                    dimension,
                    model_name,
                    len(load_audit.missing),
                    list(load_audit.missing)[:8],
                )
            if load_audit.unexpected:
                log.error(
                    "%s model load unexpected keys dim=%s model=%s count=%d example=%s",
                    scorer_name,
                    dimension,
                    model_name,
                    len(load_audit.unexpected),
                    list(load_audit.unexpected)[:8],
                )
            if self.warn_on_load_issues:
                log.warning(
                    "%s model load issues detected dim=%s model=%s (missing=%d unexpected=%d)",
                    scorer_name,
                    dimension,
                    model_name,
                    len(load_audit.missing),
                    len(load_audit.unexpected),
                )

        # 2) Self-test (warn only on failure)
        if self.self_test_enabled:
            try:
                st = run_self_test_if_available(
                    model,
                    device=self.self_test_device,
                    n_trials=self.self_test_trials,
                )
                if st is None:
                    return

                ok = bool(st.get("ok", False))
                if load_audit is not None and not load_audit.ok:
                    # treat load issues as failure even if self-test passes
                    ok = False

                if (not ok) and self.warn_on_selftest_fail:
                    summary = st.get("summary", "")
                    line = (
                        summary.splitlines()[1]
                        if summary and "\n" in summary
                        else summary
                    )
                    log.warning(
                        "%s self_test FAILED dim=%s model=%s %s",
                        scorer_name,
                        dimension,
                        model_name,
                        line,
                    )
            except Exception as e:
                log.exception(
                    "%s self_test crashed dim=%s model=%s err=%s",
                    scorer_name,
                    dimension,
                    model_name,
                    e,
                )

    def check_loaded_model(
        self,
        *,
        scorer_name: str,
        dimension: str,
        model_name: str,
        model,
        load_audit: Optional[LoadAudit] = None,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns a dict you can stash if you want:
          {'ok': bool, 'load_ok': bool, 'self_test': {...}|None, 'model_id': str}
        Logging policy:
          - ERROR on load issues (missing/unexpected)
          - WARNING only if overall ok=False
          - SILENT on success
        """
        if not self.enabled:
            return {
                "ok": True,
                "load_ok": True,
                "self_test": None,
                "model_id": model_id,
            }

        # Cache key: (scorer, dim, model_id-ish)
        mid = model_id or hash_dict(model_name)
        key = (scorer_name, dimension, mid)
        if key in self._seen:
            return {
                "ok": True,
                "load_ok": True,
                "self_test": None,
                "model_id": mid,
            }
        self._seen.add(key)

        load_ok = True
        if load_audit is not None and not load_audit.ok:
            load_ok = False
            if load_audit.missing:
                log.error(
                    "%s model load missing keys dim=%s model=%s count=%d example=%s",
                    scorer_name,
                    dimension,
                    model_name,
                    len(load_audit.missing),
                    list(load_audit.missing)[:8],
                )
            if load_audit.unexpected:
                log.error(
                    "%s model load unexpected keys dim=%s model=%s count=%d example=%s",
                    scorer_name,
                    dimension,
                    model_name,
                    len(load_audit.unexpected),
                    list(load_audit.unexpected)[:8],
                )

        st = None
        st_ok = True
        if self.self_test_enabled:
            st = run_self_test_if_available(
                model,
                device=self.self_test_device,
                n_trials=self.self_test_trials,
            )
            if st is not None:
                st_ok = bool(st.get("ok", False))

        ok = bool(load_ok and st_ok)

        # WARN only on failure
        if not ok:
            summary = ""
            if isinstance(st, dict):
                summary = st.get("summary", "")
            line = (
                summary.splitlines()[1]
                if summary and "\n" in summary
                else summary
            )
            log.warning(
                "%s model health FAILED dim=%s model=%s load_ok=%s self_test_ok=%s %s",
                scorer_name,
                dimension,
                model_name,
                load_ok,
                st_ok,
                line,
            )

        return {"ok": ok, "load_ok": load_ok, "self_test": st, "model_id": mid}
