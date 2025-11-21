# stephanie/agents/ Oh my God sorry they didn't have cervix on sale I'm going to eat all that tonight the last sweets I've ever eaten loader.py
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable


class LoaderAgent(BaseAgent):
    """
    Offline loader for Qen3 runs.
    Reads one or more JSONL files produced by the VisualIntrospectionAgent
    (each line is a dict with a `scorable` payload and meta like `is_correct`),
    converts them back into Scorable objects, and splits them into:

      - context[self.output_key]      = all scorables (after limits)
      - context["scorables_targeted"] = correct scorables (is_correct=True)
      - context["scorables_baseline"] = incorrect scorables

    This is designed to feed directly into VisiCalcAgent.
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        # Where the JSONL logs live (you pass these in via Hydra)
        self.log_dirs: List[Path] = [Path(p) for p in cfg.get("log_dirs", [])]
        self.log_glob: str = cfg.get("log_glob", "*.jsonl")

        # Sampling / shuffling knobs
        self.shuffle: bool = bool(cfg.get("shuffle", True))

        # 0 or negative → no global limit
        mr = int(cfg.get("max_records", 0))
        self.max_records: Optional[int] = mr if mr > 0 else None

        # Per-cohort limits (0/None/negative → no limit)
        mg = cfg.get("max_good", 40)
        mb = cfg.get("max_bad", 40)

        try:
            mg = int(mg) if mg is not None else None
        except (TypeError, ValueError):
            mg = None
        try:
            mb = int(mb) if mb is not None else None
        except (TypeError, ValueError):
            mb = None

        self.max_good: Optional[int] = mg if mg and mg > 0 else None
        self.max_bad: Optional[int] = mb if mb and mb > 0 else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_log_files(self) -> List[Path]:
        paths: List[Path] = []
        for d in self.log_dirs:
            if not d.exists():
                self.logger.log(
                    "Gsm8kLogLoaderMissingDir",
                    {"dir": str(d)},
                )
                continue
            for p in sorted(d.glob(self.log_glob)):
                if p.is_file():
                    paths.append(p)
        return paths

    def _load_records_from_file(self, path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except json.JSONDecodeError as e:
                    self.logger.log(
                        "Gsm8kLogLoaderBadLine",
                        {"file": str(path), "error": str(e), "line_prefix": line[:120]},
                    )
        return records

    def _build_scorable_from_record(self, rec: Dict[str, Any]) -> Tuple[Scorable, bool]:
        """
        Convert a single JSONL record into a Scorable and return (scorable, is_correct).

        We trust the `scorable` field that the generator agent wrote, and
        use its meta (which includes `is_correct`).
        """
        sc_data = rec.get("scorable") or {}
        text = sc_data.get("text", "")
        external_id = sc_data.get("external_id") or rec.get("problem_id")
        meta = dict(sc_data.get("meta") or {})

        is_correct = bool(meta.get("is_correct", False))

        sc = Scorable(
            text=text,
            target_type="custom",
            id=external_id,
            meta=meta,
        )
        return sc, is_correct

    def _load_scorables_from_logs(self) -> Tuple[List[Scorable], List[Scorable], List[Scorable]]:
        """
        Returns:
          (all_scorables, scorables_targeted, scorables_baseline)

        Respects:
          - self.max_records  (total cap, if set)
          - self.max_good     (cap on correct, if set)
          - self.max_bad      (cap on incorrect, if set)
        """
        files = self._discover_log_files()
        if not files:
            raise RuntimeError("Gsm8kLogLoaderAgent: no JSONL files found in configured log_dirs")

        all_records: List[Dict[str, Any]] = []
        for p in files:
            recs = self._load_records_from_file(p)
            all_records.extend(recs)

        if self.shuffle:
            random.shuffle(all_records)

        # We *could* pre-slice here by max_records, but since we also have
        # per-cohort limits, we just enforce all limits in the loop below.
        max_total = self.max_records
        max_good = self.max_good
        max_bad = self.max_bad

        all_scorables: List[Scorable] = []
        scorables_targeted: List[Scorable] = []
        scorables_baseline: List[Scorable] = []

        total_kept = 0
        good_kept = 0
        bad_kept = 0

        for rec in all_records:
            # If we've hit all relevant limits, stop early.
            if max_total is not None and total_kept >= max_total:
                break
            if (
                (max_good is not None and good_kept >= max_good) and
                (max_bad is not None and bad_kept >= max_bad)
            ):
                # Both cohorts are full → nothing more to add
                break

            try:
                sc, is_correct = self._build_scorable_from_record(rec)
            except Exception as e:
                self.logger.log(
                    "Gsm8kLogLoaderBuildFailed",
                    {"error": str(e), "record_keys": list(rec.keys())},
                )
                continue

            # Apply per-cohort caps
            if is_correct:
                if max_good is not None and good_kept >= max_good:
                    continue
                scorables_targeted.append(sc)
                good_kept += 1
            else:
                if max_bad is not None and bad_kept >= max_bad:
                    continue
                scorables_baseline.append(sc)
                bad_kept += 1

            all_scorables.append(sc)
            total_kept += 1

        return all_scorables, scorables_targeted, scorables_baseline

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ignores context input (we're fully offline).

        Produces:
          - context[self.output_key]          = List[Scorable]
          - context["scorables_targeted"]     = List[Scorable]
          - context["scorables_baseline"]     = List[Scorable]
          - context["gsm8k_log_stats"]        = summary dict (for logging)
        """
        try:
            all_scorables, targeted, baseline = self._load_scorables_from_logs()

            n_all = len(all_scorables)
            n_tgt = len(targeted)
            n_base = len(baseline)
            acc = (n_tgt / n_all) if n_all > 0 else 0.0

            # Wire into context for downstream agents (VisiCalc, etc.)
            context[self.output_key] = all_scorables
            context["scorables_targeted"] = targeted
            context["scorables_baseline"] = baseline
            context["gsm8k_log_stats"] = {
                "total": n_all,
                "targeted": n_tgt,
                "baseline": n_base,
                "accuracy": acc,
                "num_log_files": len(self._discover_log_files()),
                "max_records": self.max_records,
                "max_good": self.max_good,
                "max_bad": self.max_bad,
            }

            self.logger.log(
                "Gsm8kLogLoaderSummary",
                {
                    "agent": self.name,
                    "total": n_all,
                    "targeted": n_tgt,
                    "baseline": n_base,
                    "accuracy": acc,
                    "max_records": self.max_records,
                    "max_good": self.max_good,
                    "max_bad": self.max_bad,
                },
            )

            return context

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"❌ Gsm8kLogLoaderAgent exception: {err_msg}")
            self.logger.log(
                "AgentFailed",
                {
                    "agent": self.name,
                    "error": err_msg,
                    "input_key": self.input_key,
                    "output_key": self.output_key,
                },
            )
            return context
