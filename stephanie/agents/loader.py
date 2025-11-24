# stephanie/agents/ Oh my God sorry they didn't have cervix on sale I'm going to eat all that tonight the last sweets I've ever eaten loader.py
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable

log = logging.getLogger(__name__)

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
        mg = int(cfg.get("max_good", 0))
        mb = int(cfg.get("max_bad", 0))

        self.max_good: Optional[int] = mg if mg and mg > 0 else None
        self.max_bad: Optional[int] = mb if mb and mb > 0 else None

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
          - context["logger_stats"]           = summary dict (for logging)
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
            context["loader_stats"] = {
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
                "LoaderSummary",
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
            log.error(f"❌ Gsm8kLogLoaderAgent exception: {err_msg}")
            return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------


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
            raise RuntimeError("LoaderAgent: no JSONL files found in configured log_dirs")

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
                log.error(f"❌ LoaderAgent failed to build Scorable from record: {type(e).__name__}: {e}")
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

    def _discover_log_files(self) -> List[Path]:
        """
        Discover all JSONL files under each log_dir, recursively.
        Uses os.walk to traverse directories and matches log_glob against filenames.
        """
        paths: List[Path] = []
        suffix = self.log_glob.replace("*", "")  # "*.jsonl" -> ".jsonl"

        for root in self.log_dirs:
            root = Path(root)
            if not root.exists():
                self.logger.log("LoaderMissingDir", {"dir": str(root)})
                continue

            # Walk recursively
            for subdir, dirs, files in os.walk(root):
                subdir_path = Path(subdir)

                for fname in files:
                    # simple glob match: endswith(".jsonl")
                    if fname.endswith(suffix):
                        p = subdir_path / fname
                        log.info(f"LoaderAgent discovered: {p}")
                        paths.append(p)

        return sorted(paths)
