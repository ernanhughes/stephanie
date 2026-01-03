# stephanie/memory/pairrm_ranking_store.py
from __future__ import annotations

from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.pairrm_ranking import PairRMRankingORM


class PairRMRankingStore(BaseSQLAlchemyStore):
    orm_model = PairRMRankingORM
    name = "pairrm_rankings"
    id_attr = "id"

    def upsert(self, data: Dict[str, Any]) -> PairRMRankingORM:
        """
        Upsert using the natural key expected by PairRMRankingTool.

        Required keys:
          scorable_type, scorable_id, tool_name, candidate_id, rank, score
        Optional:
          run_id, candidate_index, meta
        """
        required = ["scorable_type", "scorable_id", "tool_name", "candidate_id", "rank", "score"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"PairRMRankingStore.upsert missing required keys: {missing}")

        def op(s: Session) -> PairRMRankingORM:
            q = (
                s.query(PairRMRankingORM)
                .filter(PairRMRankingORM.scorable_type == data["scorable_type"])
                .filter(PairRMRankingORM.scorable_id == int(data["scorable_id"]))
                .filter(PairRMRankingORM.tool_name == data["tool_name"])
                .filter(PairRMRankingORM.candidate_id == data["candidate_id"])
            )

            # run_id is part of the natural key, may be NULL
            if data.get("run_id") is None:
                q = q.filter(PairRMRankingORM.run_id.is_(None))
            else:
                q = q.filter(PairRMRankingORM.run_id == data["run_id"])

            obj = q.one_or_none()
            if obj is None:
                obj = PairRMRankingORM(
                    scorable_type=data["scorable_type"],
                    scorable_id=int(data["scorable_id"]),
                    tool_name=data["tool_name"],
                    run_id=data.get("run_id"),
                    candidate_id=data["candidate_id"],
                )

            obj.candidate_index = int(data.get("candidate_index", obj.candidate_index or 0))
            obj.rank = int(data["rank"])
            obj.score = float(data["score"])

            # keep meta flexible
            meta = data.get("meta")
            if meta is not None:
                obj.meta = meta

            s.add(obj)
            s.flush()
            return obj

        return self._run(op)
