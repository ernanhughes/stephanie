# stephanie/memory/memcube_store.py
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from sqlalchemy import asc, desc

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.memcube import MemCubeORM
from stephanie.utils.hash_utils import hash_text

# ---------- helpers ----------------------------------------------------------


def _stable_int_id(parts: Iterable[str]) -> int:
    """
    Deterministic 31-bit positive int from a list of strings.
    Satisfies MemCubeORM.scorable_id: Integer (not string).
    """
    h = hash_text("||".join(parts))
    # take 12 hex chars (~48 bits) then mod into signed-32 range for safety
    return int(h[:12], 16) % 2_000_000_000


def _stable_digest(payload: Dict[str, Any]) -> str:
    """
    Stable SHA1 hex digest over a JSON-serializable payload with sorted keys.
    Converts non-primitive types where reasonable (lists/tuples/dicts/str/float/int/bool/None).
    """

    def _ser(x):
        if isinstance(x, dict):
            return {k: _ser(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple)):
            return [_ser(v) for v in x]
        return x

    data = json.dumps(
        _ser(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hash_text(data)


def _merge_extra(
    a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    out = dict(a or {})
    out.update(b or {})
    return out


_version_num_re = re.compile(r"(\d+)")


def _version_key(v: Optional[str]) -> Tuple[int, str]:
    """
    Sort key for version strings like 'v1', 'v12', '1.2.3'. Falls back to lexicographic tie-breaker.
    """
    if not v:
        return (0, "")
    nums = [int(m) for m in _version_num_re.findall(v)]
    return (nums[0] if nums else 0, v)


def _ensure_id(data: Dict[str, Any]) -> str:
    """
    Build a stable ID IF missing, based on (scorable_id, scorable_type, dimension, version, source, model).
    You can adjust the fields involved to match your governance policy.
    """
    if data.get("id"):
        return data["id"]

    base = {
        "scorable_id": data.get("scorable_id"),
        "scorable_type": data.get("scorable_type"),
        "dimension": data.get("dimension"),
        "version": data.get("version", "v1"),
        "source": data.get("source"),
        "model": data.get("model"),
    }
    digest = _stable_digest(base)
    # explicit: id includes version (your ORM comment mentions this)
    vid = f"{digest}:{base['version']}"
    data["id"] = vid
    if "version" not in data or data["version"] is None:
        data["version"] = "v1"
    return vid


def _merge_extra(
    existing: Optional[Dict[str, Any]], incoming: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Shallow merge for extra_data; incoming wins on key conflicts.
    """
    base = dict(existing or {})
    base.update(incoming or {})
    return base


# ---------- store ------------------------------------------------------------


class MemCubeStore(BaseSQLAlchemyStore):
    """
    Comprehensive store for MemCubeORM:
      - insert / upsert / bulk ops
      - filtered queries (by scorable, dimension, version, source, model, date, scores)
      - housekeeping (ttl pruning)
      - mutators (increment usage, refine results, touch)
      - exports (jsonl)
    """

    orm_model = MemCubeORM
    # most-recent first as a sensible default
    default_order_by = desc(MemCubeORM.created_at)

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "memcubes"

    # -------- core CRUD you already have from prior version (abridged) -------

    def insert(
        self,
        data: Dict[str, Any],
        *,
        if_exists: Literal["error", "skip", "update"] = "error",
        merge_extra: bool = True,
    ) -> MemCubeORM:
        def op(s):
            # Ensure id if missing
            if not data.get("id"):
                base = {
                    "scorable_id": data.get("scorable_id"),
                    "scorable_type": data.get("scorable_type"),
                    "dimension": data.get("dimension"),
                    "version": data.get("version", "v1"),
                    "source": data.get("source"),
                    "model": data.get("model"),
                }
                digest = _stable_digest(base)
                data["id"] = f"{digest}:{base['version']}"
                data.setdefault("version", "v1")

            existing = s.query(MemCubeORM).filter_by(id=data["id"]).first()
            if existing:
                if if_exists == "error":
                    raise ValueError(f"MemCube exists id={data['id']}")
                if if_exists == "skip":
                    return existing
                for k, v in data.items():
                    if k == "extra_data" and merge_extra:
                        existing.extra_data = _merge_extra(
                            existing.extra_data, v
                        )
                    elif k != "id":
                        setattr(existing, k, v)
                obj = existing
                action = "MemCubeUpdated"
            else:
                data.setdefault("priority", 5)
                data.setdefault("sensitivity", "public")
                data.setdefault("usage_count", 0)
                data.setdefault("extra_data", {})
                obj = MemCubeORM(**data)
                s.add(obj)
                action = "MemCubeInserted"

            if self.logger:
                self.logger.log(action, obj.to_dict())
            return obj

        return self._run(op)

    def upsert(
        self, data: Dict[str, Any], *, merge_extra: bool = True
    ) -> MemCubeORM:
        # composite uniqueness: (scorable_id, scorable_type, dimension, version)
        def op(s):
            existing = (
                s.query(MemCubeORM)
                .filter_by(
                    scorable_id=data["scorable_id"],
                    scorable_type=data["scorable_type"],
                    dimension=data.get("dimension"),
                    version=data["version"],
                )
                .first()
            )
            if existing:
                for k, v in data.items():
                    if k == "extra_data" and merge_extra:
                        existing.extra_data = _merge_extra(
                            existing.extra_data, v
                        )
                    else:
                        setattr(existing, k, v)
                obj = existing
                action = "MemCubeUpdated"
            else:
                if not data.get("id"):
                    base = {
                        "scorable_id": data["scorable_id"],
                        "scorable_type": data["scorable_type"],
                        "dimension": data.get("dimension"),
                        "version": data["version"],
                        "source": data.get("source"),
                        "model": data.get("model"),
                    }
                    data["id"] = f"{_stable_digest(base)}:{data['version']}"
                data.setdefault("priority", 5)
                data.setdefault("sensitivity", "public")
                data.setdefault("usage_count", 0)
                data.setdefault("extra_data", {})
                obj = MemCubeORM(**data)
                s.add(obj)
                action = "MemCubeInserted"

            if self.logger:
                self.logger.log(action, obj.to_dict())
            return obj

        return self._run(op)

    # -------- calibration API -------------------------------------------------

    def store_calibration(
        self,
        kind: str,  # e.g., "risk"
        payload: Dict[str, Any],  # expects at least {"domain": "...", ...}
        *,
        version: str = "v1",
        model: Optional[str] = None,
        source: str = "calibration",
        sensitivity: str = "public",
        ttl: Optional[int] = None,
        priority: int = 5,
        merge_extra: bool = True,
    ) -> MemCubeORM:
        """
        Persist a per-domain calibration record as a MemCube row.

        Mapping:
          scorable_type = "calibration"
          dimension     = <kind> (e.g., "risk")
          scorable_id   = stable int from (calibration, kind, domain)
          content       = JSON string of payload (human-readable)
          extra_data    = payload again (queryable JSON), plus tags
        """
        domain = str(payload.get("domain", "general"))
        scorable_id = _stable_int_id(["calibration", kind, domain])

        data = {
            "scorable_id": scorable_id,
            "scorable_type": "calibration",
            "content": json.dumps(payload, ensure_ascii=False),
            "dimension": kind,
            "original_score": None,  # unused here
            "refined_score": float(payload.get("ece"))
            if "ece" in payload
            else None,
            "refined_content": None,
            "version": version,
            "source": source,
            "model": model,
            "priority": priority,
            "sensitivity": sensitivity,
            "ttl": ttl,
            "usage_count": 0,
            # keep domain in JSONB for easy filtering later
            "extra_data": _merge_extra(
                {
                    "domain": domain,
                    "tags": ["calibration", kind],
                    "calibration": True,
                },
                payload,
            ),
        }
        return self.upsert(data, merge_extra=merge_extra)

    def get_calibration(
        self,
        kind: str,
        domain: str,
        *,
        version: Optional[str] = None,
        newest_first: bool = True,
    ) -> Optional[MemCubeORM]:
        """
        Retrieve one calibration record for a kind+domain.
        If version is None, returns newest by created_at.
        """

        def op(s):
            q = s.query(MemCubeORM).filter(
                MemCubeORM.scorable_type == "calibration",
                MemCubeORM.dimension == kind,
                MemCubeORM.extra_data["domain"].astext == str(domain),
            )
            if version:
                q = q.filter(MemCubeORM.version == version)
            q = q.order_by(
                desc(MemCubeORM.created_at)
                if newest_first
                else asc(MemCubeORM.created_at)
            )
            return q.first()

        return self._run(op)

    def list_calibrations(
        self,
        kind: str,
        *,
        domains: Optional[List[str]] = None,
        version: Optional[str] = None,
        limit: int = 1000,
    ) -> List[MemCubeORM]:
        def op(s):
            q = s.query(MemCubeORM).filter(
                MemCubeORM.scorable_type == "calibration",
                MemCubeORM.dimension == kind,
            )
            if domains:
                q = q.filter(
                    MemCubeORM.extra_data["domain"].astext.in_(
                        [str(d) for d in domains]
                    )
                )
            if version:
                q = q.filter(MemCubeORM.version == version)
            return q.order_by(desc(MemCubeORM.created_at)).limit(limit).all()

        return self._run(op)

    def bulk_insert(
        self,
        items: Iterable[Dict[str, Any]],
        *,
        on_conflict: Literal["skip", "update"] = "skip",
        merge_extra: bool = True,
    ) -> List[MemCubeORM]:
        """
        Efficient bulk add. Uses id match for conflict behavior.
        """

        def op(s):
            out = []
            for item in items:
                _ensure_id(item)
                existing = s.query(MemCubeORM).filter_by(id=item["id"]).first()
                if existing:
                    if on_conflict == "update":
                        for k, v in item.items():
                            if k == "extra_data" and merge_extra:
                                setattr(
                                    existing,
                                    "extra_data",
                                    _merge_extra(existing.extra_data, v),
                                )
                            elif k != "id":
                                setattr(existing, k, v)
                        out.append(existing)
                    else:
                        out.append(existing)
                    continue

                item.setdefault("priority", 5)
                item.setdefault("sensitivity", "public")
                item.setdefault("usage_count", 0)
                item.setdefault("extra_data", {})
                obj = MemCubeORM(**item)
                s.add(obj)
                out.append(obj)
            return out

        return self._run(op)

    # --------------------
    # GETTERS
    # --------------------
    def get_by_id(self, cube_id: str) -> Optional[MemCubeORM]:
        def op(s):
            return s.query(MemCubeORM).filter_by(id=cube_id).first()

        return self._run(op)

    def get_by_scorable(
        self,
        scorable_id: int,
        *,
        scorable_type: Optional[str] = None,
        dimension: Optional[str] = None,
        limit: int = 100,
        newest_first: bool = True,
    ) -> List[MemCubeORM]:
        def op(s):
            q = s.query(MemCubeORM).filter(
                MemCubeORM.scorable_id == scorable_id
            )
            if scorable_type:
                q = q.filter(MemCubeORM.scorable_type == scorable_type)
            if dimension:
                q = q.filter(MemCubeORM.dimension == dimension)
            q = q.order_by(
                desc(MemCubeORM.created_at)
                if newest_first
                else asc(MemCubeORM.created_at)
            )
            return q.limit(limit).all()

        return self._run(op)

    def get_latest(
        self,
        scorable_id: int,
        *,
        scorable_type: str,
        dimension: Optional[str] = None,
    ) -> Optional[MemCubeORM]:
        """
        Latest by created_at (robust across arbitrary version strings).
        """

        def op(s):
            q = s.query(MemCubeORM).filter_by(
                scorable_id=scorable_id, scorable_type=scorable_type
            )
            if dimension:
                q = q.filter_by(dimension=dimension)
            return q.order_by(desc(MemCubeORM.created_at)).first()

        return self._run(op)

    def get_by_composite(
        self,
        *,
        scorable_id: int,
        scorable_type: str,
        dimension: Optional[str],
        version: str,
    ) -> Optional[MemCubeORM]:
        def op(s):
            return (
                s.query(MemCubeORM)
                .filter_by(
                    scorable_id=scorable_id,
                    scorable_type=scorable_type,
                    dimension=dimension,
                    version=version,
                )
                .first()
            )

        return self._run(op)

    # --------------------
    # SEARCH / FILTERS
    # --------------------
    def search(
        self,
        *,
        scorable_ids: Optional[List[int]] = None,
        scorable_type: Optional[str] = None,
        dimensions: Optional[List[str]] = None,
        version: Optional[str] = None,
        source: Optional[str] = None,
        model: Optional[str] = None,
        sensitivity: Optional[str] = None,
        min_original_score: Optional[float] = None,
        min_refined_score: Optional[float] = None,
        created_from: Optional[datetime] = None,
        created_to: Optional[datetime] = None,
        order_desc: bool = True,
        limit: int = 200,
    ) -> List[MemCubeORM]:
        def op(s):
            q = s.query(MemCubeORM)

            if scorable_ids:
                q = q.filter(MemCubeORM.scorable_id.in_(scorable_ids))
            if scorable_type:
                q = q.filter(MemCubeORM.scorable_type == scorable_type)
            if dimensions:
                q = q.filter(MemCubeORM.dimension.in_(dimensions))
            if version:
                q = q.filter(MemCubeORM.version == version)
            if source:
                q = q.filter(MemCubeORM.source == source)
            if model:
                q = q.filter(MemCubeORM.model == model)
            if sensitivity:
                q = q.filter(MemCubeORM.sensitivity == sensitivity)
            if min_original_score is not None:
                q = q.filter(MemCubeORM.original_score >= min_original_score)
            if min_refined_score is not None:
                q = q.filter(MemCubeORM.refined_score >= min_refined_score)
            if created_from:
                q = q.filter(MemCubeORM.created_at >= created_from)
            if created_to:
                q = q.filter(MemCubeORM.created_at <= created_to)

            q = q.order_by(
                desc(MemCubeORM.created_at)
                if order_desc
                else asc(MemCubeORM.created_at)
            )
            return q.limit(limit).all()

        return self._run(op)

    # --------------------
    # MUTATORS
    # --------------------
    def increment_usage(
        self, cube_id: str, by: int = 1
    ) -> Optional[MemCubeORM]:
        def op(s):
            obj = s.query(MemCubeORM).filter_by(id=cube_id).first()
            if not obj:
                return None
            obj.usage_count = int(obj.usage_count or 0) + by
            if self.logger:
                self.logger.log(
                    "MemCubeUsageIncremented",
                    {"id": cube_id, "by": by, "new": obj.usage_count},
                )
            return obj

        return self._run(op)

    def set_refined_result(
        self,
        cube_id: str,
        *,
        refined_score: Optional[float] = None,
        refined_content: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        merge_extra: bool = True,
    ) -> Optional[MemCubeORM]:
        def op(s):
            obj = s.query(MemCubeORM).filter_by(id=cube_id).first()
            if not obj:
                return None
            if refined_score is not None:
                obj.refined_score = refined_score
            if refined_content is not None:
                obj.refined_content = refined_content
            if extra_data:
                obj.extra_data = (
                    _merge_extra(obj.extra_data, extra_data)
                    if merge_extra
                    else (extra_data or {})
                )
            if self.logger:
                self.logger.log("MemCubeRefined", obj.to_dict())
            return obj

        return self._run(op)

    def touch(self, cube_id: str) -> Optional[MemCubeORM]:
        """
        Forces last_modified to update by toggling a harmless field (usage_count no-op).
        """
        return self.increment_usage(cube_id, by=0)

    def delete_by_id(self, cube_id: str) -> bool:
        def op(s):
            obj = s.query(MemCubeORM).filter_by(id=cube_id).first()
            if not obj:
                return False
            s.delete(obj)
            if self.logger:
                self.logger.log("MemCubeDeleted", {"id": cube_id})
            return True

        return self._run(op)

    # --------------------
    # TTL / HOUSEKEEPING
    # --------------------
    def list_expired(
        self, *, now: Optional[datetime] = None, limit: int = 1000
    ) -> List[MemCubeORM]:
        """
        Returns cubes where ttl is set and (created_at + ttl days) < now.
        """
        now = now or datetime.utcnow()

        def op(s):
            q = s.query(MemCubeORM).filter(MemCubeORM.ttl.isnot(None))
            # Project in Python to avoid dialect-specific INTERVAL arithmetic
            res = q.order_by(asc(MemCubeORM.created_at)).limit(limit).all()
            expired = []
            for r in res:
                try:
                    if r.created_at and r.ttl is not None:
                        if (r.created_at + timedelta(days=int(r.ttl))) < now:
                            expired.append(r)
                except Exception:
                    # be robust to weird data
                    continue
            return expired

        return self._run(op)

    def prune_expired(
        self,
        *,
        delete: bool = True,
        now: Optional[datetime] = None,
        limit: int = 1000,
    ) -> int:
        """
        Deletes (or returns count if delete=False) expired items based on TTL.
        """
        expired = self.list_expired(now=now, limit=limit)
        if not delete:
            return len(expired)

        def op(s):
            count = 0
            for r in expired:
                s.delete(r)
                count += 1
            if self.logger and count:
                self.logger.log("MemCubeTTLPruned", {"count": count})
            return count

        return self._run(op)

    # --------------------
    # EXPORTS
    # --------------------
    def export_jsonl(
        self,
        path: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10000,
    ) -> int:
        """
        Dumps selected MemCubes to JSONL for offline analysis.
        """
        filters = filters or {}
        rows = self.search(
            scorable_ids=filters.get("scorable_ids"),
            scorable_type=filters.get("scorable_type"),
            dimensions=filters.get("dimensions"),
            version=filters.get("version"),
            source=filters.get("source"),
            model=filters.get("model"),
            sensitivity=filters.get("sensitivity"),
            min_original_score=filters.get("min_original_score"),
            min_refined_score=filters.get("min_refined_score"),
            created_from=filters.get("created_from"),
            created_to=filters.get("created_to"),
            order_desc=True,
            limit=limit,
        )
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(
                    json.dumps(
                        r.to_dict(include_extra=True), ensure_ascii=False
                    )
                    + "\n"
                )
                n += 1
        if self.logger:
            self.logger.log("MemCubeExported", {"path": path, "count": n})
        return n
