# stephanie/memory/memory_genotype_store.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import desc
from sqlalchemy.orm import Session

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.memory_genotype import (MemoryEvolutionRunORM,
                                           MemoryGenotypeORM)


class MemoryGenotypeStore(BaseSQLAlchemyStore):
    orm_model = MemoryGenotypeORM
    name = "memory_genotypes"
    id_attr = "id"
    default_order_by = desc(MemoryGenotypeORM.created_at)

    def get_by_name(self, name: str) -> Optional[MemoryGenotypeORM]:
        def op(s: Session) -> Optional[MemoryGenotypeORM]:
            return s.query(MemoryGenotypeORM).filter(MemoryGenotypeORM.name == name).one_or_none()

        return self._run(op)

    def upsert(self, data: Dict[str, Any]) -> MemoryGenotypeORM:
        """
        Upsert by (id) if provided else by (name).
        """
        def op(s: Session) -> MemoryGenotypeORM:
            obj = None
            if data.get("id") is not None:
                obj = s.query(MemoryGenotypeORM).filter(MemoryGenotypeORM.id == data["id"]).one_or_none()
            if obj is None and data.get("name"):
                obj = s.query(MemoryGenotypeORM).filter(MemoryGenotypeORM.name == data["name"]).one_or_none()

            if obj is None:
                obj = MemoryGenotypeORM()

            # Mutable updates
            for k in [
                "parent_id",
                "generation",
                "name",
                "description",
                "spec",
                "tags",
            ]:
                if k in data and data[k] is not None:
                    setattr(obj, k, data[k])

            if "is_active" in data and data["is_active"] is not None:
                obj.is_active = int(data["is_active"])

            s.add(obj)
            s.flush()
            return obj

        return self._run(op)

    def set_active(self, genotype_id: int) -> None:
        """
        Convenience: deactivate all, activate one.
        """
        def op(s: Session) -> None:
            s.query(MemoryGenotypeORM).update({MemoryGenotypeORM.is_active: 0})
            s.query(MemoryGenotypeORM).filter(MemoryGenotypeORM.id == genotype_id).update(
                {MemoryGenotypeORM.is_active: 1}
            )

        self._run(op)

    def get_active(self) -> Optional[MemoryGenotypeORM]:
        def op(s: Session) -> Optional[MemoryGenotypeORM]:
            return (
                s.query(MemoryGenotypeORM)
                .filter(MemoryGenotypeORM.is_active == 1)
                .order_by(MemoryGenotypeORM.updated_at.desc())
                .first()
            )

        return self._run(op)

    def update_fitness_stats(self, genotype_id: int, fitness: float) -> None:
        """
        Rolling mean update (simple, cheap).
        """
        def op(s: Session) -> None:
            g = s.query(MemoryGenotypeORM).filter(MemoryGenotypeORM.id == genotype_id).one()
            n = int(g.fitness_count or 0)
            m = g.fitness_mean
            if m is None or n <= 0:
                g.fitness_mean = float(fitness)
                g.fitness_count = 1
            else:
                g.fitness_mean = (m * n + float(fitness)) / (n + 1)
                g.fitness_count = n + 1
            s.add(g)

        self._run(op)


class MemoryEvolutionRunStore(BaseSQLAlchemyStore):
    orm_model = MemoryEvolutionRunORM
    name = "memory_evolution_runs"
    id_attr = "id"
    default_order_by = desc(MemoryEvolutionRunORM.created_at)

    def create_run(self, *, genotype_id: int, **kwargs: Any) -> MemoryEvolutionRunORM:
        """
        Create a run row for one genotype evaluation.

        kwargs supported (all optional):
          run_id, goal_id, suite, domain, model_name,
          fitness, perf, cost, delay, epistemic, stability, memory_eff,
          metrics, diagnosis, notes
        """
        def op(s: Session) -> MemoryEvolutionRunORM:
            r = MemoryEvolutionRunORM(genotype_id=genotype_id, **kwargs)
            s.add(r)
            s.flush()
            return r

        return self._run(op)

    def list_runs(
        self,
        *,
        genotype_id: Optional[int] = None,
        goal_id: Optional[str] = None,
        suite: Optional[str] = None,
        limit: int = 200,
    ) -> List[MemoryEvolutionRunORM]:
        def op(s: Session) -> List[MemoryEvolutionRunORM]:
            q = s.query(MemoryEvolutionRunORM)
            if genotype_id is not None:
                q = q.filter(MemoryEvolutionRunORM.genotype_id == genotype_id)
            if goal_id is not None:
                q = q.filter(MemoryEvolutionRunORM.goal_id == goal_id)
            if suite is not None:
                q = q.filter(MemoryEvolutionRunORM.suite == suite)
            return q.order_by(MemoryEvolutionRunORM.created_at.desc()).limit(limit).all()

        return self._run(op)

    def latest_run(self, *, genotype_id: int) -> Optional[MemoryEvolutionRunORM]:
        def op(s: Session) -> Optional[MemoryEvolutionRunORM]:
            return (
                s.query(MemoryEvolutionRunORM)
                .filter(MemoryEvolutionRunORM.genotype_id == genotype_id)
                .order_by(MemoryEvolutionRunORM.created_at.desc())
                .first()
            )

        return self._run(op)
