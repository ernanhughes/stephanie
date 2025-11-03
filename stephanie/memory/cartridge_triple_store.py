# stephanie/memory/cartridge_triple_store.py
from __future__ import annotations

from sqlalchemy import case, func

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.cartridge_domain import CartridgeDomainORM
from stephanie.models.cartridge_triple import CartridgeTripleORM
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM
from stephanie.scoring.scorable import ScorableType


class CartridgeTripleStore(BaseSQLAlchemyStore):
    orm_model = CartridgeTripleORM
    default_order_by = "id"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "cartridge_triples"

    # ------------------------------
    # Core APIs
    # ------------------------------

    def insert(self, data: dict) -> CartridgeTripleORM:
        def op(s):
            
                existing = (
                    s.query(CartridgeTripleORM)
                    .filter_by(
                        cartridge_id=data["cartridge_id"],
                        subject=data["subject"],
                        predicate=data["predicate"],
                        object=data["object"],
                    )
                    .first()
                )

                if existing:
                    if "confidence" in data and existing.confidence != data["confidence"]:
                        existing.confidence = data["confidence"]
                    return existing

                new_triple = CartridgeTripleORM(**data)
                s.add(new_triple)
                return new_triple

        result = self._run(op)
        if self.logger:
            self.logger.log("TripleInsertedOrUpdated", data)
        return result

    def get_by_id(self, triple_id: int) -> CartridgeTripleORM | None:
        def op(s):
            
                return s.query(CartridgeTripleORM).filter_by(id=triple_id).first()
        return self._run(op)

    def get_triples(self, cartridge_id: int) -> list[CartridgeTripleORM]:
        def op(s):
            
                return s.query(CartridgeTripleORM).filter_by(cartridge_id=cartridge_id).all()
        return self._run(op)

    def delete_triples(self, cartridge_id: int):
        def op(s):
            
                s.query(CartridgeTripleORM).filter_by(cartridge_id=cartridge_id).delete()
        self._run(op)
        if self.logger:
            self.logger.log("TriplesDeleted", {"cartridge_id": cartridge_id})

    def has_triples(self, cartridge_id: int) -> bool:
        def op(s):
            
                return (
                    s.query(CartridgeTripleORM)
                    .filter_by(cartridge_id=cartridge_id)
                    .first()
                    is not None
                )
        return self._run(op)

    def set_triples(self, cartridge_id: int, triples: list[tuple[str, str, str, float]]):
        def op(s):
            
                # clear existing
                s.query(CartridgeTripleORM).filter_by(cartridge_id=cartridge_id).delete()
                # re-add
                for subj, pred, obj, conf in triples:
                    s.add(CartridgeTripleORM(
                        cartridge_id=cartridge_id,
                        subject=subj,
                        predicate=pred,
                        object=obj,
                        confidence=float(conf),
                    ))
        self._run(op)
        if self.logger:
            self.logger.log("TriplesSet", {"cartridge_id": cartridge_id, "count": len(triples)})

    # ------------------------------
    # Retrieval Methods
    # ------------------------------

    def retrieve_top_triplets_by_score(
        self, goal_id: int, score_weights: dict[str, float], top_k: int = 20
    ) -> list[CartridgeTripleORM]:
        def op(s):
            
                subq = (
                    s.query(
                        EvaluationORM.scorable_id.label("triplet_id"),
                        func.sum(
                            case(
                                *[
                                    (ScoreORM.dimension == dim, ScoreORM.score * weight)
                                    for dim, weight in score_weights.items()
                                ],
                                else_=0,
                            )
                        ).label("weighted_score"),
                    )
                    .join(ScoreORM, ScoreORM.evaluation_id == EvaluationORM.id)
                    .filter(
                        EvaluationORM.goal_id == goal_id,
                        EvaluationORM.scorable_type == ScorableType.TRIPLE,
                    )
                    .group_by(EvaluationORM.scorable_id)
                    .subquery()
                )

                return (
                    s.query(CartridgeTripleORM)
                    .join(subq, subq.c.triplet_id == CartridgeTripleORM.id)
                    .order_by(subq.c.weighted_score.desc())
                    .limit(top_k)
                    .all()
                )

        triplets = self._run(op)
        if self.logger:
            self.logger.log("TripletsRetrievedByScore", {
                "goal_id": goal_id,
                "score_weights": score_weights,
                "count": len(triplets),
            })
        return triplets

    def retrieve_top_triplets_by_domain(
        self, goal_text: str, domain_classifier, top_k: int = 20, min_score: float = 0.6
    ) -> list[CartridgeTripleORM]:
        def op(s):
            
                goal_domains = domain_classifier.classify(goal_text, top_k=5, min_score=min_score)
                domain_names = [d[0] for d in goal_domains]

                return (
                    s.query(CartridgeTripleORM)
                    .join(CartridgeDomainORM, CartridgeTripleORM.cartridge_id == CartridgeDomainORM.cartridge_id)
                    .filter(CartridgeDomainORM.domain.in_(domain_names))
                    .distinct()
                    .limit(top_k)
                    .all()
                )

        triplets = self._run(op)
        if self.logger:
            self.logger.log("TripletsRetrievedByDomain", {
                "goal_text": goal_text,
                "domains": [d[0] for d in domain_classifier.classify(goal_text, top_k=5, min_score=min_score)],
                "count": len(triplets),
            })
        return triplets
