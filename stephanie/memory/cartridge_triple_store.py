# stephanie/memory/cartridge_triple_store.py

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.cartridge_domain import CartridgeDomainORM
from stephanie.models.cartridge_triple import CartridgeTripleORM
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM
from stephanie.scoring.scorable_factory import TargetType


class CartridgeTripleStore(BaseSQLAlchemyStore):
    orm_model = CartridgeTripleORM
    default_order_by = CartridgeTripleORM.id.desc()
    
    def __init__(self, session: Session, logger=None):
        super().__init__(session, logger)
        self.name = "cartridge_triples"

    def name(self) -> str:
        return self.name

    def insert(self, data: dict) -> CartridgeTripleORM:
        """
        Insert or update a triple for a cartridge manually.
        Expected dict keys: cartridge_id, subject, predicate, object, (optional) confidence
        """
        query = self.session.query(CartridgeTripleORM).filter_by(
            cartridge_id=data["cartridge_id"],
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
        )

        existing = query.first()

        if existing:
            updated = False
            if "confidence" in data and existing.confidence != data["confidence"]:
                existing.confidence = data["confidence"]
                updated = True
            if updated:
                self.session.commit()
                if self.logger:
                    self.logger.log("TripleUpdated", data)
            return existing
        else:
            new_triple = CartridgeTripleORM(**data)
            self.session.add(new_triple)
            self.session.commit()
            if self.logger:
                self.logger.log("TripleInserted", data)
            return new_triple

    def get_by_id(self, triple_id: int) -> CartridgeTripleORM:
        return self.session.query(CartridgeTripleORM).get(triple_id)

    def get_triples(self, cartridge_id: int) -> list[CartridgeTripleORM]:
        return (
            self.session.query(CartridgeTripleORM)
            .filter_by(cartridge_id=cartridge_id)
            .all()
        )

    def delete_triples(self, cartridge_id: int):
        self.session.query(CartridgeTripleORM).filter_by(
            cartridge_id=cartridge_id
        ).delete()
        self.session.commit()
        if self.logger:
            self.logger.log("TriplesDeleted", {"cartridge_id": cartridge_id})

    def has_triples(self, cartridge_id: int) -> bool:
        return (
            self.session.query(CartridgeTripleORM)
            .filter_by(cartridge_id=cartridge_id)
            .first()
            is not None
        )

    def set_triples(
        self, cartridge_id: int, triples: list[tuple[str, str, str, float]]
    ):
        """Clear and re-add triples for a cartridge."""
        self.delete_triples(cartridge_id)
        for subj, pred, obj, conf in triples:
            self.insert(
                {
                    "cartridge_id": cartridge_id,
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "confidence": float(conf),
                }
            )

    # ------------------------------
    # Retrieval Methods
    # ------------------------------

    def retrieve_top_triplets_by_score(
        self, goal_id: int, score_weights: dict[str, float], top_k: int = 20
    ) -> list[CartridgeTripleORM]:
        """
        Retrieve top triplets for a given goal, ranked by weighted score.
        score_weights: dict of {dimension: weight} for aggregation.
        """
        subq = (
            self.session.query(
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
                EvaluationORM.scorable_type == TargetType.TRIPLE,
            )
            .group_by(EvaluationORM.scorable_id)
            .subquery()
        )

        query = (
            self.session.query(CartridgeTripleORM)
            .join(subq, subq.c.triplet_id == CartridgeTripleORM.id)
            .order_by(subq.c.weighted_score.desc())
        )

        triplets = query.limit(top_k).all()

        if self.logger:
            self.logger.log("TripletsRetrievedByScore", {
                "goal_id": goal_id,
                "score_weights": score_weights,
                "count": len(triplets)
            })

        return triplets

    def retrieve_top_triplets_by_domain(
        self, goal_text: str, domain_classifier, top_k: int = 20, min_score: float = 0.6
    ) -> list[CartridgeTripleORM]:
        """
        Retrieve triplets relevant to the domains of a goal.
        Uses the domain_classifier to infer domains from goal_text.
        """
        goal_domains = domain_classifier.classify(goal_text, top_k=5, min_score=min_score)
        domain_names = [d[0] for d in goal_domains]

        query = (
            self.session.query(CartridgeTripleORM)
            .join(CartridgeDomainORM, CartridgeTripleORM.cartridge_id == CartridgeDomainORM.cartridge_id)
            .filter(CartridgeDomainORM.domain.in_(domain_names))
        )

        triplets = query.distinct().limit(top_k).all()

        if self.logger:
            self.logger.log("TripletsRetrievedByDomain", {
                "goal_text": goal_text,
                "domains": domain_names,
                "count": len(triplets)
            })

        return triplets
