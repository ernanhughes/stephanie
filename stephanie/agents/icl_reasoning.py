from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import case, func

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.domain_classifier import DomainClassifier
from stephanie.models.cartridge_domain import CartridgeDomainORM
from stephanie.models.cartridge_triple import CartridgeTripleORM
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM
from stephanie.models.theorem import CartridgeORM, TheoremORM
from stephanie.scoring.scorable_factory import TargetType


class ICLReasoningAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.top_k_triplets = cfg.get("top_k_triplets", 5)
        self.min_score_threshold = cfg.get("min_triplet_score", 0.6)
        self.use_embeddings = cfg.get("use_triplet_embeddings", False)
        self.score_weights = cfg.get(
            "score_weights",
            {
                "usefulness": 0.4,
                "clarity": 0.2,
                "specificity": 0.2,
                "confidence": 0.1,
                "novelty": 0.1,
            },
        )
        self.prompt_template = cfg.get(
            "icl_prompt_template", "icl_reasoning_prompt.txt"
        )
        self.domain_classifier = DomainClassifier(
            memory,
            logger,
            cfg.get("domain_seed_config_path", "config/domain/seeds.yaml"),
        )


    async def run(self, context: dict) -> dict:
        goal = context.get("goal")

        top_triplets = self.retrieve_top_triplets(goal)
        top_theorems = self.retrieve_top_theorems(goal)

        learned_facts = self.format_triplets_as_facts(top_triplets)
        learned_theorems = self.format_theorems(top_theorems)

        merged_context = {
            "goal": goal,
            "learned_facts": learned_facts,
            "learned_theorems": learned_theorems,
            **context,
        }

        prompt = self.prompt_loader.load_prompt(self.cfg, merged_context)
        response = self.call_llm(prompt, context=context)

        self.logger.log(
            "ICLPromptResponse",
            {"goal": goal["goal_text"], "prompt": prompt, "response": response},
        )

        context[self.output_key] = response
        return context

    def retrieve_top_theorems(self, goal: dict, top_k=3) -> list[TheoremORM]:
        session = self.memory.session
        goal_vec = self.memory.embedding.get_or_create(goal["goal_text"])
        all_theorems = session.query(TheoremORM).all()

        scored_theorems = sorted(
            all_theorems,
            key=lambda thm: self.similarity(goal_vec, thm.statement),
            reverse=True
        )

        return scored_theorems[:top_k]

    def format_theorems(self, theorems: list[TheoremORM]) -> str:
        lines = []
        for theorem in theorems:
            lines.append(f"- {theorem.statement} (Proof: {theorem.proof or 'N/A'})")
        return "\n".join(lines)

    def retrieve_top_triplets(self, goal: dict) -> list[CartridgeTripleORM]:
        session = self.memory.session
        goal_text = goal["goal_text"]
        goal_domains = self.domain_classifier.classify(goal_text)
        goal_domain_names = [d[0] for d in goal_domains]

        query = (
           session.query(CartridgeTripleORM)
            .join(CartridgeTripleORM.cartridge)
            .join(CartridgeORM.domains_rel)
            .filter(CartridgeDomainORM.domain.in_(goal_domain_names))
        )        # Create a mapping of dimensions to their weights

        triplets = query.distinct().all()
        
        self.logger.log("TripletsRetrievedByDomain", {
                "goal": goal_text,
                "domains": goal_domain_names,
                "triplet_count": len(triplets),
            })

        return triplets[:self.top_k_triplets] if self.top_k_triplets else triplets

    def retrieve_top_triplets_by_score(self, goal: dict) -> list[CartridgeTripleORM]:
        session = self.memory.session
        goal_id = goal["id"]
        weight_map = self.score_weights  # e.g., {"usefulness": 0.5, "clarity": 0.3}

        # Subquery to compute weighted scores per triplet, grouped by Evaluation.target_id
        subq = (
            session.query(
                EvaluationORM.target_id.label("triplet_id"),
                func.sum(
                    case(
                        *[
                            (ScoreORM.dimension == dim, ScoreORM.score * weight)
                            for dim, weight in weight_map.items()
                        ],
                        else_=0,
                    )
                ).label("weighted_score"),
            )
            .join(ScoreORM, ScoreORM.evaluation_id == EvaluationORM.id)
            .filter(
                EvaluationORM.goal_id == goal_id,
                EvaluationORM.target_type == TargetType.TRIPLE,
            )
            .group_by(EvaluationORM.target_id)
            .subquery()
        )

        # Join back with CartridgeTripleORM
        query = (
            session.query(CartridgeTripleORM)
            .join(subq, subq.c.triplet_id == CartridgeTripleORM.id)
            .order_by(subq.c.weighted_score.desc())
        )

        triplets = query.all()

        # Optional: embedding filtering
        if self.use_embeddings:
            goal_vec = self.memory.embedding.get_or_create(goal["goal_text"])
            triplets = [
                t
                for t in triplets
                if self.similarity(goal_vec, t) >= self.min_score_threshold
            ]

        return triplets[: self.top_k_triplets]


    def similarity(self, goal_vec, text: str) -> float:
        vec = self.memory.embedding.get_or_create(text)
        return float(cosine_similarity([goal_vec], [vec])[0][0])

    def build_prompt(self, goal_text: str, facts: str) -> str:
        prompt_context = {"goal": goal_text, "facts": facts}
        return self.prompt_loader.from_file(
            self.prompt_template, self.cfg, prompt_context
        )

    def format_triplets_as_facts(self, triplets: list) -> str:
        """
        Accepts a list of CartridgeTripleORM objects or dicts and returns a
        bullet list of formatted triplet strings like (subject, predicate, object).
        """
        lines = []
        for triplet in triplets:
            if isinstance(triplet, dict):
                subj = triplet.get("subject")
                pred = triplet.get("predicate")
                obj = triplet.get("object")
            else:
                subj = getattr(triplet, "subject", None)
                pred = getattr(triplet, "predicate", None)
                obj = getattr(triplet, "object", None)

            if subj and pred and obj:
                lines.append(f"({subj}, {pred}, {obj})")
        return "\n".join(lines)
