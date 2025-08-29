# stephanie/agents/icl_reasoning.py
from sklearn.metrics.pairwise import cosine_similarity

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.models.cartridge_domain import CartridgeDomainORM
from stephanie.models.cartridge_triple import CartridgeTripleORM
from stephanie.models.theorem import TheoremORM


class ICLReasoningAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.top_k_triplets = cfg.get("top_k_triplets", 5)
        self.min_value_threshold = cfg.get("min_triplet_score", 0.6)
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
        self.domain_classifier = ScorableClassifier(
            memory,
            logger,
            cfg.get("domain_seed_config_path", "config/domain/seeds.yaml"),
        )

    async def run(self, context: dict) -> dict:
        goal = context.get("goal")

        self.report({
            "event": "start",
            "step": "ICLReasoning",
            "goal": goal.get("goal_text", "")
        })

        # --- Step 1: Gather inputs
        top_triplets = self.retrieve_top_triplets(goal)
        top_theorems = self.retrieve_top_theorems(goal)

        self.report({
            "event": "inputs_collected",
            "step": "ICLReasoning",
            "triplet_count": len(top_triplets),
            "theorem_count": len(top_theorems),
            "triplet_examples": [f"({t.subject}, {t.predicate}, {t.object})" for t in top_triplets[:3]],
            "theorem_examples": [t.statement[:100] for t in top_theorems[:2]]
        })

        # --- Step 2: Format facts + theorems
        learned_facts = self.format_triplets_as_facts(top_triplets)
        learned_theorems = self.format_theorems(top_theorems)

        merged_context = {
            "goal": goal,
            "learned_facts": learned_facts,
            "learned_theorems": learned_theorems,
            **context,
        }

        # --- Step 3: Generate reasoning
        prompt = self.prompt_loader.load_prompt(self.cfg, merged_context)
        response = self.call_llm(prompt, context=context)

        self.report({
            "event": "reasoning_generated",
            "step": "ICLReasoning",
            "prompt_excerpt": prompt[:300],
            "response_excerpt": response[:300],
        })

        context[self.output_key] = response

        self.report({
            "event": "completed",
            "step": "ICLReasoning",
            "output_length": len(response),
        })
        return context

    def retrieve_top_theorems(self, goal: dict, top_k=3) -> list[TheoremORM]:
        session = self.memory.session
        goal_vec = self.memory.embedding.get_or_create(goal["goal_text"])
        all_theorems = session.query(TheoremORM).all()

        scored_theorems = sorted(
            all_theorems,
            key=lambda thm: self.similarity(goal_vec, thm.statement),
            reverse=True,
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
            .join(CartridgeTripleORM.cartridge)  # link to cartridge
            .join(CartridgeDomainORM, CartridgeDomainORM.cartridge_id == CartridgeTripleORM.cartridge_id)  # link to domains
            .filter(CartridgeDomainORM.domain.in_(goal_domain_names))
        )

        triplets = query.distinct().all()

        self.logger.log(
            "TripletsRetrievedByDomain",
            {
                "goal": goal_text,
                "domains": goal_domain_names,
                "triplet_count": len(triplets),
            },
        )

        return triplets[: self.top_k_triplets] if self.top_k_triplets else triplets

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
