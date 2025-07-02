<!-- Merged Python Code Files -->


## File: autonomous_worldview_cycle.py

`python
from datetime import datetime
from stephanie.agents.world.worldview_pipeline_runner import WorldviewPipelineRunner
from stephanie.agents.world.worldview_evaluator import WorldviewEvaluatorAgent
from stephanie.agents.world.belief_tuner import BeliefTunerAgent
from stephanie.agents.world.worldview_merger import WorldviewMergerAgent
from stephanie.agents.world.worldview_audit import WorldviewAuditAgent
from stephanie.core.knowledge_cartridge import KnowledgeCartridge


class AutonomousWorldviewCycleAgent:
    """
    Drives autonomous self-refinement of a worldview via pipeline execution,
    belief assimilation, evaluation, and tuning.
    """

    def __init__(self, worldview, logger=None, config=None):
        self.worldview = worldview
        self.logger = logger or self._default_logger()
        self.config = config or {}
        self.pipeline_runner = WorldviewPipelineRunner(worldview, logger=self.logger)
        self.evaluator = WorldviewEvaluatorAgent(worldview, logger=self.logger)
        self.belief_tuner = BeliefTunerAgent(worldview, logger=self.logger)
        self.merger = WorldviewMergerAgent(worldview, logger=self.logger)
        self.audit = WorldviewAuditAgent(worldview, logger=self.logger)

    def cycle_once(self):
        """
        Perform one self-improvement loop:
        1. Select goal
        2. Run pipeline
        3. Evaluate results
        4. Generate/update beliefs
        5. Tune beliefs
        6. Audit and log
        """
        goal = self._select_goal()
        if not goal:
            self.logger.log("NoGoalAvailable", {"timestamp": datetime.utcnow().isoformat()})
            return

        self.logger.log("CycleStarted", {"goal_id": goal["id"]})
        result = self.pipeline_runner.run(goal_id=goal["id"])
        evaluation = self.evaluator.evaluate(goal, result)
        cartridge = self._create_cartridge(goal, result, evaluation)

        self.belief_tuner.tune_from_result(goal, result, evaluation)
        self.worldview.add_cartridge(cartridge)
        self.audit.record_cycle(goal, cartridge)

        self.logger.log("CycleCompleted", {
            "goal_id": goal["id"],
            "score": evaluation.aggregate(),
            "timestamp": datetime.utcnow().isoformat()
        })

    def run_forever(self, interval_sec=3600):
        import time
        while True:
            self.cycle_once()
            time.sleep(interval_sec)

    def _select_goal(self):
        goals = self.worldview.list_goals()
        # Prioritize unevaluated, high-potential, or user-specified goals
        for goal in goals:
            if not goal.get("last_evaluated"):
                return goal
        return goals[0] if goals else None

    def _create_cartridge(self, goal, result, evaluation):
        cartridge = KnowledgeCartridge(
            goal=goal["text"],
            domain=goal.get("domain", "general"),
            source="autonomous_cycle",
        )
        cartridge.add_summary(str(result)[:500])
        cartridge.add_score(evaluation.aggregate())
        cartridge.add_timestamp()

        return cartridge

    def _default_logger(self):
        class DummyLogger:
            def log(self, tag, payload): print(f"[{tag}] {payload}")
        return DummyLogger()
``n

## File: belief_injest.py

`python
from stephanie.models.belief import BeliefORM
from stephanie.models.cartridge import CartridgeORM
from stephanie.models.goal import GoalORM
from stephanie.scoring.svm_scorer import SVMScorer
from stephanie.utils.summarizer import summarize_text
from stephanie.utils.embedding import EmbeddingManager

from sqlalchemy.orm import Session
from datetime import datetime

class BeliefIngestAgent:
    def __init__(self, db: Session, scorer: SVMScorer, embedding: EmbeddingManager, logger=None):
        self.db = db
        self.scorer = scorer
        self.embedding = embedding
        self.logger = logger

    def ingest_document(self, text: str, worldview_id: int, goal_id: int = None, source_uri: str = None):
        """
        Extracts belief(s) from a document and stores them.
        """
        # Step 1: Summarize key point(s)
        summary = summarize_text(text)

        # Step 2: Score belief utility and novelty
        goal_text = self._get_goal_text(goal_id)
        score_bundle = self.scorer.score(
            goal={"goal_text": goal_text},
            hypothesis={"text": summary},
            dimensions=["alignment", "novelty"]
        )
        alignment_score = score_bundle.results["alignment"].score
        novelty_score = score_bundle.results["novelty"].score

        # Step 3: Store belief in DB
        belief = BeliefORM(
            worldview_id=worldview_id,
            cartridge_id=None,  # optional: link to a CartridgeORM if available
            summary=summary,
            rationale="Auto-ingested from source document",
            utility_score=alignment_score,
            novelty_score=novelty_score,
            domain=self._infer_domain(summary),
            status="active",
            created_at=datetime.utcnow()
        )

        self.db.add(belief)
        self.db.commit()

        if self.logger:
            self.logger.log("BeliefIngested", {
                "summary": summary,
                "alignment": alignment_score,
                "novelty": novelty_score,
                "worldview_id": worldview_id
            })

        return belief

    def _get_goal_text(self, goal_id: int):
        if not goal_id:
            return "Understand and extend self-improving AI systems"
        goal = self.db.query(GoalORM).filter_by(id=goal_id).first()
        return goal.description if goal else ""

    def _infer_domain(self, text: str):
        # Placeholder: you could do zero-shot classification here
        return "ai.research.self_improvement"
``n

## File: belief_tuner.py

`python
from stephanie.models.belief import BeliefORM
from sqlalchemy.orm import Session
from datetime import datetime


class BeliefTunerAgent:
    def __init__(self, db: Session, logger=None):
        self.db = db
        self.logger = logger

    def tune_belief(
        self,
        belief_id: int,
        delta: float,
        source: str,
        rationale: str = None,
        override_score: float = None
    ):
        """Adjust or override belief trust score"""
        belief = self.db.query(BeliefORM).get(belief_id)
        if not belief:
            return None

        old_score = belief.score or 0.5

        if override_score is not None:
            belief.score = override_score
        else:
            belief.score = max(0.0, min(1.0, old_score + delta))

        belief.last_tuned = datetime.utcnow()
        belief.last_tune_source = source
        belief.last_tune_rationale = rationale or "Tuned via agent"

        self.db.commit()

        self.logger.log("BeliefTuned", {
            "belief_id": belief.id,
            "old_score": old_score,
            "new_score": belief.score,
            "source": source,
            "rationale": rationale
        })

        return belief

    def tune_by_external_signal(self, belief_text: str, signal: dict):
        """Find belief by text match and tune it based on external input"""
        matches = self.db.query(BeliefORM).filter(BeliefORM.summary.ilike(f"%{belief_text}%")).all()
        for belief in matches:
            self.tune_belief(
                belief_id=belief.id,
                delta=signal.get("delta", -0.2),
                source=signal.get("source", "external"),
                rationale=signal.get("rationale")
            )
``n

## File: context_primer.py

`python


class ContextPrimerAgent:
    def __init__(self, memory, embedding_model, logger=None):
        self.memory = memory
        self.embedding_model = embedding_model
        self.logger = logger

    def generate_hints(self, goal_text: str, top_k=5) -> list[str]:
        goal_emb = self.embedding_model.encode(goal_text)
        similar_beliefs = self.memory.beliefs.find_similar(goal_emb, top_k=top_k)
        
        hints = []
        for belief in similar_beliefs:
            if belief.usefulness_score > 0.6:  # configurable
                hints.append(belief.brief_summary or belief.title)
        
        if self.logger:
            self.logger.log("ContextPrimingGenerated", {
                "goal": goal_text,
                "num_hints": len(hints),
            })
        
        return hints
``n

## File: goal_link.py

`python
from stephanie.models.world_view import WorldviewORM
from stephanie.models.goal import GoalORM
from sklearn.metrics.pairwise import cosine_similarity

class GoalLinkingAgent:
    def __init__(self, embedding_manager, memory, logger):
        self.embedding = embedding_manager
        self.memory = memory
        self.logger = logger

    def find_or_link_worldview(self, goal: dict, threshold=0.88) -> WorldviewORM:
        """Given a new goal, find the best matching worldview (or create a new one)."""
        goal_embedding = self.embedding.get_or_create(goal["description"])
        candidates = WorldviewORM.load_all()
        
        best_match = None
        best_score = 0.0
        for candidate in candidates:
            sim = cosine_similarity(goal_embedding, candidate.embedding_vector)
            if sim > best_score:
                best_score = sim
                best_match = candidate

        if best_score >= threshold:
            self.logger.log("GoalLinkedToWorldview", {
                "goal": goal["description"],
                "worldview_id": best_match.id,
                "similarity": best_score
            })
            return best_match
        else:
            # Create new worldview
            new_view = WorldviewORM.create_from_goal(goal)
            self.logger.log("NewWorldviewCreatedFromGoal", {
                "goal": goal["description"],
                "worldview_id": new_view.id
            })
            return new_view

    def relate_to_belief_systems(self, goal: dict):
        """Suggest belief systems relevant to this goal"""
        goal_embedding = self.embedding.get_or_create(goal["description"])
        belief_docs = self.memory.belief.get_all()

        scored_beliefs = []
        for belief in belief_docs:
            belief_emb = self.embedding.get_or_create(belief["summary"])
            score = cosine_similarity(goal_embedding, belief_emb)
            scored_beliefs.append((score, belief))

        scored_beliefs.sort(reverse=True)
        return [b for score, b in scored_beliefs if score > 0.75]
``n

## File: worldview_audit.py

`python
from stephanie.models import BeliefTuneLogORM, BeliefORM, WorldviewORM
from sqlalchemy.orm import Session
import matplotlib.pyplot as plt
import pandas as pd

class WorldviewAuditAgent:
    def __init__(self, db: Session, logger=None):
        self.db = db
        self.logger = logger

    def get_belief_tuning_log(self, worldview_id: int):
        beliefs = self.db.query(BeliefORM).filter_by(worldview_id=worldview_id).all()
        log_entries = []
        for belief in beliefs:
            logs = self.db.query(BeliefTuneLogORM).filter_by(belief_id=belief.id).all()
            for log in logs:
                log_entries.append({
                    "belief_id": belief.id,
                    "title": belief.title,
                    "old_score": log.old_score,
                    "new_score": log.new_score,
                    "source": log.source,
                    "rationale": log.rationale,
                    "tuned_at": log.tuned_at
                })
        return log_entries

    def visualize_score_drift(self, belief_id: int):
        logs = self.db.query(BeliefTuneLogORM).filter_by(belief_id=belief_id).order_by(BeliefTuneLogORM.tuned_at).all()
        times = [log.tuned_at for log in logs]
        scores = [log.new_score for log in logs]

        plt.figure()
        plt.plot(times, scores, marker='o')
        plt.title(f"Belief {belief_id} Score Drift")
        plt.xlabel("Time")
        plt.ylabel("Score")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def source_influence_summary(self, worldview_id: int):
        logs = self.get_belief_tuning_log(worldview_id)
        df = pd.DataFrame(logs)
        return df.groupby("source")["belief_id"].count().sort_values(ascending=False)

    def get_recent_merges(self, worldview_id: int):
        # Requires merge logs or versioning system to be in place
        pass

    def flag_suspect_tuning(self, worldview_id: int, threshold: float = 0.5):
        logs = self.get_belief_tuning_log(worldview_id)
        df = pd.DataFrame(logs)
        df["score_change"] = (df["new_score"] - df["old_score"]).abs()
        flagged = df[df["score_change"] > threshold]
        return flagged
``n

## File: worldview_controller.py

`python
from stephanie.agents.base_agent import BaseAgent

class WorldviewControllerAgent(BaseAgent):
    def __init__(self, cfg, memory, logger, tools, pipelines):
        super().__init__(cfg, memory=memory, logger=logger)
        self.tools = tools  # e.g. { "arxiv": ArxivSearcher(), "profiler": ProfilerAgent(), ... }
        self.pipelines = pipelines  # Callable pipeline configs
        self.active_worldview = None

    def load_or_create_worldview(self, goal: dict):
        """Load an existing worldview aligned with the goal, or create a new one."""
        from stephanie.models.world_view import WorldviewORM
        # Use domain + goal hash or embedding similarity
        matched = WorldviewORM.find_nearest(goal)
        if matched:
            self.active_worldview = matched
            self.logger.log("WorldviewReused", {"worldview_id": matched.id})
        else:
            self.active_worldview = WorldviewORM.create_from_goal(goal)
            self.logger.log("WorldviewCreated", {"goal": goal})

    def run_pipeline(self, goal: dict):
        """Run the worldview through a full processing/evaluation cycle."""
        self.load_or_create_worldview(goal)

        for step in self.cfg.get("worldview_steps", ["search", "profile", "score", "ingest", "evaluate"]):
            if step == "search":
                self._search_and_add_sources(goal)
            elif step == "profile":
                self._profile_documents()
            elif step == "score":
                self._score_candidates()
            elif step == "ingest":
                self._ingest_beliefs()
            elif step == "evaluate":
                self._evaluate_and_update()

        self.logger.log("WorldviewCycleComplete", {"goal": goal})

    def _search_and_add_sources(self, goal: dict):
        sources = self.tools["arxiv"].search(goal["query"])
        self.active_worldview.add_sources(sources)

    def _profile_documents(self):
        profiler = self.tools["profiler"]
        self.active_worldview.profiled_docs = profiler.process(self.active_worldview.sources)

    def _score_candidates(self):
        scorer = self.tools["scorer"]
        self.active_worldview.scored_docs = scorer.score_all(self.active_worldview.profiled_docs)

    def _ingest_beliefs(self):
        ingestor = self.tools["belief_ingest"]
        cartridges = ingestor.ingest(self.active_worldview.scored_docs)
        for c in cartridges:
            self.active_worldview.add_cartridge(c)

    def _evaluate_and_update(self):
        evaluator = self.tools["evaluator"]
        score = evaluator.evaluate(self.active_worldview)
        self.active_worldview.update_score(score)

    def export(self):
        return self.active_worldview.export_to_markdown()

    def visualize(self):
        # Hook for worldview visualizer
        pass

    def run_autonomous_loop(self):
        """Optionally let worldview self-run repeatedly (e.g. daily update)"""
        if self.cfg.get("autonomous", False):
            import time
            interval = self.cfg.get("autonomous_interval", 86400)  # default 24h
            while True:
                self.run_pipeline(self.active_worldview.goal)
                time.sleep(interval)
``n

## File: worldview_create.py

`python
from stephanie.agents.base_agent import BaseAgent
from stephanie.worldview.db.locator import WorldviewDBLocator


class WorldViewCeate(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.base_directory = cfg.get("base_directory", "worldviews")
        self.locater = WorldviewDBLocator(self.base_directory)

    async def run(self, context: dict) -> dict:
        # Implement agent logic here
        goal_text = context.get("goal", {}).get("goal_text", "")
        if not goal_text:
            self.logger.log("WorldviewCreateNoGoal", {"context": context})
            return context
        path = self.locater.create_worldview(goal_text, self.memory.session)
        context["worldview_path"] = path
        return context
``n

## File: worldview_evaluator.py

`python
from stephanie.models.world_view import WorldviewORM
from stephanie.models.belief import BeliefORM
from stephanie.models.cartridge import CartridgeORM
from stephanie.models.icl_example import ICLExampleORM
from stephanie.scoring.svm_scorer import SVMScorer
from stephanie.scoring.score_bundle import ScoreBundle

from datetime import datetime
from sqlalchemy.orm import Session


class WorldviewEvaluatorAgent:
    def __init__(self, db: Session, scorer: SVMScorer, logger=None):
        self.db = db
        self.scorer = scorer
        self.logger = logger

    def evaluate(self, worldview_id: int, goal: dict, dimensions: list[str] = ["alignment", "utility", "novelty"]) -> dict:
        results = {}

        beliefs = self.db.query(BeliefORM).filter_by(worldview_id=worldview_id).all()
        cartridges = self.db.query(CartridgeORM).filter_by(worldview_id=worldview_id).all()

        belief_results = []
        for belief in beliefs:
            text = belief.summary + "\n" + (belief.rationale or "")
            bundle = self.scorer.score(goal, {"text": text}, dimensions=dimensions)
            belief_results.append((belief, bundle))

            self.logger.log("WorldviewBeliefScored", {
                "belief_id": belief.id,
                "scores": bundle.to_dict()
            })

        cartridge_results = []
        for cartridge in cartridges:
            thesis = cartridge.schema.get("core_thesis", "")
            bundle = self.scorer.score(goal, {"text": thesis}, dimensions=dimensions)
            cartridge_results.append((cartridge, bundle))

            self.logger.log("WorldviewCartridgeScored", {
                "cartridge_id": cartridge.id,
                "scores": bundle.to_dict()
            })

        results["beliefs"] = belief_results
        results["cartridges"] = cartridge_results

        return results

    def generate_report(self, evaluation_results) -> str:
        report_lines = ["## 🧾 Worldview Evaluation Report\n"]
        for belief, bundle in evaluation_results.get("beliefs", []):
            report_lines.append(f"### Belief: {belief.summary}")
            for dim, res in bundle.results.items():
                report_lines.append(f"- {dim.capitalize()}: {res.score:.2f} ({res.rationale})")
            report_lines.append("")

        for cart, bundle in evaluation_results.get("cartridges", []):
            report_lines.append(f"### Cartridge: {cart.goal}")
            for dim, res in bundle.results.items():
                report_lines.append(f"- {dim.capitalize()}: {res.score:.2f} ({res.rationale})")
            report_lines.append("")

        return "\n".join(report_lines)
``n

## File: worldview_generation.py

`python
class ToolPermissions:
    def __init__(self, enable_web=False, enable_arxiv=False, enable_huggingface=False):
        self.enable_web = enable_web
        self.enable_arxiv = enable_arxiv
        self.enable_huggingface = enable_huggingface

class WorldviewContext:
    def __init__(self, worldview_id, tools: ToolPermissions, embeddings):
        self.id = worldview_id
        self.tools = tools
        self.embeddings = embeddings
        self.beliefs = []
        self.domains = []
        self.goals = []
``n

## File: worldview_merger.py

`python
from stephanie.models.world_view import WorldviewORM
from stephanie.models.belief import BeliefORM
from stephanie.models.icl_example import ICLExampleORM
from stephanie.models.cartridge import CartridgeORM

from datetime import datetime
from sqlalchemy.orm import Session
import uuid

class WorldviewMergerAgent:
    def __init__(self, db: Session, embedding, logger=None):
        self.db = db
        self.embedding = embedding
        self.logger = logger

    def merge(self, source_ids: list[int], target_id: int) -> int:
        """
        Merge source worldviews into the target worldview (by ID).
        """
        target = self.db.query(WorldviewORM).filter_by(id=target_id).first()
        if not target:
            raise ValueError(f"Target worldview {target_id} not found")

        for src_id in source_ids:
            src = self.db.query(WorldviewORM).filter_by(id=src_id).first()
            if not src:
                continue

            self._merge_beliefs(src_id, target_id)
            self._merge_icl_examples(src_id, target_id)
            self._merge_cartridges(src_id, target_id)

            if self.logger:
                self.logger.log("WorldviewMerged", {
                    "source": src_id,
                    "target": target_id,
                    "timestamp": datetime.utcnow().isoformat()
                })

        self.db.commit()
        return target_id

    def _merge_beliefs(self, source_id, target_id):
        beliefs = self.db.query(BeliefORM).filter_by(worldview_id=source_id).all()
        target_beliefs = self.db.query(BeliefORM).filter_by(worldview_id=target_id).all()

        existing_summaries = set(b.summary for b in target_beliefs)

        for b in beliefs:
            if b.summary not in existing_summaries:
                merged = BeliefORM(
                    worldview_id=target_id,
                    summary=b.summary,
                    rationale=f"[Merged from worldview {source_id}] {b.rationale or ''}",
                    utility_score=b.utility_score,
                    novelty_score=b.novelty_score,
                    domain=b.domain,
                    status="active",
                    created_at=datetime.utcnow()
                )
                self.db.add(merged)

    def _merge_icl_examples(self, source_id, target_id):
        examples = self.db.query(ICLExampleORM).filter_by(worldview_id=source_id).all()
        for e in examples:
            new_e = ICLExampleORM(
                worldview_id=target_id,
                prompt=e.prompt,
                response=e.response,
                task_type=e.task_type,
                source=f"Merged from {source_id}",
                created_at=datetime.utcnow()
            )
            self.db.add(new_e)

    def _merge_cartridges(self, source_id, target_id):
        cartridges = self.db.query(CartridgeORM).filter_by(worldview_id=source_id).all()
        for c in cartridges:
            merged_cart = CartridgeORM(
                worldview_id=target_id,
                goal=c.goal,
                generation=c.generation,
                schema=c.schema,
                created_at=datetime.utcnow(),
                source=f"Merged from worldview {source_id}"
            )
            self.db.add(merged_cart)
``n

## File: worldview_pipeline_runner.py

`python
# worldview_pipeline_runner.py

from datetime import datetime
from stephanie.worldview.worldview import Worldview
from stephanie.registry.pipeline import PipelineRegistry
from stephanie.agents.logger_mixin import LoggingMixin


class WorldviewPipelineRunner(LoggingMixin):
    """
    Runs pipelines within a given worldview context.
    Orchestrates execution based on worldview goals and configuration.
    """

    def __init__(self, worldview: Worldview, logger=None):
        self.worldview = worldview
        self.logger = logger or self._init_logger()

    def run(self, goal_id: str, pipeline_name: str = None, input_overrides: dict = None):
        """
        Runs a pipeline associated with a specific goal from the worldview.

        Args:
            goal_id (str): The ID of the goal to pursue
            pipeline_name (str): Optionally override the default pipeline for the goal
            input_overrides (dict): Optional overrides for runtime input
        """
        goal = self.worldview.get_goal(goal_id)
        if not goal:
            raise ValueError(f"Goal {goal_id} not found in worldview")

        pipeline_key = pipeline_name or goal.get("default_pipeline")
        if not pipeline_key:
            raise ValueError(f"No pipeline specified for goal {goal_id}")

        pipeline = get_pipeline_by_name(pipeline_key)
        if not pipeline:
            raise ValueError(f"Pipeline '{pipeline_key}' not found")

        inputs = goal.get("input") or {}
        if input_overrides:
            inputs.update(input_overrides)

        self.logger.log("PipelineExecutionStarted", {
            "goal_id": goal_id,
            "pipeline": pipeline_key,
            "timestamp": datetime.utcnow().isoformat(),
            "inputs": inputs,
        })

        # Run the pipeline in the context of worldview beliefs/tools
        result = pipeline.run(
            inputs=inputs,
            context={
                "beliefs": self.worldview.get_belief_context(),
                "tools": self.worldview.get_enabled_tools(),
                "embedding": self.worldview.get_embedding_model(),
                "memory": self.worldview.memory,
            }
        )

        self.logger.log("PipelineExecutionFinished", {
            "goal_id": goal_id,
            "pipeline": pipeline_key,
            "timestamp": datetime.utcnow().isoformat(),
            "output_summary": str(result)[:300],
        })

        return result
``n

## File: worldview_visualizer.py

`python
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from stephanie.core.knowledge_cartridge import KnowledgeCartridge


class WorldviewVisualizer:
    """
    Visualizes key components of a worldview: belief systems, goal influence, cartridge scores,
    tuning changes over time, and belief network connectivity.
    """

    def __init__(self, worldview, logger=None):
        self.worldview = worldview
        self.logger = logger or self._default_logger()

    def plot_cartridge_scores(self):
        """Plot cartridge scores over time per domain"""
        cartridges = self.worldview.list_cartridges()
        domain_scores = defaultdict(list)

        for c in cartridges:
            domain = c.domain or "unknown"
            score = c.score or 0
            timestamp = c.timestamp or "n/a"
            domain_scores[domain].append((timestamp, score))

        for domain, scores in domain_scores.items():
            scores = sorted(scores, key=lambda x: x[0])
            timestamps, values = zip(*scores)
            plt.figure(figsize=(10, 4))
            plt.plot(timestamps, values, label=f"{domain} scores")
            plt.xticks(rotation=45)
            plt.title(f"Belief Score Over Time - {domain}")
            plt.xlabel("Timestamp")
            plt.ylabel("Score")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def plot_belief_tuning_history(self):
        """Plot belief tuning history if available"""
        tuning_data = self.worldview.audit.get_tuning_history()
        if not tuning_data:
            print("No tuning history available.")
            return

        for belief_id, records in tuning_data.items():
            timestamps = [r["timestamp"] for r in records]
            values = [r["new_score"] for r in records]

            plt.figure(figsize=(8, 3))
            plt.plot(timestamps, values, marker='o')
            plt.title(f"Tuning History for Belief {belief_id}")
            plt.xlabel("Time")
            plt.ylabel("Score")
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.show()

    def draw_belief_influence_graph(self):
        """Draw a graph of how cartridges, beliefs, and goals are linked"""
        G = nx.DiGraph()

        for goal in self.worldview.list_goals():
            goal_id = f"goal:{goal['id']}"
            G.add_node(goal_id, label=goal["text"], type="goal")

        for cartridge in self.worldview.list_cartridges():
            cid = f"cart:{cartridge.id}"
            G.add_node(cid, label=cartridge.summary[:50], type="cartridge")

            # Link cartridge to goal
            if cartridge.goal_id:
                G.add_edge(f"goal:{cartridge.goal_id}", cid)

            # Link beliefs (if structured that way)
            for belief in cartridge.beliefs:
                bid = f"belief:{belief.get('id', belief.get('title', '')[:20])}"
                G.add_node(bid, label=belief.get("title", ""), type="belief")
                G.add_edge(cid, bid)

        pos = nx.spring_layout(G, k=0.5)
        plt.figure(figsize=(12, 8))

        colors = []
        for n in G.nodes(data=True):
            if n[1]['type'] == "goal":
                colors.append("lightblue")
            elif n[1]['type'] == "cartridge":
                colors.append("lightgreen")
            else:
                colors.append("lightcoral")

        nx.draw(G, pos, node_color=colors, with_labels=True, font_size=8, arrows=True)
        plt.title("Worldview Belief Influence Graph")
        plt.show()

    def _default_logger(self):
        class DummyLogger:
            def log(self, tag, payload): print(f"[{tag}] {payload}")
        return DummyLogger()
``n
