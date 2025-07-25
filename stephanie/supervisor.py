# stephanie/supervisor.py

import json
import os
from datetime import datetime, timezone
from uuid import uuid4

import hydra
from dependency_injector.wiring import Provide, inject
from omegaconf import DictConfig, OmegaConf

from stephanie.constants import (GOAL, NAME, PIPELINE, PIPELINE_RUN_ID,
                                 PROMPT_DIR, RUN_ID, SAVE_CONTEXT,
                                 SKIP_IF_COMPLETED, STAGE)
from stephanie.containers import AppContainer
from stephanie.engine.cycle_watcher import CycleWatcher
from stephanie.engine.meta_confidence import MetaConfidenceTracker
from stephanie.engine.self_validation import SelfValidationEngine
from stephanie.engine.state_tracker import StateTracker
from stephanie.engine.training_controller import TrainingController
from stephanie.logs.json_logger import JSONLogger
from stephanie.memory import MemoryTool
from stephanie.protocols.base import Protocol
from stephanie.registry.agent_registry import AgentRegistry
from stephanie.registry.registry import register
from stephanie.reports import ReportFormatter
from stephanie.rules.symbolic_rule_applier import SymbolicRuleApplier
from stephanie.utils.timing import time_function

from tabulate import tabulate
import time


class PipelineStage:
    def __init__(self, name: str, config: dict, stage_dict: dict):
        self.name = name
        self.description = config.get("description", "")
        self.cls = config.get("cls", "")
        self.enabled = config.get("enabled", True)
        self.iterations = config.get("iterations", 1)
        self.stage_dict = stage_dict


class SingleAgentPipeline:
    def __init__(self, agent_name, config, container: AppContainer):
        self.agent = AgentRegistry(config).get(agent_name)
        self.goal_input = config.input_path
        self.container = container

    def run(self):
        pass
        # goals = load_goal_list(self.goal_input)
        # for goal in goals:
        #     result = self.agent.run(goal)
        # wrap it in pipeline logs, score evals, etc.

container = AppContainer()

class Supervisor:
    def __init__(self, cfg, memory=None, logger=None, container: AppContainer = None):
        self.cfg = cfg
        self.container = container or AppContainer()
        self.container.init_resources()  # Important!
        self.memory = memory or MemoryTool(cfg=cfg.db, logger=logger)
        self.logger = logger or JSONLogger(log_path=cfg.logger.log_path)
        self.logger.log("SupervisorInit", {"cfg": cfg})
        self.rule_applier = SymbolicRuleApplier(cfg, self.memory, self.logger)
        print(f"Parsing pipeline stages from config: {cfg.pipeline}")
        self.pipeline_stages = self._parse_pipeline_stages(cfg.pipeline.stages)

        # Initialize and register core components
        state_tracker = StateTracker(cfg, self.memory, self.logger)
        confidence_tracker = MetaConfidenceTracker(cfg, self.memory, self.logger)
        cycle_watcher = CycleWatcher(cfg, self.memory, self.logger)

        # Stub judgment and reward model evaluators
        def reward_model_fn(goal, doc_a, doc_b):
            return "a" if len(doc_a) >= len(doc_b) else "b"

        def llm_judge_fn(goal, doc_a, doc_b):
            return "a"  # Placeholder logic

        validator = SelfValidationEngine(
            cfg=cfg,
            memory=self.memory,
            logger=self.logger,
            reward_model=reward_model_fn,
            llm_judge=llm_judge_fn,
        )

        def trainer_fn(goal, dimension):
            print(f"[TRAIN] Training RM for goal: {goal}, dimension: {dimension}")
            # Insert actual training logic here

        training_controller = TrainingController(
            cfg=cfg,
            memory=self.memory,
            logger=self.logger,
            validator=validator,
            tracker=confidence_tracker,
            trainer_fn=trainer_fn,
        )

        register("state_tracker", state_tracker)
        register("confidence_tracker", confidence_tracker)
        register("cycle_watcher", cycle_watcher)
        register("training_controller", training_controller)
        register("self_validation", validator)
        self.logger.log(
            "SupervisorComponentsRegistered",
            {
                "state_tracker": state_tracker,
                "confidence_tracker": confidence_tracker,
                "cycle_watcher": cycle_watcher,
                "training_controller": training_controller,
            },
        )

    def _parse_pipeline_stages(
        self, stage_configs: list[dict[str, any]]
    ) -> list[PipelineStage]:
        """Parse and validate pipeline stages from config."""
        stages = []
        for stage_config in stage_configs:
            name = stage_config.get("name")
            if not stage_config.get("enabled"):
                print(f"Skipping disabled stage: {name}")
                continue
            stage_dict = self.cfg.agents[name]
            self.logger.log("StageContext", {"stage_dict": stage_dict})
            stages.append(PipelineStage(name, stage_config, stage_dict))
        return stages

    async def run_pipeline_config(self, input_data: dict) -> dict:
        """
        Run all stages defined in config.
        Each stage loads its class dynamically via hydra.utils.get_class()
        """
        self.logger.log("PipelineStart", input_data)
        input_file = input_data.get("input_file", self.cfg.get("input_file", None))

        if input_file and os.path.exists(input_file):
            self.logger.log("BatchProcessingStart", {"file": input_file})
            with open(input_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    goal_dict = json.loads(line)
                    goal_orm = self.memory.goals.get_or_create(goal_dict)

                    run_id = goal_orm.get("id", f"goal_{i}")
                    context = {
                        GOAL: goal_dict,
                        RUN_ID: run_id,
                        "prompt_dir": self.cfg.paths.prompts,
                        PIPELINE: [stage.name for stage in self.pipeline_stages],
                    }
                    try:
                        await self._run_pipeline_stages(context)
                    except Exception as e:
                        self.logger.log(
                            "BatchItemFailed",
                            {"index": i, "run_id": run_id, "error": str(e)},
                        )
            self.logger.log("BatchProcessingComplete", {"file": input_file})
            return {"status": "completed_batch", "input_file": input_file}

        goal_dict = self.get_goal(input_data)
        run_id = str(uuid4())
        pipeline_list = [stage.name for stage in self.pipeline_stages]

        context = input_data.copy()
        context.update(
            {
                RUN_ID: run_id,
                PIPELINE: pipeline_list,
                PROMPT_DIR: self.cfg.paths.prompts,
                "goal": goal_dict,
            }
        )

        # Create and store PipelineRun
        pipeline_run_data = {
            "name": self.cfg.get("pipeline", {}).get(NAME, "UnnamedPipelineRun"),
            "tag": self.cfg.get("pipeline", {}).get("tag", "default"),
            "description": self.cfg.get("pipeline", {}).get("description", ""),
            "goal_id": goal_dict.get("id"),
            "embedding_type": self.memory.embedding.type,
            "embedding_dimensions": self.memory.embedding.dim,
            "run_id": run_id,
            "pipeline": pipeline_list,  # Should be list of strings like ["generation", "judge"]
            "strategy": context.get("strategy"),
            "model_name": self.cfg.get("model.name", "unknown"),
            "run_config": OmegaConf.to_container(self.cfg),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Insert into DB
        run_id = self.memory.pipeline_runs.insert(pipeline_run_data)
        context[PIPELINE_RUN_ID] = run_id

        # Now allow lookahead or other steps to adjust context
        context = await self.maybe_adjust_pipeline(context)
        context = self.rule_applier.apply(context)
        return await self._run_pipeline_stages(context)

    def _parse_pipeline_stages_from_list(
        self, stage_names: list[str]
    ) -> list[PipelineStage]:
        return [
            PipelineStage(name, self.cfg.pipeline.stages[name], self.cfg.agents[name])
            for name in stage_names
            if name in self.cfg.agents
        ]

    @time_function(logger=None)
    async def _run_pipeline_stages(self, context: dict) -> dict:
        for stage in self.pipeline_stages:
            context = await self._run_single_stage(stage, context)

            # Post-judgment hook
            if self.cfg.get("post_judgment", {}).get("enabled", False):
                context = await self._maybe_run_pipeline_judge(context)
        self._print_pipeline_summary(context)
        return context

    async def _maybe_run_pipeline_judge(self, context: dict) -> dict:
        """
        Optionally run a pipeline judge to evaluate the context after all stages.
        This is useful for final validation or scoring.
        """
        if not self.cfg.get("pipeline_judge", {}).get("enabled", False):
            return context
        self.logger.log("PipelineJudgeStart", {"context_keys": list(context.keys())})

        judge_cfg = OmegaConf.to_container(self.cfg.post_judgment, resolve=True)
        stage_dict = OmegaConf.to_container(
            self.cfg.agents.pipeline_judge, resolve=True
        )
        judge_cls = hydra.utils.get_class(judge_cfg["cls"])
        judge_agent = judge_cls(cfg=stage_dict, memory=self.memory, logger=self.logger)
        context = await judge_agent.run(context)
        self.logger.log("PipelineJudgeEnd", {"context_keys": list(context.keys())})
        return context


    @inject
    async def _run_with_protocol(
        self,
        context: dict,
        protocol_name: str,
        protocol: Protocol = Provide["container.protocol_selector"]
    ):
        """
        Runs the given protocol.
        Dependency Injector resolves `protocol` based on `protocol_name`.
        """
        result = protocol.run(context)
        return result
    
    async def _run_single_stage(self, stage: PipelineStage, context: dict) -> dict:
        stage_details = {
            STAGE: stage.name,
        }
        start_time = time.time()
        context.setdefault("STAGE_DETAILS", []).append(stage_details)
        if not stage.enabled:
            self.logger.log("PipelineStageSkipped", {STAGE: stage.name, "reason": "disabled_in_config"})
            stage_details["status"] = "⏭️ skipped"
            return context

        try:
            cls = hydra.utils.get_class(stage.cls)
            stage_dict = OmegaConf.to_container(stage.stage_dict, resolve=True)
            stage_details["agent"] = stage.cls.split(".")[-1]
            self.rule_applier.apply_to_agent(stage_dict, context)

            # Try loading saved context
            goal_id = context.get(GOAL, {}).get("id")
            saved_context = self.load_context(stage_dict, goal_id=goal_id)
            if saved_context:
                self.logger.log("PipelineStageSkipped", {STAGE: stage.name, "reason": "context_loaded"})
                return {**context, **saved_context}

            agent_args = {
                "cfg": stage_dict,
                "memory": self.memory,
                "logger": self.logger,
            }
            if "full_cfg" in cls.__init__.__code__.co_varnames:
                agent_args["full_cfg"] = self.cfg

            agent = cls(**agent_args)

            self.logger.log("PipelineStageStart", {STAGE: stage.name})

            for i in range(stage.iterations or 1):
                self.logger.log("PipelineIterationStart", {STAGE: stage.name, "iteration": i + 1})
                context = await agent.run(context)

                if self.rule_applier.rules:
                    self.rule_applier.track_pipeline_stage(stage_dict, context)

                self.logger.log("PipelineIterationEnd", {STAGE: stage.name, "iteration": i + 1})

            self.save_context(stage_dict, context)
            self._save_pipeline_stage(stage, context, stage_dict)
            self.logger.log("PipelineStageEnd", {STAGE: stage.name})

            stage_details["status"] = "✅ completed"
            stage_details["duration"] = time.time() - start_time
            return context

        except Exception as e:
            self.logger.log("PipelineStageFailed", {"stage": stage.name, "error": str(e)})
            stage_details["status"] = "💀 failed"
            stage_details["duration"] = time.time() - start_time
            return context
        
    def _save_pipeline_stage(self, stage: PipelineStage, context: dict, stage_dict: dict):
        """
        Saves the current pipeline stage along with input/output context.
        """
        run_id = context.get(RUN_ID)
        if not run_id:
            self.logger.log("PipelineStageSkipped", {"reason": "no_run_id_in_context"})
            return

        # Get input/output context IDs
        input_context_id = None
        output_context_id = None

        if hasattr(self.memory, 'context') and self.memory.context:
            # Get latest saved context ID from memory.context
            latest_context = self.memory.context.get_latest(run_id=run_id)
            if latest_context:
                output_context_id = latest_context.id

            # If this is not the first stage, get previous context
            prev_context = self.memory.context.get_previous(run_id=run_id)
            if prev_context:
                input_context_id = prev_context.id

        # Build stage data
        stage_data = {
            "stage_name": stage.name,
            "agent_class": stage.cls,
            "protocol_used": context.get("protocol_used", "unknown"),
            "goal_id": context.get(GOAL, {}).get("id"),
            "run_id": run_id,
            "pipeline_run_id": context.get(PIPELINE_RUN_ID),
            "input_context_id": input_context_id,
            "output_context_id": output_context_id,
            "status": "accepted",
            "score": context.get("score"),
            "confidence": context.get("confidence"),
            "symbols_applied": context.get("symbols_applied"),
            "extra_data": {
                "model": self.cfg.get("model", {}).get("name"),
                "agent_config": stage_dict,
            },
            "exportable": True,
            "reusable": True,
            "invalidated": False,
        }


        return self.memory.pipeline_stages.insert(stage_data)

    def generate_report(self, context: dict[str, any], run_id: str) -> str:
        """Generate a report based on the pipeline context."""
        formatter = ReportFormatter(self.cfg.report.path)
        report = formatter.format_report(context)
        # self.memory.report.log(
        #     run_id, str(context.get("goal")), report, self.cfg.report.path
        # )
        self.logger.log(
            "ReportGenerated", {RUN_ID: run_id, "report_snippet": report[:100]}
        )
        return report

    @time_function(logger=None)
    def save_context(self, cfg: DictConfig, context: dict):
        if self.memory and cfg.get(SAVE_CONTEXT, True):
            run_id = context.get(RUN_ID)
            name = cfg.get(NAME, "NoAgentNameInConfig")
            try:
                self.memory.context.save(run_id, name, context, cfg)
            except Exception as e:
                self.logger.log(
                    "ContextSaveFailed",
                    {
                        "run_id": run_id,
                        "name": name,
                        "error": str(e),
                        "context_keys": list(context.keys()),
                    },
                )
                for k, v in context.items():
                    self.inspect_context_serializability(v, f"context[{repr(k)}]")
                    print(f"Context Key: {k}, Type: {type(v)}")



            self.logger.log(
                "ContextSaved",
                {NAME: name, RUN_ID: run_id, "context_keys": list(context.keys())},
            )

    def inspect_context_serializability(self, obj, path="context"):
        try:
            json.dumps(obj)
        except TypeError as e:
            print(f"❌ Non-serializable at {path} → {type(obj)}: {e}")
            if isinstance(obj, dict):
                for k, v in obj.items():
                    self.inspect_context_serializability(v, f"{path}[{repr(k)}]")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    self.inspect_context_serializability(item, f"{path}[{i}]")
            elif hasattr(obj, "__dict__"):
                for attr, val in vars(obj).items():
                    self.inspect_context_serializability(val, f"{path}.{attr}")
            else:
                print(f"⚠️ Unknown non-serializable object at {path}: {type(obj)}")
        else:
            # Optional: print serializable paths
            pass


    def inspect_non_serializable(self, obj, path="context"):
        try:
            json.dumps(obj)
        except TypeError as e:
            print(f"❌ Non-serializable at {path} → {type(obj)}: {e}")
            if isinstance(obj, dict):
                for k, v in obj.items():
                    self.inspect_non_serializable(v, f"{path}[{repr(k)}]")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    self.inspect_non_serializable(item, f"{path}[{i}]")
            elif hasattr(obj, "__dict__"):
                for attr, val in vars(obj).items():
                    self.inspect_non_serializable(val, f"{path}.{attr}")




    def load_context(self, cfg: DictConfig, goal_id: int):
        if self.memory and cfg.get(SKIP_IF_COMPLETED, False):
            name = cfg.get(NAME, None)
            if name and self.memory.context.has_completed(goal_id, name):
                saved_context = self.memory.context.load(goal_id, name)
                if saved_context:
                    self.logger.log("ContextLoaded", {"Goal Id": goal_id, NAME: name})
                    return saved_context
        return None

    async def maybe_adjust_pipeline(self, context: dict) -> dict:
        """
        Optionally run DOTSPlanner and/or LookaheadAgent to revise or select the pipeline.
        """
        goal = context.get("goal", {})

        # === RUN DOTS PLANNER FIRST (STRATEGY PLANNER) ===
        if self.cfg.get("planner", {}).get("enabled", False):
            try:
                planner_cfg = OmegaConf.to_container(self.cfg.planner, resolve=True)
                planner_cls = hydra.utils.get_class(planner_cfg["cls"])
                planner = planner_cls(
                    cfg=planner_cfg, memory=self.memory, logger=self.logger
                )

                context = await planner.run(context)

                if "suggested_pipeline" in context:
                    suggested = context["suggested_pipeline"]
                    self.logger.log(
                        "PipelineUpdatedByDOTSPlanner",
                        {
                            "strategy": context.get("strategy", "unknown"),
                            "suggested": suggested,
                        },
                    )
                    self.pipeline_stages = self._parse_pipeline_stages_from_list(
                        suggested
                    )
            except Exception as e:
                self.logger.log("DOTSPlannerFailed", {"error": str(e)})

        # === RUN LOOKAHEAD SECOND (OPTIONAL REFLECTIVE OVERRIDE) ===
        if not self.cfg.get("dynamic", {}).get("lookahead_enabled", False):
            return context

        try:
            lookahead_cfg = OmegaConf.to_container(self.cfg.dynamic, resolve=True)
            stage_dict = OmegaConf.to_container(self.cfg.agents.lookahead, resolve=True)
            stage_dict = self.rule_applier.apply_to_agent(stage_dict, context)
            agent_cls = hydra.utils.get_class(lookahead_cfg["cls"])
            lookahead_agent = agent_cls(
                cfg=stage_dict, memory=self.memory, logger=self.logger
            )

            self.logger.log("LookaheadStart", {"goal": goal})
            context[PIPELINE] = [stage.name for stage in self.pipeline_stages]
            context["agent_registry"] = OmegaConf.to_container(
                OmegaConf.load("config/registry/agent_registry.yaml")["agents"]
            )
            updated_context = await lookahead_agent.run(context)

            if "suggested_pipeline" in updated_context:
                suggested = updated_context["suggested_pipeline"]
                self.logger.log(
                    "PipelineUpdatedByLookahead",
                    {
                        "original": [stage.name for stage in self.pipeline_stages],
                        "suggested": suggested,
                    },
                )
                self.pipeline_stages = self._parse_pipeline_stages_from_list(suggested)
            return updated_context

        except Exception as e:
            self.logger.log("LookaheadFailed", {"error": str(e)})
            return context

    async def rerun_pipeline(self, run_id: str) -> dict:
        """
        Re-run a previously stored pipeline run by its run_id.
        """
        self.logger.log("PipelineRerunStart", {"run_id": run_id})

        # Step 1: Load pipeline run
        pipeline_run = self.memory.pipeline_runs.get_by_run_id(run_id)
        if not pipeline_run:
            raise ValueError(f"No pipeline run found with run_id={run_id}")

        # Step 2: Load goal object
        goal = self.memory.goals.get_by_id(pipeline_run.goal_id)
        if not goal:
            raise ValueError(f"No goal found with goal_id={pipeline_run.goal_id}")

        # Step 3: Build context
        context = {
            "goal": goal,
            RUN_ID: run_id,
            PIPELINE: pipeline_run.pipeline,
            "strategy": pipeline_run.strategy,
            "model_config": pipeline_run.run_config,
            PROMPT_DIR: self.cfg.paths.prompts,
        }

        # Optional: override pipeline stages to match recorded run
        self.pipeline_stages = self._parse_pipeline_stages_from_list(
            pipeline_run.pipeline
        )

        # Optional: reapply lookahead suggestion or symbolic context (or skip it for pure repeatability)
        # context["lookahead"] = pipeline_run.lookahead_context
        # context["symbolic_suggestion"] = pipeline_run.symbolic_suggestion

        # Step 4: Run
        context = await self._run_pipeline_stages(context)

        # Step 5: Generate report (optional)
        self.generate_report(context, run_id)

        self.logger.log("PipelineRerunComplete", {"run_id": run_id})
        return context

    def analyze_pipeline_deltas(self, goal_id: int):
        from stephanie.analysis.reflection_delta import compare_pipeline_runs

        deltas = compare_pipeline_runs(self.memory, goal_id)
        for delta in deltas:
            self.logger.log("ReflectionDeltaComputed", delta)

    def get_goal(self, input_data: dict) -> dict:
        goal_dict = input_data.get("goal")
        if not goal_dict:
            raise ValueError("Missing 'goal' key in input_data")
        goal_orm = self.memory.goals.get_or_create(goal_dict)

        merged = goal_orm.to_dict()
        for key, value in goal_dict.items():
            if value is not None:
                if key in merged and merged[key] != value:
                    self.logger.log(
                        "GoalContextOverride",
                        {
                            "field": key,
                            "original": merged[key],
                            "override": value,
                            "note": "Overriding goal field from context",
                        },
                    )
                merged[key] = value
        return merged

    # Inside your supervisor.py
    def check_for_retraining(self):
        query = """
        SELECT dimension, AVG(uncertainty) as avg_uncertainty
        FROM scoring_history
        WHERE created_at > NOW() - INTERVAL '1 day'
        GROUP BY dimension
        """
        results = self.memory.session.execute(query).fetchall()
        
        for r in results:
            if r.avg_uncertainty > self.cfg.get("retrain_threshold", 0.75):
                self.logger.log("EBTRetrainNeeded", {
                    "dimension": r.dimension,
                    "avg_uncertainty": r.avg_uncertainty
                })
                # Trigger retraining
                self._trigger_ebt_retraining(r.dimension)

    def _print_pipeline_summary(self, context:dict):
        print("\n🖇️ Pipeline Execution Summary:\n")
        print(f"\n🆔 Pipeline: {context.get(PIPELINE)} Run ID: {context.get(RUN_ID)}")
        summary = context.get("STAGE_DETAILS", {})
        table = tabulate(
            summary,
            headers="keys",
            tablefmt="fancy_grid"
        )
        print(table)
        self.logger.log("PipelineSummaryPrinted", {"summary": summary})    

__all__ = ["Supervisor", "container"]