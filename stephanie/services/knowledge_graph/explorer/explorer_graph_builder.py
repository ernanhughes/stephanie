# stephanie/services/knowledge_graph/explorer/explorer_graph_builder.py
from __future__ import annotations

from typing import Optional

from .explorer_graph import (
    ExplorerGraph, ExplorerEdge, ExplorerNode,
    NodeType, EdgeType, make_node_id
)

# NOTE: We intentionally type-hint as "Any" to avoid import cycles.
# In your repo you can replace these with real ORM imports if you want.
from typing import Any


class ExplorerGraphBuilder:
    """
    Builds an ExplorerGraph from a PlanTraceORM, optionally including:
    - ExecutionStep nodes
    - Evaluation nodes
    - Score nodes (ScoreORM rows)
    """

    def build_from_plan_trace(
        self,
        plan_trace: Any,
        *,
        include_evaluations: bool = True,
        include_scores: bool = True,
    ) -> ExplorerGraph:
        trace_node_id = make_node_id(NodeType.PLAN_TRACE.value, plan_trace.id)
        g = ExplorerGraph(
            root_id=trace_node_id,
            meta={
                "plan_trace_id": plan_trace.id,
                "pipeline_run_id": getattr(plan_trace, "pipeline_run_id", None),
                "version": "v1",
            },
        )

        # Root node
        g.get_or_add_node(
            trace_node_id,
            NodeType.PLAN_TRACE.value,
            label=f"PlanTrace {plan_trace.id}",
            meta={
                "trace_id": getattr(plan_trace, "trace_id", None),
                "task_type": getattr(plan_trace, "task_type", None),
                "goal_text": getattr(plan_trace, "goal_text", None),
            },
        )

        steps = sorted(getattr(plan_trace, "execution_steps", []) or [], key=lambda s: s.step_order)

        prev_step_node_id: Optional[str] = None
        for step in steps:
            step_node_id = make_node_id(NodeType.EXEC_STEP.value, step.id)

            g.get_or_add_node(
                step_node_id,
                NodeType.EXEC_STEP.value,
                label=f"{step.step_order}: {step.step_id}",
                meta={
                    "step_order": step.step_order,
                    "step_id": step.step_id,
                    "step_type": getattr(step, "step_type", None),
                    "agent_role": getattr(step, "agent_role", None),
                    "description": step.description,
                    "evaluation_id": getattr(step, "evaluation_id", None),
                    "output_embedding_id": getattr(step, "output_embedding_id", None),
                    # store output_text only if you want (can be big)
                    # "output_text": step.output_text,
                },
            )

            # Trace -> Step edge
            g.add_edge(ExplorerEdge(
                src=trace_node_id,
                dst=step_node_id,
                edge_type=EdgeType.HAS_STEP.value,
                meta={"step_order": step.step_order},
            ))

            # Step ordering edges
            if prev_step_node_id is not None:
                g.add_edge(ExplorerEdge(
                    src=prev_step_node_id,
                    dst=step_node_id,
                    edge_type=EdgeType.NEXT.value,
                    meta={},
                ))
            prev_step_node_id = step_node_id

            # Evaluation + scores
            if include_evaluations and getattr(step, "evaluation", None) is not None:
                ev = step.evaluation
                ev_node_id = make_node_id(NodeType.EVALUATION.value, ev.id)

                g.get_or_add_node(
                    ev_node_id,
                    NodeType.EVALUATION.value,
                    label=f"Evaluation {ev.id}",
                    meta={
                        "goal_id": getattr(ev, "goal_id", None),
                        "plan_trace_id": getattr(ev, "plan_trace_id", None),
                        "pipeline_run_id": getattr(ev, "pipeline_run_id", None),
                        "scorable_type": getattr(ev, "scorable_type", None),
                        "scorable_id": getattr(ev, "scorable_id", None),
                        "query_type": getattr(ev, "query_type", None),
                        "query_id": getattr(ev, "query_id", None),
                        "symbolic_rule_id": getattr(ev, "symbolic_rule_id", None),
                        "reasoning_strategy": getattr(ev, "reasoning_strategy", None),
                    },
                )

                g.add_edge(ExplorerEdge(
                    src=step_node_id,
                    dst=ev_node_id,
                    edge_type=EdgeType.HAS_EVALUATION.value,
                    meta={},
                ))

                if include_scores:
                    for s in getattr(ev, "dimension_scores", []) or []:
                        score_node_id = make_node_id(NodeType.SCORE.value, s.id)
                        g.get_or_add_node(
                            score_node_id,
                            NodeType.SCORE.value,
                            label=f"{s.dimension}={s.score}",
                            meta={
                                "dimension": s.dimension,
                                "score": s.score,
                                "weight": getattr(s, "weight", None),
                                "source": getattr(s, "source", None),
                                "prompt_hash": getattr(s, "prompt_hash", None),
                                # rationale can be big; include or not
                                "rationale": getattr(s, "rationale", None),
                            },
                        )
                        g.add_edge(ExplorerEdge(
                            src=ev_node_id,
                            dst=score_node_id,
                            edge_type=EdgeType.HAS_SCORE.value,
                            meta={},
                        ))

        return g
