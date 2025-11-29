# stephanie/scoring/scorable_factory.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from stephanie.data.plan_trace import ExecutionStep, PlanTrace
from stephanie.models.cartridge_triple import CartridgeTripleORM
from stephanie.models.casebook import CaseORM
from stephanie.models.chat import (ChatConversationORM, ChatMessageORM,
                                   ChatTurnORM)
from stephanie.models.document import DocumentORM
from stephanie.models.document_section import DocumentSectionORM
from stephanie.models.dynamic_scorable import DynamicScorableORM
from stephanie.models.hypothesis import HypothesisORM
from stephanie.models.prompt import PromptORM
from stephanie.models.theorem import CartridgeORM, TheoremORM


# Enum defining all the supported types of scoreable targets
class ScorableType:
    AGENT_OUTPUT = "agent_output"
    CARTRIDGE = "cartridge"
    CASE = "case"
    CASE_SCORABLE = "case_scorable"
    CHUNK = "chunk"
    CONVERSATION = "conversation"  # full conversation
    CONVERSATION_TURN = "conversation_turn"  # user→assistant pair
    CONVERSATION_MESSAGE = "conversation_message"  # single message
    CUSTOM = "custom"
    DOCUMENT = "document"
    DOCUMENT_SECTION = "document_section"
    DYNAMIC = "dynamic"
    GOAL = "goal"
    HYPOTHESIS = "hypothesis"
    IDEA = "idea"
    PLAN_TRACE = "plan_trace"
    PLAN_TRACE_STEP = "plan_trace_step"
    PROMPT = "prompt"
    PROMPT_RESPONSE = "prompt_response"
    RESPONSE = "response"
    SYMBOLIC_RULE = "symbolic_rule"
    TRAINING = "training"
    THEOREM = "theorem"
    TRIPLE = "triple"
    REFINEMENT = "refinement"
    VPM = "vpm"


class MissingDomainsError(ValueError):
    """Raised when a Scorable has no canonical 'domains' list."""

    pass


class Scorable:
    def __init__(
        self,
        text: str,
        id: str = "",
        target_type: str = "custom",
        meta: Dict[str, Any] = None,
        domains: Dict[str, Any] = None,
        ner: Dict[str, Any] = None,
    ):
        self._id = id
        self._text = text
        self._target_type = target_type
        self._metadata = meta or {}
        self._domains = domains or {}  # <-- keep as passed
        self._ner = ner or {}

    @property
    def meta(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def domains(self) -> Dict[str, Any]:
        return self._domains

    @property
    def ner(self) -> Dict[str, Any]:
        return self._ner

    @property
    def text(self) -> str:
        return self._text

    @property
    def id(self) -> str:
        return self._id

    @property
    def target_type(self) -> str:
        return self._target_type

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "text": self._text,
            "target_type": self._target_type,
            "metadata": self._metadata,
            "domains": self._domains,
            "ner": self._ner,
        }

    def __repr__(self):
        preview = self._text[:30].replace("\n", " ")
        return (
            f"Scorable(id='{self._id}', "
            f"target_type='{self._target_type}', "
            f"text_preview='{preview}...')"
        )

    def primary_domain(self) -> tuple[str | None, float | None, str | None]:
        items = (
            self._domains
            if isinstance(self._domains, list)
            else self._domains.get("items") or self._domains.get("domains")
        )
        if isinstance(items, list):
            return self.select_primary_domain(items)
        return None, None, None

    # Standardized domain selection from your known format:
    # [{'score': 0.62, 'domain': 'evaluation', 'source': 'seed'}, ...]
    @staticmethod
    def select_primary_domain(
        items: List[Dict[str, Any]] | None,
        prefer_sources: tuple[str, ...] = ("seed", "goal", "meta"),
    ) -> tuple[str | None, float | None, str | None]:
        if not items:
            return None, None, None
        # prefer source priority, then highest score
        ranked = sorted(
            items,
            key=lambda x: (
                (
                    prefer_sources.index(str(x.get("source")))
                    if str(x.get("source")) in prefer_sources
                    else 999
                ),
                -float(x.get("score", 0.0)),
            ),
        )
        top = ranked[0]
        return str(top.get("domain")), float(top.get("score", 0.0)), str(top.get("source"))


    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Scorable:
        return Scorable(
            text=d.get("text", ""),
            id=str(d.get("id") or d.get("scorable_id") or ""),
            target_type=d.get("target_type", "custom"),
            meta=d.get("metadata") or d.get("meta") or {},
            domains=d.get("domains") or [],
            ner=d.get("ner") or {},
        )

    @staticmethod
    def get_goal_text(
        scorable: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        goal_text = ""
        if hasattr(scorable, "goal_ref"):
            goal_text = scorable.get("goal_ref", {}).get("text", "")
        else:
            goal_text = context.get("goal", {}).get("goal_text", "")
        return goal_text


class ScorableFactory:
    """
    Factory for turning ORM objects, dicts, or plain text into unified Scorable objects.
    Now supports summary vs. full text representations.
    """

    @staticmethod
    def get_text(
        title: str, summary: str, text: str, mode: str = "default"
    ) -> str:
        """
        Utility to produce either a summary or full text representation of a document.

        Args:
            title: The document's title
            summary: A short summary
            text: The full content
            mode: "summary" or "full"/"default"
        """
        title = title or ""
        summary = summary or ""
        text = text or ""

        if mode == "summary" and summary:
            return f"{title}\n\n{summary}"
        return f"{title}\n\n{text or summary}"

    @staticmethod
    def from_orm(obj, mode: str = "default") -> Scorable:
        """
        Convert an ORM object into a Scorable.
        Mode controls whether to return summary, full, or default text.
        """

        if isinstance(obj, PromptORM):
            return ScorableFactory.from_prompt_pair(obj, mode)

        elif isinstance(obj, CartridgeORM):
            text = ScorableFactory.get_text(
                obj.title, obj.summary, obj.markdown_content, mode
            )
            return Scorable(
                id=obj.id, text=text, target_type=ScorableType.CARTRIDGE
            )

        elif isinstance(obj, HypothesisORM):
            text = obj.text or ""
            return Scorable(
                id=obj.id, text=text, target_type=ScorableType.HYPOTHESIS
            )

        elif isinstance(obj, CartridgeTripleORM):
            text = f"{obj.subject} {obj.relation} {obj.object}"
            return Scorable(
                id=obj.id, text=text, target_type=ScorableType.TRIPLE
            )

        elif isinstance(obj, TheoremORM):
            text = obj.statement or ""
            return Scorable(
                id=obj.id, text=text, target_type=ScorableType.THEOREM
            )

        elif isinstance(obj, DocumentORM):
            text = ScorableFactory.get_text(
                obj.title, obj.summary, obj.text, mode
            )
            return Scorable(
                id=obj.id, text=text, target_type=ScorableType.DOCUMENT
            )

        elif isinstance(obj, DocumentSectionORM):
            text = f"{obj.section_name}\n{obj.section_text}"
            return Scorable(
                id=obj.id, text=text, target_type=ScorableType.DOCUMENT_SECTION
            )

        elif isinstance(obj, CaseORM):
            text = ScorableFactory.get_text(
                obj.title, obj.summary, obj.text, mode
            )
            return Scorable(
                id=obj.id, text=text, target_type=ScorableType.CASE
            )

        elif isinstance(obj, PlanTrace):
            text = " ".join(
                [
                    obj.plan_signature or "",
                    obj.final_output_text or "",
                    " ".join(
                        [s.output_text for s in obj.execution_steps[:3]]
                    ),  # optional context
                ]
            )
            return Scorable(
                id=obj.trace_id, text=text, target_type=ScorableType.PLAN_TRACE
            )

        elif isinstance(obj, ChatConversationORM):
            text = "\n".join(
                [f"{m.role}: {m.text}" for m in obj.messages]
            ).strip()
            return Scorable(
                id=str(obj.id),
                text=text,
                target_type=ScorableType.CONVERSATION,
                meta=obj.to_dict(include_messages=False),
            )

        elif isinstance(obj, ChatTurnORM):
            user_text = obj.user_message.text if obj.user_message else ""
            assistant_text = (
                obj.assistant_message.text if obj.assistant_message else ""
            )
            text = f"USER: {user_text}\nASSISTANT: {assistant_text}"
            return Scorable(
                id=str(obj.id),
                text=text.strip(),
                target_type=ScorableType.CONVERSATION_TURN,
                meta=obj.to_dict(),
            )

        elif isinstance(obj, ChatMessageORM):
            text = obj.text or ""
            return Scorable(
                id=str(obj.id),
                text=text.strip(),
                target_type=ScorableType.CONVERSATION_MESSAGE,
                meta=obj.to_dict(),
            )

        elif isinstance(obj, DynamicScorableORM):
            return Scorable(
                id=str(obj.id),
                text=obj.text.strip(),
                target_type=ScorableType.DYNAMIC,
                meta=obj.to_dict(),
            )

        if hasattr(obj, "id"):
            text = getattr(obj, "text", "") or ""
            return Scorable(
                id=str(obj.id), text=text, target_type=ScorableType.CUSTOM
            )

        else:
            raise ValueError(f"Unsupported ORM type for scoring: {type(obj)}")

    @staticmethod
    def from_prompt_pair(
        obj: PromptORM, mode: str = "prompt+response"
    ) -> Scorable:
        """Handles PromptORM with different modes (prompt_only, response_only, prompt+response, summary)."""
        prompt = obj.prompt_text or ""
        response = ScorableFactory._strip_think_blocks(obj.response_text) or ""
        target_type = ScorableType.PROMPT

        if mode == "prompt_only":
            text = prompt
        elif mode == "response_only":
            text = response
            target_type = ScorableType.RESPONSE
        elif mode == "summary":
            snippet = (
                (response[:200] + "...")
                if response
                else (prompt[:200] + "...")
            )
            text = snippet
            target_type = ScorableType.PROMPT_RESPONSE
        elif mode == "prompt+response" or mode == "default":
            text = f"{prompt}\n\n{response}"
            target_type = ScorableType.PROMPT_RESPONSE
        else:
            raise ValueError(f"Invalid prompt scoring mode: {mode}")

        return Scorable(id=obj.id, text=text, target_type=target_type)

    @staticmethod
    def from_dict(
        data: dict, target_type: ScorableType = None, mode: str = "default"
    ) -> Scorable:
        """Convert dicts to Scorable. Supports summary vs. full text where applicable."""
        if target_type is None:
            target_type = (
                data.get("target_type")
                or data.get("scorable_type")
                or "document"
            )

        if target_type == ScorableType.CASE_SCORABLE:
            text = ""
            meta = data.get("meta") or {}
            if isinstance(meta, dict):
                text = meta.get("text") or meta.get("content") or ""
            if not text:
                text = (
                    f"[{data.get('role')}] {data.get('scorable_type') or ''}"
                )

            return Scorable(
                id=str(data.get("scorable_id") or data.get("id", "")),
                text=text.strip(),
                target_type=ScorableType.CASE_SCORABLE,
            )

        if target_type == ScorableType.VPM:
            from stephanie.scoring.vpm_scorable import VPMScorable

            return VPMScorable(
                id=str(
                    data.get("id")
                    or data.get("vpm_id")
                    or f"vpm:{data.get('run_id')}:{data.get('step')}"
                ),
                run_id=int(data.get("run_id", 0)),
                step=int(data.get("step", 0)),
                metric_names=data.get("metric_names", []),
                values=data.get("values", []),
                order_weights=data.get("order_weights"),
                metric_weights=data.get("metric_weights"),
                meta=data.get("metadata", {}),
            )

        # fallback to doc-like behavior
        title = data.get("title", "")
        summary = data.get("summary", "")
        in_text = data.get("text", "")
        text = ScorableFactory.get_text(title, summary, in_text, mode)
        return Scorable(
            id=str(data.get("scorable_id", data.get("id", ""))), text=text, target_type=target_type
        )

    @staticmethod
    def from_text(
        text: str, target_type: ScorableType, meta: dict = {}
    ) -> Scorable:
        """Convert plain text to Scorable. Supports summary truncation."""
        return Scorable(id="", text=text, target_type=target_type, meta=meta)

    @staticmethod
    def from_plan_trace(
        trace: PlanTrace,
        goal_text: str,
        mode: str = "default",
        step: Optional[ExecutionStep] = None,
    ) -> Scorable:
        """
        Convert a PlanTrace into a Scorable object for scoring.
        Mode can be used to customize how the trace is represented as text.

        Args:
            trace: The PlanTrace object to convert
            mode: How to represent the trace ('default', 'single_step', 'full_trace')
            step: Optional ExecutionStep for single-step mode

        Returns:
            Scorable: A scorable object representing the trace or step
        """
        if mode == "single_step" and step is not None:
            # Format a single step for scoring
            step_text = f"Step {step.step_order}: {step.step_type}\n"
            step_text += (
                f"Description: {step.description or 'No description'}\n"
            )

            # Add input if available
            if hasattr(step, "input_text") and step.input_text:
                step_text += f"Input: {step.input_text[:500]}...\n"

            # Add output with truncation
            output_text = step.output_text or ""
            step_text += f"Output: {output_text[:1000]}"
            if len(output_text) > 1000:
                step_text += "..."

            # Create scorable for this step
            return Scorable(
                id=f"{trace.trace_id}_step_{step.step_id}",
                text=step_text,
                target_type=ScorableType.PLAN_TRACE_STEP,
            )

        elif mode == "full_trace":
            # Format the complete trace for scoring
            trace_text = f"Goal: {goal_text or 'No goal text'}\n\n"
            trace_text += "Pipeline Execution Steps:\n\n"

            # Add all steps
            for i, step in enumerate(trace.execution_steps, 1):
                trace_text += (
                    f"{i}. {step.step_type}: {step.description[:100]}...\n"
                )
                output_text = step.output_text or ""
                trace_text += f"   Output: {output_text[:200]}"
                if len(output_text) > 200:
                    trace_text += "...\n\n"
                else:
                    trace_text += "\n\n"

            # Add final output
            final_output = trace.final_output_text or ""
            trace_text += f"Final Output: {final_output[:500]}"
            if len(final_output) > 500:
                trace_text += "..."

            return Scorable(
                id=trace.trace_id,
                text=trace_text,
                target_type=ScorableType.PLAN_TRACE,
            )

        else:
            # Default mode - goal + final output
            final_output = trace.final_output_text or ""

            return Scorable(
                id=trace.trace_id,
                text=f"{goal_text}\n\n{final_output}",
                target_type=ScorableType.PLAN_TRACE,
            )

    @staticmethod
    def from_id(memory, scorable_type: str, scorable_id: str):
        """
        Resolve a scorable object from memory by its type and id,
        returning a Scorable. Raises ValueError if unsupported or not found.
        """

        # Dispatch table mapping target_type -> (getter_function, cast_type)
        dispatch = {
            ScorableType.CONVERSATION: (memory.chats.get_conversation, int),
            ScorableType.CONVERSATION_TURN: (memory.chats.get_turn_by_id, int),
            ScorableType.CONVERSATION_MESSAGE: (
                memory.chats.get_message_by_id,
                int,
            ),
            ScorableType.DOCUMENT: (memory.documents.get_by_id, int),
            ScorableType.HYPOTHESIS: (memory.hypotheses.get_by_id, int),
            ScorableType.CARTRIDGE: (memory.cartridges.get_by_id, int),
            ScorableType.TRIPLE: (memory.cartridge_triples.get_by_id, int),
            ScorableType.PROMPT: (memory.prompts.get_by_id, int),
            ScorableType.THEOREM: (
                memory.theorems.get_by_id,
                int,
            ),  # plural fixed
            ScorableType.CASE: (memory.casebooks.get_case_by_id, int),
            ScorableType.CASE_SCORABLE: (
                memory.casebooks.get_case_scorable_by_id,
                int,
            ),
            ScorableType.PLAN_TRACE: (memory.plan_traces.get_by_id, str),
            ScorableType.PLAN_TRACE_STEP: (
                memory.plan_traces.get_step_by_id,
                str,
            ),
        }

        if scorable_type not in dispatch:
            raise ValueError(
                f"Unsupported target type for text extraction: {scorable_type}"
            )

        getter, caster = dispatch[scorable_type]
        orm = getter(caster(scorable_id))

        if orm is None:
            raise ValueError(
                f"No object found for {scorable_type} id={scorable_id}"
            )

        return ScorableFactory.from_orm(orm)

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        if not text:
            return text
        """Remove <think>...</think> sections from text."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
