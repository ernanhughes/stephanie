# stephanie/scoring/scorable_factory.py
from typing import Optional

from stephanie.data.plan_trace import ExecutionStep, PlanTrace
from stephanie.models.cartridge_triple import CartridgeTripleORM
from stephanie.models.casebook import CaseORM
from stephanie.models.chat import (ChatConversationORM, ChatMessageORM,
                                   ChatTurnORM)
from stephanie.models.document import DocumentORM
from stephanie.models.hypothesis import HypothesisORM
from stephanie.models.prompt import PromptORM
from stephanie.models.theorem import CartridgeORM, TheoremORM
from stephanie.scoring.scorable import Scorable


# Enum defining all the supported types of scoreable targets
class TargetType:
    AGENT_OUTPUT = "agent_output"
    DOCUMENT = "document"
    CONVERSATION = "conversation"       # full conversation
    CONVERSATION_TURN = "conversation_turn"  # user→assistant pair
    CONVERSATION_MESSAGE = "conversation_message"  # single message
    GOAL = "goal"
    CASE = "case"
    CASE_SCORABLE = "case_scorable"
    HYPOTHESIS = "hypothesis"
    CARTRIDGE = "cartridge"
    TRIPLE = "triple"
    CHUNK = "chunk"
    PROMPT = "prompt"
    IDEA = "idea"
    RESPONSE = "response"
    PROMPT_RESPONSE = "prompt_response"
    TRAINING = "training"
    THEOREM = "theorem"
    SYMBOLIC_RULE = "symbolic_rule"
    CUSTOM = "custom"
    REFINEMENT = "refinement"
    PLAN_TRACE = "plan_trace"
    PLAN_TRACE_STEP = "plan_trace_step"


class ScorableFactory:
    """
    Factory for turning ORM objects, dicts, or plain text into unified Scorable objects.
    Now supports summary vs. full text representations.
    """

    @staticmethod
    def get_text(title: str, summary: str, text: str, mode: str = "default") -> str:
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
            text = ScorableFactory.get_text(obj.title, obj.summary, obj.markdown_content, mode)
            return Scorable(id=obj.id, text=text, target_type=TargetType.CARTRIDGE)

        elif isinstance(obj, HypothesisORM):
            text = obj.text or ""
            return Scorable(id=obj.id, text=text, target_type=TargetType.HYPOTHESIS)

        elif isinstance(obj, CartridgeTripleORM):
            text = f"{obj.subject} {obj.relation} {obj.object}"
            return Scorable(id=obj.id, text=text, target_type=TargetType.TRIPLE)

        elif isinstance(obj, TheoremORM):
            text = obj.statement or ""
            return Scorable(id=obj.id, text=text, target_type=TargetType.THEOREM)

        elif isinstance(obj, DocumentORM):
            text = ScorableFactory.get_text(obj.title, obj.summary, obj.text, mode)
            return Scorable(id=obj.id, text=text, target_type=TargetType.DOCUMENT)

        elif isinstance(obj, CaseORM):
            text = ScorableFactory.get_text(obj.title, obj.summary, obj.text, mode)
            return Scorable(id=obj.id, text=text, target_type=TargetType.CASE)

        elif isinstance(obj, PlanTrace):
            text = " ".join([
                obj.plan_signature or "",
                obj.final_output_text or "",
            " ".join([s.output_text for s in obj.execution_steps[:3]])  # optional context
            ])
            return Scorable(id=obj.trace_id, text=text, target_type=TargetType.PLAN_TRACE)

        elif isinstance(obj, ChatConversationORM):
            text = "\n".join([f"{m.role}: {m.text}" for m in obj.messages]).strip()
            return Scorable(
                id=str(obj.id),
                text=text,
                target_type=TargetType.CONVERSATION,
                meta=obj.to_dict(include_messages=False)
            )

        elif isinstance(obj, ChatTurnORM):
            user_text = obj.user_message.text if obj.user_message else ""
            assistant_text = obj.assistant_message.text if obj.assistant_message else ""
            text = f"USER: {user_text}\nASSISTANT: {assistant_text}"
            return Scorable(
                id=str(obj.id),
                text=text.strip(),
                target_type=TargetType.CONVERSATION_TURN,
                meta=obj.to_dict()
            )

        elif isinstance(obj, ChatMessageORM):
            text = obj.text or ""
            return Scorable(
                id=str(obj.id),
                text=text.strip(),
                target_type=TargetType.CONVERSATION_MESSAGE,
                meta=obj.to_dict()
            )
        else:
            raise ValueError(f"Unsupported ORM type for scoring: {type(obj)}")

    @staticmethod
    def from_prompt_pair(obj: PromptORM, mode: str = "prompt+response") -> Scorable:
        """Handles PromptORM with different modes (prompt_only, response_only, prompt+response, summary)."""
        prompt = obj.prompt or ""
        response = obj.response or ""
        target_type = TargetType.PROMPT

        if mode == "prompt_only":
            text = prompt
        elif mode == "response_only":
            text = response
            target_type = TargetType.RESPONSE
        elif mode == "summary":
            snippet = (response[:200] + "...") if response else (prompt[:200] + "...")
            text = snippet
            target_type = TargetType.PROMPT_RESPONSE
        elif mode == "prompt+response" or mode == "default":
            text = f"{prompt}\n\n{response}"
            target_type = TargetType.PROMPT_RESPONSE
        else:
            raise ValueError(f"Invalid prompt scoring mode: {mode}")

        return Scorable(id=obj.id, text=text, target_type=target_type)

    @staticmethod
    def from_dict(data: dict, target_type: TargetType = None, mode: str = "default") -> Scorable:
        """Convert dicts to Scorable. Supports summary vs. full text where applicable."""
        if target_type is None:
            target_type = data.get("target_type") or data.get("scorable_type") or "document"

        if target_type == TargetType.CASE_SCORABLE:
            text = ""
            meta = data.get("meta") or {}
            if isinstance(meta, dict):
                text = meta.get("text") or meta.get("content") or ""
            if not text:
                text = f"[{data.get('role')}] {data.get('scorable_type') or ''}"

            return Scorable(
                id=str(data.get("scorable_id") or data.get("id", "")),
                text=text.strip(),
                target_type=TargetType.CASE_SCORABLE,
            )

        # fallback to doc-like behavior
        title = data.get("title", "")
        summary = data.get("summary", "")
        in_text = data.get("text", "")
        text = ScorableFactory.get_text(title, summary, in_text, mode)
        return Scorable(id=str(data.get("id", "")), text=text, target_type=target_type)

    @staticmethod
    def from_text(text: str, target_type: TargetType) -> Scorable:
        """Convert plain text to Scorable. Supports summary truncation."""
        return Scorable(id="", text=text, target_type=target_type)

    @staticmethod
    def from_plan_trace(trace: PlanTrace, goal_text: str, mode: str = "default", step: Optional[ExecutionStep] = None) -> Scorable:
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
            step_text += f"Description: {step.description or 'No description'}\n"
            
            # Add input if available
            if hasattr(step, 'input_text') and step.input_text:
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
                target_type=TargetType.PLAN_TRACE_STEP
            )
        
        elif mode == "full_trace":
            # Format the complete trace for scoring
            trace_text = f"Goal: {goal_text or 'No goal text'}\n\n"
            trace_text += "Pipeline Execution Steps:\n\n"
            
            # Add all steps
            for i, step in enumerate(trace.execution_steps, 1):
                trace_text += f"{i}. {step.step_type}: {step.description[:100]}...\n"
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
                target_type=TargetType.PLAN_TRACE
            )
        
        else:
            # Default mode - goal + final output
            final_output = trace.final_output_text or ""
            
            return Scorable(
                id=trace.trace_id,
                text=f"{goal_text}\n\n{final_output}",
                target_type=TargetType.PLAN_TRACE
            )
        

    @staticmethod
    def from_id(memory, scorable_type: str, scorable_id: str):
        """
        Resolve a scorable object from memory by its type and id,
        returning a Scorable. Raises ValueError if unsupported or not found.
        """

        # Dispatch table mapping target_type -> (getter_function, cast_type)
        dispatch = {
            TargetType.CONVERSATION: (memory.chats.get_conversation, int),
            TargetType.CONVERSATION_TURN: (memory.chats.get_turn_by_id, int),
            TargetType.CONVERSATION_MESSAGE: (memory.chats.get_message_by_id, int),
            TargetType.DOCUMENT: (memory.documents.get_by_id, int),
            TargetType.HYPOTHESIS: (memory.hypotheses.get_by_id, int),
            TargetType.CARTRIDGE: (memory.cartridges.get_by_id, int),
            TargetType.TRIPLE: (memory.cartridge_triples.get_by_id, int),
            TargetType.PROMPT: (memory.prompts.get_by_id, int), 
            TargetType.THEOREM: (memory.theorems.get_by_id, int),  # plural fixed
            TargetType.CASE: (memory.casebooks.get_case_by_id, int),
            TargetType.CASE_SCORABLE: (memory.casebooks.get_case_scorable_by_id, int),
            TargetType.PLAN_TRACE: (memory.plan_traces.get_by_id, str),
            TargetType.PLAN_TRACE_STEP: (memory.plan_traces.get_step_by_id, str),
        }

        if scorable_type not in dispatch:
            raise ValueError(f"Unsupported target type for text extraction: {scorable_type}")

        getter, caster = dispatch[scorable_type]
        orm = getter(caster(scorable_id))

        if orm is None:
            raise ValueError(f"No object found for {scorable_type} id={scorable_id}")

        return ScorableFactory.from_orm(orm)
