# stephanie/scoring/scorable_factory.py
from enum import Enum as PyEnum

from typing import Optional
from stephanie.data.plan_trace import ExecutionStep, PlanTrace
from stephanie.models.cartridge_triple import CartridgeTripleORM
from stephanie.models.document import DocumentORM
from stephanie.models.prompt import PromptORM
from stephanie.models.theorem import CartridgeORM, TheoremORM
from stephanie.scoring.scorable import Scorable


# Enum defining all the supported types of scoreable targets
class TargetType:
    DOCUMENT = "document"
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
    """ Why am I hitting the cash It shouldn't be slamming the cash by now
    Factory for turning various content types into unified Scorable objects.
    """

    @staticmethod
    def from_orm(obj, mode: str = "default") -> Scorable:
        """
        Convert an ORM object to a Scorable.
        Dispatches based on the object's class type.
        """
        if isinstance(obj, PromptORM):
            return ScorableFactory.from_prompt_pair(obj, mode)
        elif isinstance(obj, CartridgeORM):
            return Scorable(
                id=obj.id, text=obj.markdown_content, target_type=TargetType.CARTRIDGE
            )
        elif isinstance(obj, CartridgeTripleORM):
            # For a triple, we concatenate subject, relation, and object as a textual representation
            return Scorable(
                id=obj.id,
                text=f"{obj.subject} {obj.relation} {obj.object}",
                target_type=TargetType.TRIPLE,
            )
        elif isinstance(obj, TheoremORM):
            return Scorable(
                id=obj.id, text=obj.statement, target_type=TargetType.THEOREM
            )
        elif isinstance(obj, DocumentORM):
            title = obj.title or ""
            summary = obj.summary or ""
            content = obj.content or ""

            if title and summary:
                text = f"#Title\n{title}\n\n## Summary\n{summary}"
            elif content:
                text = content
            else:
                text = title or summary  # fallback if only one exists

            return Scorable(id=obj.id, text=text, target_type=TargetType.DOCUMENT)
        else:
            raise ValueError(f"Unsupported ORM type for scoring: {type(obj)}")

    @staticmethod
    def from_prompt_pair(obj: PromptORM, mode: str = "prompt+response") -> Scorable:
        """
        Handles PromptORM objects that contain both prompt and response.
        The `mode` parameter controls whether to extract only the prompt, only the response,
        or a concatenated version of both.
        """
        prompt = obj.prompt or ""
        response = obj.response or ""
        target_type = TargetType.PROMPT

        if mode == "prompt_only":
            text = prompt
        elif mode == "response_only":
            text = response
            target_type = TargetType.RESPONSE
        elif mode == "prompt+response":
            text = f"{prompt}\n\n{response}"
            target_type = TargetType.PROMPT_RESPONSE
        else:
            raise ValueError(f"Invalid prompt scoring mode: {mode}")

        return Scorable(id=obj.id, text=text, target_type=target_type)

    @staticmethod
    def from_dict(data: dict, target_type: TargetType = None) -> Scorable:
        """
        Converts a plain dictionary into a Scorable, using optional fields like
        title, summary, and content for DOCUMENT types.
        """
        if target_type is None:
            target_type = data.get("target_type", "document")
        if "text" in data: # If text is provided, use it directly
            return Scorable(id=str(data.get("id", "")), text=data["text"], target_type=target_type)
        if target_type == "document":
            title = data.get("title", "")
            summary = data.get("summary", "")
            content = data.get("content", "")
            if title and summary:
                text = f"#Title\n{title}\n\n## Summary\n{summary}"
            elif content:
                text = content
            else:
                text = title or summary
        elif target_type == "triple":
            text = (
                f"{data.get('subject')} {data.get('relation')} {data.get('object')}",
            )
        else:
            text = data.get("text", "")

        return Scorable(id=str(data.get("id")), text=text, target_type=target_type)


    @staticmethod
    def from_text(text: str, target_type: TargetType) -> Scorable:
        """
        Converts a plain dictionary into a Scorable, using optional fields like
        title, summary, and content for DOCUMENT types.
        """
        return Scorable(id="", text=text, target_type=target_type)

    @staticmethod
    def from_plan_trace(trace: PlanTrace, mode: str = "default", step: Optional[ExecutionStep] = None) -> Scorable:
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
            trace_text = f"Goal: {trace.goal_text or 'No goal text'}\n\n"
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
            goal_text = trace.goal_text or ""
            final_output = trace.final_output_text or ""
            
            return Scorable(
                id=trace.trace_id,
                text=f"{goal_text}\n\n{final_output}",
                target_type=TargetType.PLAN_TRACE
            )
        

    @staticmethod
    def from_id(memory, target_type: str, target_id: str) -> str:
        """
        Extracts the text content from a Scorable object.
        """ 
        orm = None
        if target_type == TargetType.DOCUMENT:
            orm = memory.document.get_by_id(target_id)
        elif target_type == TargetType.HYPOTHESIS:
            orm = memory.hypothesis.get_by_id(target_id)
        elif target_type == TargetType.CARTRIDGE:
            orm = memory.cartridge.get_by_id(target_id)
        elif target_type == TargetType.TRIPLE:
            orm = memory.triple.get_by_id(target_id)
        elif target_type == TargetType.PROMPT:
            orm = memory.prompt.get_by_id(target_id)
        elif target_type == TargetType.THEOREM:
            orm = memory.theorem.get_by_id(target_id)
        elif target_type == TargetType.PLAN_TRACE:
            orm = memory.plan_trace.get_by_id(target_id)
        if orm is not None:
            scorable = ScorableFactory.from_orm(orm)
            return scorable
        else:
            raise ValueError(f"Unsupported target type for text extraction: {target_type}")
