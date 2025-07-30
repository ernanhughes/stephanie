# stephanie/scoring/scorable_factory.py
from enum import Enum as PyEnum

from stephanie.data.plan_trace import PlanTrace
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
            return Scorable(id=data.get("id", ""), text=data["text"], target_type=target_type)
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

        return Scorable(id=data.get("id"), text=text, target_type=target_type)


    @staticmethod
    def from_text(text: str, target_type: TargetType) -> Scorable:
        """
        Converts a plain dictionary into a Scorable, using optional fields like
        title, summary, and content for DOCUMENT types.
        """
        return Scorable(id="", text=text, target_type=target_type)

    @staticmethod
    def from_plan_trace(trace: PlanTrace, mode: str = "default") -> Scorable:
        """
        Convert a PlanTrace into a Scorable object for scoring.
        Mode can be used to customize how the trace is represented as text.
        """
        if mode == "default":
            text = trace.goal_text + "\n\n" + trace.final_output_text
        elif mode == "full_trace":
            step_texts = "\n".join([f"[{s.step_id}] {s.output_text}" for s in trace.execution_steps])
            text = f"Goal: {trace.goal_text}\n\nSteps:\n{step_texts}\n\nFinal: {trace.final_output_text}"
        else:
            raise ValueError(f"Unsupported PlanTrace scoring mode: {mode}")
        
        return Scorable(
            id=trace.trace_id,
            text=text,
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
