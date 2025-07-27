# stephanie/scoring/scorable_factory.py
from enum import Enum as PyEnum

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
    REFINEMENT = "refinement"  # For SRFT-style usage


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
