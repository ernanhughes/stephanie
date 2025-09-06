# stephanie/scoring/scorable.py
from typing import Any, Dict


class Scorable:
    def __init__(self, text: str, id: str = "", target_type: str = "custom", meta: Dict[str, Any] = None):
        self._id = id
        self._text = text
        self._target_type = target_type
        self._metadata = meta or {}

    @property
    def meta(self) -> Dict[str, Any]:
        return self._metadata


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
            "metadata": self._metadata  
        }

    def __repr__(self):
        preview = self._text[:30].replace("\n", " ")
        return (
            f"Scorable(id='{self._id}', "
            f"target_type='{self._target_type}', "
            f"text_preview='{preview}...')"
        )
