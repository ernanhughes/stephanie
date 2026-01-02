from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SpineConfig:
    """
    Configuration for spine construction.
    """
    page_image_root: str
    max_pages: int | None = None
    min_figure_area_frac: float = 0.02

    enable_docling: bool = True
    enable_figures: bool = True
    enable_ocr: bool = False

    dump_enabled: bool = True
    signals_enabled: bool = True

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "SpineConfig":
        return cls(**cfg)
