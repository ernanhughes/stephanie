# stephanie/tools/spec_from_document.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

TEMPLATES = [
    (
        r"\bsoftmax\b",
        {
            "function_name": "softmax",
            "description": "Softmax over last dim",
            "equations": ["exp(x)/sum(exp(x))"],
            "input_shape": [4],
            "output_shape": [4],
            "golden_input": [1.0, 2.0, 3.0, 4.0],
            "golden_output": [0.032, 0.087, 0.236, 0.645],
        },
    ),
    (
        r"\bl2[-\s]?norm(aliz(e|ation))?\b",
        {
            "function_name": "l2_normalize",
            "description": "Normalize to unit L2 norm",
            "equations": ["x/||x||_2"],
            "input_shape": [5],
            "output_shape": [5],
            "golden_input": [3.0, 4.0],
            "golden_output": [0.6, 0.8],
        },
    ),
]


def build_spec_from_text(text: str) -> Dict[str, Any]:
    low = text.lower()
    for pat, spec in TEMPLATES:
        if re.search(pat, low):
            return spec
    return {
        "function_name": "identity",
        "description": "Return input",
        "equations": [],
        "input_shape": [2],
        "output_shape": [2],
        "golden_input": [1.0, 2.0],
        "golden_output": 3.0,
    }


def save_spec(spec: Dict[str, Any], out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(spec, indent=2))
    return out_path
