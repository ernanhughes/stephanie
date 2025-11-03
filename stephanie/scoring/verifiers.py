# stephanie/scoring/verifiers.py
from __future__ import annotations

import contextlib
import io
import re
import traceback
from typing import Any, Dict, List


def code_verifier(prompt: str, response: str, meta: Dict[str, Any] = None) -> float:
    """
    Verifier for code-based tasks.
    
    Args:
        prompt (str): The original coding question (not directly used, but for compatibility).
        response (str): The model's generated code as text.
        meta (dict, optional): Metadata with test cases, e.g.:
            {
                "tests": [
                    {"input": (2, 3), "expected": 5},
                    {"input": (10, -1), "expected": 9}
                ],
                "entrypoint": "add_numbers"   # function name expected in response
            }
    
    Returns:
        float: Score in [0.0, 1.0], fraction of passed test cases.
    """
    if not response or not meta:
        return 0.0

    tests: List[Dict] = meta.get("tests", [])
    entrypoint: str = meta.get("entrypoint")

    # Create a local namespace for execution
    local_ns = {}

    try:
        # Capture stdout during execution (safety)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(response, {}, local_ns)
    except Exception:
        traceback.print_exc()
        return 0.0

    if entrypoint not in local_ns:
        # Entrypoint function not found
        return 0.0

    fn = local_ns[entrypoint]
    if not callable(fn):
        return 0.0

    passed = 0
    for test in tests:
        try:
            args = test.get("input", ())
            if not isinstance(args, tuple):
                args = (args,)
            expected = test.get("expected")

            result = fn(*args)
            if result == expected:
                passed += 1
        except Exception:
            traceback.print_exc()
            continue

    if not tests:
        return 0.5  # no tests provided, neutral score
    return passed / len(tests)


def boxed_math_verifier(prompt: str, response: str, meta: Dict[str, Any] = None) -> float:
    """
    Verifier for math problems with answers written as \\boxed{...}.
    
    Args:
        prompt (str): The original question (not used directly, but kept for compatibility).
        response (str): The model's generated output.
        meta (dict, optional): Metadata that may contain the expected answer.
            Example: {"expected": "42"}
    
    Returns:
        float: 1.0 if correct, 0.0 if wrong, 0.5 if unable to verify.
    """
    if not response:
        return 0.0

    # Extract content inside \boxed{ ... }
    match = re.search(r"\\boxed\{([^}]*)\}", response)
    if not match:
        # No boxed answer found — return partial credit
        return 0.5

    predicted = match.group(1).strip()

    # If expected answer is provided in metadata
    expected = (meta or {}).get("expected")
    if expected is None:
        # No ground truth — just confirm that an answer was boxed
        return 1.0 if predicted else 0.5

    # Normalize (remove extra spaces, maybe leading zeros)
    predicted_norm = predicted.replace(" ", "")
    expected_norm = str(expected).replace(" ", "")

    # Check for equality
    return 1.0 if predicted_norm == expected_norm else 0.0
