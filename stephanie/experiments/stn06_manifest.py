"""Frozen STN06 held-out task manifest.

Eight independent, self-contained Python tasks are specified here as plain data.
Each entry carries three disjoint file sets:

* ``workspace_files`` — the initial repository both experimental arms receive.
* ``reference_solution`` — the committed correct change. It is SEALED and never
  placed in an agent-facing workspace. It exists only so the independent oracle
  can be validated before any paid run.
* ``oracle_files`` — the sealed independent evaluator. Also never placed in an
  agent-facing workspace. Hashed before any paid run so the oracle cannot be
  altered after results are known.

The manifest is intentionally small, deterministic and dependency-free so its
hash is reproducible across processes and machines. It is not a benchmark; it
is the frozen input to an exploratory paired experiment.

No model calls happen in this module.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# --- Task classes -----------------------------------------------------------

NEAR_TRANSFER = "near_transfer"
COMPLEX = "complex"
UNRELATED = "unrelated"

TASK_CLASS_COUNTS = {
    NEAR_TRANSFER: 4,
    COMPLEX: 2,
    UNRELATED: 2,
}


# --- Held-out fixtures ------------------------------------------------------

HELD_OUT_TASKS: list[dict[str, Any]] = [
    {
        "task_id": "stn06-near-01",
        "task_class": NEAR_TRANSFER,
        "objective": (
            "The module mathops.py has a defect: subtract() returns the wrong "
            "result. A unit test in tests/test_mathops.py currently fails. Make "
            "the smallest correct change to mathops.py so the test suite passes."
        ),
        "workspace_files": {
            "mathops.py": (
                "def add(a, b):\n"
                "    return a + b\n"
                "\n"
                "\n"
                "def subtract(a, b):\n"
                "    return a + b\n"
            ),
            "tests/test_mathops.py": (
                "import unittest\n"
                "\n"
                "from mathops import add, subtract\n"
                "\n"
                "\n"
                "class MathOpsTest(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(add(2, 3), 5)\n"
                "\n"
                "    def test_subtract(self):\n"
                "        self.assertEqual(subtract(5, 3), 2)\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "reference_solution": {
            "mathops.py": (
                "def add(a, b):\n"
                "    return a + b\n"
                "\n"
                "\n"
                "def subtract(a, b):\n"
                "    return a - b\n"
            ),
        },
        "oracle_command": "python3 -m unittest discover -s . -p 'test_*.py'",
        "oracle_files": {
            "test_mathops_oracle.py": (
                "import unittest\n"
                "\n"
                "from mathops import add, subtract\n"
                "\n"
                "\n"
                "class MathOpsOracle(unittest.TestCase):\n"
                "    def test_add_identity(self):\n"
                "        self.assertEqual(add(0, 0), 0)\n"
                "        self.assertEqual(add(-1, 1), 0)\n"
                "\n"
                "    def test_subtract_is_not_add(self):\n"
                "        self.assertEqual(subtract(5, 3), 2)\n"
                "        self.assertEqual(subtract(3, 5), -2)\n"
                "        self.assertEqual(subtract(7, 7), 0)\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "expected_exit_code": 0,
        "expected_modified_paths": ["mathops.py"],
        "prohibited_paths": ["tests/"],
    },
    {
        "task_id": "stn06-near-02",
        "task_class": NEAR_TRANSFER,
        "objective": (
            "The module ranges.py has an off-by-one defect: inclusive_range(n) "
            "is meant to return the integers 1..n inclusive. A unit test in "
            "tests/test_ranges.py currently fails. Fix ranges.py only."
        ),
        "workspace_files": {
            "ranges.py": (
                "def inclusive_range(n):\n"
                "    \"\"\"Return the integers 1..n inclusive.\"\"\"\n"
                "    return list(range(1, n))\n"
            ),
            "tests/test_ranges.py": (
                "import unittest\n"
                "\n"
                "from ranges import inclusive_range\n"
                "\n"
                "\n"
                "class RangeTest(unittest.TestCase):\n"
                "    def test_inclusive(self):\n"
                "        self.assertEqual(inclusive_range(3), [1, 2, 3])\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "reference_solution": {
            "ranges.py": (
                "def inclusive_range(n):\n"
                "    \"\"\"Return the integers 1..n inclusive.\"\"\"\n"
                "    return list(range(1, n + 1))\n"
            ),
        },
        "oracle_command": "python3 -m unittest discover -s . -p 'test_*.py'",
        "oracle_files": {
            "test_ranges_oracle.py": (
                "import unittest\n"
                "\n"
                "from ranges import inclusive_range\n"
                "\n"
                "\n"
                "class RangeOracle(unittest.TestCase):\n"
                "    def test_one(self):\n"
                "        self.assertEqual(inclusive_range(1), [1])\n"
                "\n"
                "    def test_zero(self):\n"
                "        self.assertEqual(inclusive_range(0), [])\n"
                "\n"
                "    def test_five(self):\n"
                "        self.assertEqual(inclusive_range(5), [1, 2, 3, 4, 5])\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "expected_exit_code": 0,
        "expected_modified_paths": ["ranges.py"],
        "prohibited_paths": ["tests/"],
    },
    {
        "task_id": "stn06-near-03",
        "task_class": NEAR_TRANSFER,
        "objective": (
            "textcase.py should convert a human label such as 'Hello World' into "
            "'hello_world'. The current implementation only lowercases the text, "
            "so tests/test_textcase.py fails. Fix textcase.py only."
        ),
        "workspace_files": {
            "textcase.py": (
                "def to_snake(label):\n"
                "    return label.lower()\n"
            ),
            "tests/test_textcase.py": (
                "import unittest\n"
                "\n"
                "from textcase import to_snake\n"
                "\n"
                "\n"
                "class TextCaseTest(unittest.TestCase):\n"
                "    def test_snake(self):\n"
                "        self.assertEqual(to_snake('Hello World'), 'hello_world')\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "reference_solution": {
            "textcase.py": (
                "def to_snake(label):\n"
                "    return '_'.join(label.lower().split())\n"
            ),
        },
        "oracle_command": "python3 -m unittest discover -s . -p 'test_*.py'",
        "oracle_files": {
            "test_textcase_oracle.py": (
                "import unittest\n"
                "\n"
                "from textcase import to_snake\n"
                "\n"
                "\n"
                "class TextCaseOracle(unittest.TestCase):\n"
                "    def test_single(self):\n"
                "        self.assertEqual(to_snake('Hello'), 'hello')\n"
                "\n"
                "    def test_multi(self):\n"
                "        self.assertEqual(to_snake('A B C'), 'a_b_c')\n"
                "\n"
                "    def test_collapse(self):\n"
                "        self.assertEqual(to_snake('A  B'), 'a_b')\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "expected_exit_code": 0,
        "expected_modified_paths": ["textcase.py"],
        "prohibited_paths": ["tests/"],
    },
    {
        "task_id": "stn06-near-04",
        "task_class": NEAR_TRANSFER,
        "objective": (
            "clamp.py should constrain a value into the inclusive range "
            "[low, high]. The current implementation returns the wrong bound "
            "when the value is outside the range, so tests/test_clamp.py fails. "
            "Fix clamp.py only."
        ),
        "workspace_files": {
            "clamp.py": (
                "def clamp(value, low, high):\n"
                "    return min(low, max(value, high))\n"
            ),
            "tests/test_clamp.py": (
                "import unittest\n"
                "\n"
                "from clamp import clamp\n"
                "\n"
                "\n"
                "class ClampTest(unittest.TestCase):\n"
                "    def test_below(self):\n"
                "        self.assertEqual(clamp(-5, 0, 10), 0)\n"
                "\n"
                "    def test_above(self):\n"
                "        self.assertEqual(clamp(50, 0, 10), 10)\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "reference_solution": {
            "clamp.py": (
                "def clamp(value, low, high):\n"
                "    return max(low, min(value, high))\n"
            ),
        },
        "oracle_command": "python3 -m unittest discover -s . -p 'test_*.py'",
        "oracle_files": {
            "test_clamp_oracle.py": (
                "import unittest\n"
                "\n"
                "from clamp import clamp\n"
                "\n"
                "\n"
                "class ClampOracle(unittest.TestCase):\n"
                "    def test_inside(self):\n"
                "        self.assertEqual(clamp(5, 0, 10), 5)\n"
                "\n"
                "    def test_low(self):\n"
                "        self.assertEqual(clamp(0, 0, 10), 0)\n"
                "\n"
                "    def test_high(self):\n"
                "        self.assertEqual(clamp(10, 0, 10), 10)\n"
                "\n"
                "    def test_negative_range(self):\n"
                "        self.assertEqual(clamp(-3, -10, -1), -3)\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "expected_exit_code": 0,
        "expected_modified_paths": ["clamp.py"],
        "prohibited_paths": ["tests/"],
    },
    {
        "task_id": "stn06-complex-01",
        "task_class": COMPLEX,
        "objective": (
            "records.py defines a Record type and to_totals(records) which should "
            "sum each record's amount, skipping records flagged as void. "
            "summary.py renders a report that relies on to_totals. A boundary "
            "defect makes an empty input raise instead of reporting a zero total, "
            "and void records are still counted. Identify the responsible "
            "function across the two modules and fix it."
        ),
        "workspace_files": {
            "records.py": (
                "class Record:\n"
                "    def __init__(self, amount, void=False):\n"
                "        self.amount = amount\n"
                "        self.void = void\n"
                "\n"
                "\n"
                "def to_totals(records):\n"
                "    total = records[0].amount\n"
                "    for record in records[1:]:\n"
                "        total += record.amount\n"
                "    return total\n"
            ),
            "summary.py": (
                "from records import to_totals\n"
                "\n"
                "\n"
                "def render(records):\n"
                "    return 'total={0}'.format(to_totals(records))\n"
            ),
            "tests/test_summary.py": (
                "import unittest\n"
                "\n"
                "from records import Record\n"
                "from summary import render\n"
                "\n"
                "\n"
                "class SummaryTest(unittest.TestCase):\n"
                "    def test_render(self):\n"
                "        self.assertEqual(render([Record(2), Record(3)]), 'total=5')\n"
                "\n"
                "    def test_empty(self):\n"
                "        self.assertEqual(render([]), 'total=0')\n"
                "\n"
                "    def test_void_skipped(self):\n"
                "        self.assertEqual(render([Record(2), Record(9, void=True)]), 'total=2')\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "reference_solution": {
            "records.py": (
                "class Record:\n"
                "    def __init__(self, amount, void=False):\n"
                "        self.amount = amount\n"
                "        self.void = void\n"
                "\n"
                "\n"
                "def to_totals(records):\n"
                "    total = 0\n"
                "    for record in records:\n"
                "        if record.void:\n"
                "            continue\n"
                "        total += record.amount\n"
                "    return total\n"
            ),
        },
        "oracle_command": "python3 -m unittest discover -s . -p 'test_*.py'",
        "oracle_files": {
            "test_summary_oracle.py": (
                "import unittest\n"
                "\n"
                "from records import Record, to_totals\n"
                "from summary import render\n"
                "\n"
                "\n"
                "class SummaryOracle(unittest.TestCase):\n"
                "    def test_all_void(self):\n"
                "        self.assertEqual(to_totals([Record(4, void=True)]), 0)\n"
                "\n"
                "    def test_negative(self):\n"
                "        self.assertEqual(render([Record(5), Record(-2)]), 'total=3')\n"
                "\n"
                "    def test_mixed(self):\n"
                "        self.assertEqual(\n"
                "            to_totals([Record(1), Record(2, void=True), Record(3)]), 4\n"
                "        )\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "expected_exit_code": 0,
        "expected_modified_paths": ["records.py"],
        "prohibited_paths": ["tests/"],
    },
    {
        "task_id": "stn06-complex-02",
        "task_class": COMPLEX,
        "objective": (
            "traffic.py implements a three-state traffic light. next_state() "
            "should advance green->yellow->red->green. The documented contract "
            "(see the docstring) also requires that an unknown state is treated "
            "as red. A state-table defect breaks the unknown-state rule. Fix "
            "traffic.py so the documented contract holds."
        ),
        "workspace_files": {
            "traffic.py": (
                "ORDER = ['green', 'yellow', 'red']\n"
                "\n"
                "\n"
                "def next_state(state):\n"
                "    \"\"\"Return the next state, or 'red' for an unknown state.\"\"\"\n"
                "    return ORDER[(ORDER.index(state) + 1) % len(ORDER)]\n"
            ),
            "tests/test_traffic.py": (
                "import unittest\n"
                "\n"
                "from traffic import next_state\n"
                "\n"
                "\n"
                "class TrafficTest(unittest.TestCase):\n"
                "    def test_cycle(self):\n"
                "        self.assertEqual(next_state('green'), 'yellow')\n"
                "        self.assertEqual(next_state('yellow'), 'red')\n"
                "        self.assertEqual(next_state('red'), 'green')\n"
                "\n"
                "    def test_unknown(self):\n"
                "        self.assertEqual(next_state('purple'), 'red')\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "reference_solution": {
            "traffic.py": (
                "ORDER = ['green', 'yellow', 'red']\n"
                "\n"
                "\n"
                "def next_state(state):\n"
                "    \"\"\"Return the next state, or 'red' for an unknown state.\"\"\"\n"
                "    if state not in ORDER:\n"
                "        return 'red'\n"
                "    return ORDER[(ORDER.index(state) + 1) % len(ORDER)]\n"
            ),
        },
        "oracle_command": "python3 -m unittest discover -s . -p 'test_*.py'",
        "oracle_files": {
            "test_traffic_oracle.py": (
                "import unittest\n"
                "\n"
                "from traffic import next_state\n"
                "\n"
                "\n"
                "class TrafficOracle(unittest.TestCase):\n"
                "    def test_from_unknown(self):\n"
                "        self.assertEqual(next_state('unknown'), 'red')\n"
                "        self.assertEqual(next_state(''), 'red')\n"
                "\n"
                "    def test_cycle_intact(self):\n"
                "        state = 'green'\n"
                "        seen = [state]\n"
                "        for _ in range(3):\n"
                "            state = next_state(state)\n"
                "            seen.append(state)\n"
                "        self.assertEqual(seen, ['green', 'yellow', 'red', 'green'])\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "expected_exit_code": 0,
        "expected_modified_paths": ["traffic.py"],
        "prohibited_paths": ["tests/"],
    },
    {
        "task_id": "stn06-unrelated-01",
        "task_class": UNRELATED,
        "objective": (
            "Create a new module newutil.py containing a function "
            "slugify(text) that lowercases the text, strips leading and trailing "
            "whitespace, and replaces every run of whitespace with a single "
            "hyphen. Include a one-line docstring. Do not modify any other file."
        ),
        "workspace_files": {
            "README.md": (
                "# Utility task\n"
                "\n"
                "Add a new standalone module `newutil.py` as described in the "
                "task. The repository currently has no other code.\n"
            ),
        },
        "reference_solution": {
            "newutil.py": (
                "def slugify(text):\n"
                "    \"\"\"Return a lowercase hyphen-separated slug for text.\"\"\"\n"
                "    return '-'.join(text.strip().lower().split())\n"
            ),
        },
        "oracle_command": "python3 -m unittest discover -s . -p 'test_*.py'",
        "oracle_files": {
            "test_newutil_oracle.py": (
                "import unittest\n"
                "\n"
                "from newutil import slugify\n"
                "\n"
                "\n"
                "class SlugifyOracle(unittest.TestCase):\n"
                "    def test_basic(self):\n"
                "        self.assertEqual(slugify('Hello World'), 'hello-world')\n"
                "\n"
                "    def test_trim_collapse(self):\n"
                "        self.assertEqual(slugify('  A   B  '), 'a-b')\n"
                "\n"
                "    def test_single(self):\n"
                "        self.assertEqual(slugify('Title'), 'title')\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "expected_exit_code": 0,
        "expected_modified_paths": ["newutil.py"],
        "prohibited_paths": ["README.md"],
    },
    {
        "task_id": "stn06-unrelated-02",
        "task_class": UNRELATED,
        "objective": (
            "Create a new module stats.py containing a function median(values) "
            "that returns the statistical median of a non-empty list of numbers, "
            "averaging the two central values for an even-length list. Include a "
            "docstring. Do not modify any other file."
        ),
        "workspace_files": {
            "README.md": (
                "# Statistics task\n"
                "\n"
                "Add a new standalone module `stats.py` as described in the task.\n"
            ),
        },
        "reference_solution": {
            "stats.py": (
                "def median(values):\n"
                "    \"\"\"Return the median of a non-empty list of numbers.\"\"\"\n"
                "    ordered = sorted(values)\n"
                "    size = len(ordered)\n"
                "    middle = size // 2\n"
                "    if size % 2 == 1:\n"
                "        return ordered[middle]\n"
                "    return (ordered[middle - 1] + ordered[middle]) / 2\n"
            ),
        },
        "oracle_command": "python3 -m unittest discover -s . -p 'test_*.py'",
        "oracle_files": {
            "test_stats_oracle.py": (
                "import unittest\n"
                "\n"
                "from stats import median\n"
                "\n"
                "\n"
                "class MedianOracle(unittest.TestCase):\n"
                "    def test_odd(self):\n"
                "        self.assertEqual(median([3, 1, 2]), 2)\n"
                "\n"
                "    def test_even(self):\n"
                "        self.assertEqual(median([1, 2, 3, 4]), 2.5)\n"
                "\n"
                "    def test_single(self):\n"
                "        self.assertEqual(median([7]), 7)\n"
                "\n"
                "    def test_unsorted(self):\n"
                "        self.assertEqual(median([10, 2, 8, 4]), 6)\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "expected_exit_code": 0,
        "expected_modified_paths": ["stats.py"],
        "prohibited_paths": ["README.md"],
    },
]


# --- Hashing ----------------------------------------------------------------

def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_files(files: dict[str, str]) -> str:
    payload = {path: hash_text(content) for path, content in sorted(files.items())}
    return hash_text(_canonical(payload))


def task_hash(task: dict[str, Any]) -> str:
    stable = {
        "task_id": task["task_id"],
        "task_class": task["task_class"],
        "objective": task["objective"],
        "workspace": hash_files(task["workspace_files"]),
        "reference": hash_files(task["reference_solution"]),
        "oracle": hash_files(task["oracle_files"]),
        "oracle_command": task["oracle_command"],
        "expected_exit_code": task["expected_exit_code"],
        "expected_modified_paths": sorted(task["expected_modified_paths"]),
        "prohibited_paths": sorted(task["prohibited_paths"]),
    }
    return hash_text(_canonical(stable))


def manifest(include_files: bool = False) -> dict[str, Any]:
    """Return the frozen manifest with per-task and aggregate hashes.

    ``include_files`` embeds the literal fixture contents (workspace, sealed
    reference solution and sealed oracle) so the preflight report is
    self-contained. The hash is identical either way.
    """
    tasks = []
    for task in HELD_OUT_TASKS:
        entry = {
            "task_id": task["task_id"],
            "task_class": task["task_class"],
            "objective": task["objective"],
            "objective_hash": hash_text(task["objective"]),
            "workspace_hash": hash_files(task["workspace_files"]),
            "reference_hash": hash_files(task["reference_solution"]),
            "oracle_hash": hash_files(task["oracle_files"]),
            "oracle_command": task["oracle_command"],
            "expected_exit_code": task["expected_exit_code"],
            "expected_modified_paths": sorted(task["expected_modified_paths"]),
            "prohibited_paths": sorted(task["prohibited_paths"]),
            "task_hash": task_hash(task),
            "workspace_file_paths": sorted(task["workspace_files"].keys()),
            "oracle_file_paths": sorted(task["oracle_files"].keys()),
            "reference_file_paths": sorted(task["reference_solution"].keys()),
        }
        if include_files:
            entry["workspace_files"] = dict(task["workspace_files"])
            entry["reference_solution"] = dict(task["reference_solution"])
            entry["oracle_files"] = dict(task["oracle_files"])
        tasks.append(entry)
    counts: dict[str, int] = {}
    for task in HELD_OUT_TASKS:
        counts[task["task_class"]] = counts.get(task["task_class"], 0) + 1
    return {
        "manifest_hash": hash_text(_canonical([entry["task_hash"] for entry in tasks])),
        "task_count": len(tasks),
        "task_class_counts": counts,
        "tasks": tasks,
    }


def task_by_id(task_id: str) -> dict[str, Any] | None:
    return next((task for task in HELD_OUT_TASKS if task["task_id"] == task_id), None)
