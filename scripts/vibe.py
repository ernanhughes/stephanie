### `vibe_console.py`

import os
import argparse
import textwrap
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
import os
from typing import Sequence

REPORT_PATH = "vibe_out/codecheck_report.txt"


def run_vibe_batch(file_paths: Sequence[str], report_path: str = REPORT_PATH) -> None:
    """Run VibeThinker over a list of files and write one combined report."""
    tok, model, device = load_model()

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    lines: list[str] = []
    lines.append(f"[info] Model: {MODEL_NAME}")
    lines.append(f"[info] Device: {device}")
    lines.append(f"[info] Files analyzed: {len(file_paths)}")
    lines.append("=" * 80)

    for idx, fp in enumerate(file_paths, start=1):
        lines.append("\n" + "=" * 80)
        lines.append(f"[{idx}/{len(file_paths)}] File: {fp}")
        lines.append("=" * 80)

        suggestion_text = run_on_file(tok, model, device, fp)

        lines.append("\n--- VibeThinker Suggestions ---\n")
        lines.append(suggestion_text)

    report = "\n".join(lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[info] Wrote combined report to {report_path}")


def load_model() -> tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Loading {MODEL_NAME} on device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,
        torch_dtype="bfloat16",   # matches their example
        device_map="auto",        # let HF/vLLM shard if needed
        trust_remote_code=True,
    )

    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    # If for some reason device_map="auto" doesn’t put it on CUDA for a single-GPU box:
    if device != "cuda":
        model.to(device)

    return tok, model, device

# -----------------------------
# Configuration
# -----------------------------

MODEL_NAME = "WeiboAI/VibeThinker-1.5B"
MAX_NEW_TOKENS = 1024
MAX_CODE_LINES = 1200

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".idea",
    ".mypy_cache",
}

SYSTEM_PROMPT = """
You are a senior Python code improver helping a human developer keep a large, messy codebase healthy.

Your job:
- Look at the provided Python code.
- Suggest small, concrete improvements that can be done in ~5 minutes each.
- Focus on:
  * splitting overly large functions or files,
  * renaming confusing symbols,
  * adding/fixing docstrings,
  * removing obvious dead code,
  * making future changes safer and clearer.

Rules:
- You NEVER explain how you are thinking.
- You NEVER comment about how much code you see or whether the file is complete.
- You ALWAYS treat the provided snippet as enough to make useful suggestions.
- You NEVER repeat the task description or restate the problem.
- You NEVER output JSON, XML, or any machine-readable markup.
- You NEVER use <think> or similar tags.

Format:
- Use short headings:

  Suggestion 1:
  - ...

  Suggestion 2:
  - ...

- If you show code, use fenced code blocks:

  Before:
  ```python
  # old code
  ```

After:

  ```python
  # improved code
  ```

Keep everything focused, short, and easy to apply by hand.
""".strip()

# -----------------------------

# Utility: find Python files

# -----------------------------


def find_python_files(root: str, limit: int) -> List[str]:
    """
    Walk `root` and return up to `limit` .py files, sorted by mtime (newest first).
    """
    candidates: List[Tuple[float, str]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # prune unwanted dirs
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                mtime = os.path.getmtime(fpath)
            except OSError:
                continue
            candidates.append((mtime, fpath))

    # newest first
    candidates.sort(key=lambda t: t[0], reverse=True)
    return [path for _, path in candidates[:limit]]


# -----------------------------

# Utility: basic metrics

# -----------------------------


def basic_metrics(content: str) -> Dict[str, float]:
    """
    Very cheap, approximate metrics just to give the model a bit of context.
    """
    lines = content.splitlines()
    loc = len(lines)
    num_defs = content.count("def ")
    num_classes = content.count("class ")
    return {
        "loc": float(loc),
        "num_defs": float(num_defs),
        "num_classes": float(num_classes),
    }


def format_metrics_for_prompt(metrics: Dict[str, float]) -> str:
    if not metrics:
        return "(none)"
    return ", ".join(f"{k}={v:.2f}" for k, v in sorted(metrics.items()))


def format_issues_for_prompt(issues: List[Dict[str, Any]]) -> str:
    """
    Placeholder: you can wire real linter/static results later.
    For now, we just say there are no known issues.
    """
    if not issues:
        return "(no static issues recorded)"
    lines = []
    for iss in issues[:10]:
        lines.append(
            f"- [{iss['severity']}] {iss['source']}: {iss['message']}"
        )
    if len(issues) > 10:
        lines.append(f"... +{len(issues) - 10} more")
    return "\n".join(lines)


# -----------------------------

# Prompt construction

# -----------------------------


def build_prompt(file_path: str, content: str) -> str:
    metrics = basic_metrics(content)
    metrics_summary = format_metrics_for_prompt(metrics)
    issues_summary = format_issues_for_prompt([])  # no real issues yet

    code_head = "\n".join(content.splitlines()[:MAX_CODE_LINES])

    user_prompt = f"""
You are analyzing a Python file in a large codebase.

File path: {file_path}

Approximate metrics:
{metrics_summary}

Known issues:
{issues_summary}

Code (first {MAX_CODE_LINES} lines):
```python
{code_head}
```

Task:

- Propose a few SMALL, CONCRETE improvements that can be done in ~5 minutes each.
- Focus on refactors, naming, docstrings, splitting responsibilities, and removing obvious debt.
- Avoid huge rewrites or large architectural changes.

OUTPUT FORMAT (VERY IMPORTANT):

- Do NOT output JSON.
- Do NOT output XML or any machine-readable format.
- Just write normal text like you're talking to the developer.
- Use short headings:

    Suggestion 1:
    - ...
    - ...

    Suggestion 2:
    - ...

- If you show code, use fenced code blocks:

    Before:
    ```python
    # old code
    ```

    After:
    ```python
    # improved code
    ```

- Keep everything focused and short so it's obvious what to change.
"""
    return SYSTEM_PROMPT + "\n\n" + textwrap.dedent(user_prompt).strip()


# -----------------------------
# Model loading and inference
# -----------------------------

def run_on_file(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    file_path: str,
) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError as e:
        return f"[error] Could not read {file_path}: {e}"

    user_prompt = build_prompt(file_path, content)

    messages = [
        {"role": "user", "content": user_prompt}
    ]

    chat_text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Respect model context: input + output must fit
    ctx = getattr(model.config, "max_position_embeddings", 8192)
    max_input_tokens = max(512, ctx - MAX_NEW_TOKENS - 128)

    model_inputs = tok(
        [chat_text],
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)

    gen_cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.6,  # from HF guidance
        top_p=0.95,
        top_k=None,
    )

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            generation_config=gen_cfg,
        )

    # strip prompt tokens like in their example
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
    ]

    text = tok.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return text


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Console harness for WeiboAI/VibeThinker-1.5B over a codebase."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="stephanie",
        help="Root directory of the repo to scan (default: ./stephanie)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of Python files to analyze (default: 3, newest first).",
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Analyze exactly this file path instead of scanning (optional).",
    )
    parser.add_argument(
    "--out-dir",
        type=str,
        default="e:\\Users\\ernan\\Downloads",
        help="Directory to write suggestions files into (default: vibe_out).",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="List of Python files to analyze",
    )
    parser.add_argument(
        "--report-path",
        default=REPORT_PATH,
        help="Output path for combined VibeThinker report",
    )
    args = parser.parse_args()

    if not args.files:
        print("[error] No files provided. Use --files file1.py file2.py ...")
    else:
        run_vibe_batch(args.files, report_path=args.report_path)

    tok, model, device = load_model()

    if args.single:
        files = [args.single]
    else:
        files = find_python_files(args.root, args.limit)

    if not files:
        print(f"[warn] No Python files found under {args.root}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[info] Analyzing {len(files)} file(s):")
    for idx, path in enumerate(files, start=1):
        rel = os.path.relpath(path)
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(files)}] File: {rel}")
        print("=" * 80)

        suggestions = run_on_file(tok, model, device, path)

        print("\n--- VibeThinker Suggestions ---\n")
        print(suggestions or "(no output)")
        print("\n" + "-" * 80)

        # ----- Write to file as well -----
        safe_name = rel.replace(os.sep, "__")
        out_path = os.path.join(args.out_dir, safe_name + ".txt")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"# VibeThinker suggestions for {rel}\n\n")
                f.write(suggestions or "(no output)")
                f.write("\n")
            print(f"[info] Suggestions written to {out_path}")
        except OSError as e:
            print(f"[error] Could not write suggestions to {out_path}: {e}")


if __name__ == "__main__":
    main()

