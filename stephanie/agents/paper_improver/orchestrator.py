# stephanie/agents/paper_improver/orchestrator.py

# orchestrator.py — End-to-end spec + plan → code + text → PR (optional)
import json
import argparse
from pathlib import Path
from typing import Optional

from .code_improver import CodeImprover
from .text_improver import TextImprover
from .repo_link import RepoLink
from .vpm_controller import VPMController

def run_paper_section(
    spec_path: str,
    plan_path: str,
    workdir: str = "./runs",
    backend: str = "torch",
    create_pr: bool = False,
    repo_root: str = "../.."
) -> dict:
    """
    Run code + text improvers. Optionally create PR.
    Returns combined report.
    """
    # Load inputs
    spec = json.loads(Path(spec_path).read_text())
    plan = json.loads(Path(plan_path).read_text())

    # Init improvers
    ci = CodeImprover(backend=backend, workdir=f"{workdir}/code")
    ti = TextImprover(workdir=f"{workdir}/text")
    vc = VPMController()

    # Improve code
    print("🔧 Improving code...")
    code_result = ci.improve(spec)
    code_action = vc.add_vpm_row(code_result["vpm_row"], f"code:{spec['function_name']}")
    print(f"→ Code VPM: {code_result['vpm_row']}")
    print(f"→ Controller: {code_action}")

    # Improve text
    print("📝 Improving text...")
    text_result = ti.improve(plan)
    text_action = vc.add_vpm_row(text_result["vpm_row"], f"text:{plan['section_title']}")
    print(f"→ Text VPM: {text_result['vpm_row']}")
    print(f"→ Controller: {text_action}")

    # Build report
    report = {
        "spec": spec_path,
        "plan": plan_path,
        "code": {
            "vpm_row": code_result["vpm_row"],
            "action": code_action,
            "artifacts": code_result["run_dir"],
            "passed": code_result["passed"]
        },
        "text": {
            "vpm_row": text_result["vpm_row"],
            "action": text_action,
            "artifacts": text_result["run_dir"],
            "passed": text_result["passed"]
        }
    }

    # Create PR if requested
    if create_pr:
        print("🚀 Creating PRs...")
        rl = RepoLink(repo_root=repo_root)
        if code_result["passed"]:
            rl.create_pr(code_result["run_dir"], code_result["vpm_row"], "code")
        if text_result["passed"]:
            rl.create_pr(text_result["run_dir"], text_result["vpm_row"], "text")

    # Save report
    report_path = Path(workdir) / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"✅ Report saved: {report_path}")

    return report

def main():
    parser = argparse.ArgumentParser(description="Run paper section improver")
    parser.add_argument("--spec", required=True, help="Path to spec.json")
    parser.add_argument("--plan", required=True, help="Path to plan.json")
    parser.add_argument("--workdir", default="./runs", help="Working directory")
    parser.add_argument("--backend", default="torch", choices=["torch", "numpy"], help="Code backend")
    parser.add_argument("--pr", action="store_true", help="Create PRs if passed")
    parser.add_argument("--repo-root", default="../..", help="Root of git repo")

    args = parser.parse_args()
    run_paper_section(
        spec_path=args.spec,
        plan_path=args.plan,
        workdir=args.workdir,
        backend=args.backend,
        create_pr=args.pr,
        repo_root=args.repo_root
    )

if __name__ == "__main__":
    main()