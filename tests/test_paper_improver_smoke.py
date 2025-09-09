# tests/test_paper_improver_smoke.py
import tempfile
import json
from pathlib import Path
from stephanie.agents.paper_improver.orchestrator import run_paper_section

def test_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write sample spec
        spec = {
            "function_name": "l2_normalize",
            "description": "Normalize vector to unit L2 norm",
            "equations": ["x / ||x||_2"],
            "input_shape": [5],
            "output_shape": [5],
            "golden_input": [3.0, 4.0],
            "golden_output": [0.6, 0.8]
        }
        spec_path = Path(tmpdir) / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=2))

        # Write sample plan
        plan = {
            "section_title": "L2 Normalization",
            "units": [
                {"claim_id": "C1", "claim": "L2 normalization prevents gradient explosion", "evidence": "Fig 3"},
                {"claim_id": "C2", "claim": "Preserves vector direction", "evidence": "Eq 5"}
            ],
            "entities": {"ABBR": {"L2 normalization": "L2N"}}
        }
        plan_path = Path(tmpdir) / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2))

        # Run
        report = run_paper_section(
            spec_path=str(spec_path),
            plan_path=str(plan_path),
            workdir=f"{tmpdir}/runs",
            backend="torch",
            create_pr=False
        )

        # Assert outputs
        assert report["code"]["passed"] is not None
        assert report["text"]["passed"] is not None
        assert "coverage" in report["text"]["vpm_row"]
        assert "tests_pass_rate" in report["code"]["vpm_row"]
        assert Path(report["code"]["artifacts"]).exists()
        assert Path(report["text"]["artifacts"]).exists()