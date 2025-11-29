# stephanie/components/critic/agents/teachpack_consumer.py
import json
import logging
from pathlib import Path
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.critic.utils.teachpack import train_from_teachpack
from stephanie.utils.file_utils import save_json

log = logging.getLogger(__name__)

class CriticConsumerAgent(BaseAgent):
    """
    Consumes a teachpack to train a new critic model.
    Demonstrates portable learning capability.
    """
    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.cfg = cfg
        self.memory = memory
        self.container = container
        log = logger
        
        # Configuration
        self.teachpack_path = Path(cfg.get("teachpack_path", "/models/teachpacks/default_teachpack.npz"))
        self.output_model_path = Path(cfg.get("output_model_path", "/models/teachpack_critic.joblib"))
        self.output_meta_path = Path(cfg.get("output_meta_path", "/models/teachpack_critic.meta.json"))
        self.report_dir = Path(cfg.get("report_dir", f"/runs/critic/{self.run_id}"))
        
        # Create directories
        self.output_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Initialized TeachpackConsumerAgent with teachpack={self.teachpack_path}")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the teachpack consumption"""
        log.info(f"Training new critic from teachpack: {self.teachpack_path}")
        
        try:
            # Train from teachpack
            results = train_from_teachpack(
                str(self.teachpack_path),
                str(self.output_model_path),
                str(self.output_meta_path)
            )
            
            # Generate report
            report = {
                "teachpack_path": str(self.teachpack_path),
                "output_model_path": str(self.output_model_path),
                "output_meta_path": str(self.output_meta_path),
                "auroc": results["auroc"],
                "feature_count": results["feature_count"],
                "sample_count": results["sample_count"]
            }
            
            # Save report
            report_path = self.report_dir / "teachpack_results.json"
            save_json(report, str(report_path))
            
            # Generate markdown report
            md_content = f"""# Teachpack Consumption Report

## Teachpack Information
- **Path**: {self.teachpack_path}
- **Feature Count**: {results['feature_count']}
- **Sample Count**: {results['sample_count']}

## Training Results
- **AUROC**: {results['auroc']:.3f}
- **Model Path**: {self.output_model_path}
- **Meta Path**: {self.output_meta_path}

## Interpretation
"""
            
            if results['auroc'] > 0.75:
                md_content += "✅ The teachpack successfully transferred knowledge to a new critic model with high performance.\n"
            elif results['auroc'] > 0.65:
                md_content += "⚠️ The teachpack transferred moderate knowledge to a new critic model.\n"
            else:
                md_content += "❌ The teachpack did not effectively transfer knowledge to a new critic model.\n"
            
            md_path = self.report_dir / "teachpack_report.md"
            with open(md_path, "w") as f:
                f.write(md_content)
            
            # Save to context
            context["teachpack"] = {
                "results": results,
                "report_path": str(report_path),
                "md_path": str(md_path)
            }
            
            log.info(f"Teachpack consumption complete. Report saved to {md_path}")
            return context
            
        except Exception as e:
            log.exception("Teachpack consumption failed")
            context["teachpack_error"] = str(e)
            return context