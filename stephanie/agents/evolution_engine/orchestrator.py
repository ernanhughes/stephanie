# stephanie/agents/evolution_engine/orchestrator.py
import os
from datetime import datetime

from evolution_selector import EvolutionarySelector
from slave_system import ResearchSlave

from stephanie.agents.evolution.instruction_engine import InstructionEngine
from stephanie.reports.report_generator import ReportGenerator


class ResearchOrchestrator:
    def __init__(self, base_goal, api_key, num_slaves=20, max_generations=5):
        self.base_goal = base_goal
        self.api_key = api_key
        self.num_slaves = num_slaves
        self.max_generations = max_generations
        self.output_dir = f"research_output/{base_goal.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize components
        self.instruction_engine = InstructionEngine(api_key, base_goal)
        self.fusion_engine = CartridgeMerger()
        self.scorer = CartridgeScorer(api_key)
        self.selector = EvolutionarySelector()
        self.report_generator = ReportGenerator()

    def run_evolution(self):
        """Execute full evolutionary research process"""
        # Initial setup
        generation = 0
        cartridges = []
        master_cartridge = None

        # Evolutionary loop
        while generation < self.max_generations:
            print(f"\n=== GENERATION {generation} ===")

            # 1. Generate research instructions
            if generation == 0:
                instructions = self.instruction_engine.generate_variants(
                    self.num_slaves
                )
            else:
                instructions = self._cartridges_to_instructions(cartridges)

            # 2. Distribute to slave systems
            slave_results = []
            for i, instruction in enumerate(instructions):
                print(f"  Slave {i + 1}: {instruction['instruction']}")
                slave = ResearchSlave(
                    api_key=self.api_key,
                    instruction=instruction["instruction"],
                    generation=generation,
                    parent_signature=master_cartridge.signature
                    if master_cartridge
                    else None,
                )
                result = slave.execute_research()
                if result:
                    slave_results.append(result)
                    self._save_cartridge(result, f"gen_{generation}_slave_{i}")

            # 3. Merge and evaluate
            master_cartridge = self.fusion_engine.merge(slave_results)
            master_cartridge = self.scorer.evaluate(master_cartridge)
            self._save_cartridge(master_cartridge, f"gen_{generation}_master")

            # 4. Check convergence
            if self._check_convergence(cartridges, master_cartridge):
                print("Convergence achieved. Stopping early.")
                break

            # 5. Prepare next generation
            cartridges = slave_results
            generation += 1

        # Final output
        final_report = self.report_generator.generate(master_cartridge)
        with open(f"{self.output_dir}/final_report.md", "w") as f:
            f.write(final_report)

        print(f"\nResearch complete. Final report saved to {self.output_dir}")
        return final_report

    def _save_cartridge(self, cartridge, filename):
        """Save cartridge to file"""
        # Save JSON
        with open(f"{self.output_dir}/{filename}.json", "w") as f:
            f.write(cartridge.to_json())

        # Save Markdown
        with open(f"{self.output_dir}/{filename}.md", "w") as f:
            f.write(cartridge.to_markdown())

    def _check_convergence(self, previous_cartridges, master_cartridge):
        """Determine if research has converged"""
        if not previous_cartridges:
            return False

        # Calculate quality improvement
        prev_scores = [c.quality_metrics["overall_score"] for c in previous_cartridges]
        prev_avg = sum(prev_scores) / len(prev_scores)
        current_score = master_cartridge.quality_metrics["overall_score"]

        return (current_score - prev_avg) < 0.01  # 1% improvement threshold

    # Additional helper methods...
