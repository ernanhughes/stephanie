# stephanie/agents/evolution_engine/evolution_selector.py
import random


class EvolutionarySelector:
    def __init__(self, elite_size=0.3, mutation_rate=0.2):
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_strategies = [
            "add_random_constraint",
            "change_research_method",
            "shift_perspective",
            "increase_specificity",
            "add_cross_domain",
        ]

    def next_generation(self, cartridges, num_outputs):
        """Select and evolve cartridges for next generation"""
        # Rank cartridges by quality
        ranked = sorted(
            cartridges, key=lambda c: c.quality_metrics["overall_score"], reverse=True
        )

        # Select elite cartridges
        num_elites = max(1, int(self.elite_size * len(ranked)))
        elites = ranked[:num_elites]

        # Create next generation
        next_gen = []
        for i in range(num_outputs):
            if i < len(elites):
                # Preserve elite unchanged
                next_gen.append(elites[i])
            else:
                # Create mutated variant
                base = random.choice(elites)
                variant = self.create_variant(base)
                next_gen.append(variant)

        return next_gen

    def create_variant(self, base_cartridge):
        """Create mutated variant of a cartridge"""
        import copy

        variant = copy.deepcopy(base_cartridge)
        variant.generation += 1
        variant.parent_hash = base_cartridge.signature

        # Apply mutations to instruction
        if random.random() < self.mutation_rate:
            mutation = random.choice(self.mutation_strategies)
            variant.goal = self.mutate_instruction(variant.goal, mutation)

        return variant

    def mutate_instruction(self, instruction, strategy):
        """Apply specific mutation strategy to instruction"""
        prompts = {
            "add_random_constraint": (
                "Add a random constraint to this research instruction: {instruction}"
            ),
            "change_research_method": (
                "Change the research methodology in this instruction: {instruction}"
            ),
            # Other strategies...
        }

        prompt = prompts.get(strategy, "").format(instruction=instruction)
        if not prompt:
            return instruction

        # Use LLM to implement mutation (implementation similar to InstructionEngine)
        return instruction + " [Mutated]"
