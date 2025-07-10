import random


class InstructionEngine:
    def __init__(self, api_key, base_goal):
        self.base_goal = base_goal
        self.variation_strategies = [
            "Change methodology to experimental approach",
            "Shift focus to practical applications",
            "Add time constraint: research from last 2 years",
            "Emphasize theoretical foundations",
            "Include cross-disciplinary perspective",
            "Prioritize quantitative over qualitative analysis",
            "Focus on controversial aspects"
        ]

    def generate_variants(self, num_variants=20, temperature=0.7):
        """Generate diverse research instructions"""
        variants = []
        for i in range(num_variants):
            strategy = random.choice(self.variation_strategies)
            prompt = f"""
            Base Goal: {self.base_goal}
            Variation Strategy: {strategy}
            
            Create a specific research instruction that:
            1. Maintains the core intent of the base goal
            2. Implements the variation strategy
            3. Is clear and actionable for research AI
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=150
            )
            
            variant = response.choices[0].message.content.strip()
            variants.append({
                "id": f"variant_{i+1}",
                "instruction": variant,
                "strategy": strategy
            })
        return variants