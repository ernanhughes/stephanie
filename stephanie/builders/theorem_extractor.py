# stephanie/builders/theorem_extractor.py
from stephanie.models.theorem import TheoremORM


class TheoremExtractor:
    def __init__(self, cfg, memory, prompt_loader, logger, call_llm):
        self.cfg = cfg
        self.memory = memory
        self.prompt_loader = prompt_loader
        self.logger = logger
        self.call_llm = call_llm
        self.prompt_template = cfg.get(
            "theorem_extraction_prompt", "theorem_extraction_prompt.txt"
        )

    def extract(self, sections, context):
        extracted_theorems = []
        for count, section in enumerate(sections):
            if count >= self.cfg.get("max_sections", 2):
                self.logger.log(
                    "MaxSectionsReached", {"max_sections": self.cfg.get("max_sections")}
                )
                break
            merged_context = {"section_text": section, **context}
            prompt = self.prompt_loader.from_file(
                self.prompt_template, self.cfg, merged_context
            )

            response = self.call_llm(prompt, context=context)
            theorem_statements = self.parse_llm_response(response)

            for statement in theorem_statements:
                theorem = TheoremORM(statement=statement)
                extracted_theorems.append(theorem)
                self.logger.log("TheoremExtracted", {"statement": statement[:100]})

        return extracted_theorems

    def parse_llm_response(self, response):
        # Simple parsing logic, customize as necessary
        statements = [line.strip() for line in response.split("\n") if line.strip()]
        return statements
