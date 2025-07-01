import arxiv
import json
from knowledge_cartridge import KnowledgeCartridge
from openai import OpenAI

class ResearchSlave:
    def __init__(self, api_key, instruction, generation, parent_signature=None):
        self.client = OpenAI(api_key=api_key)
        self.instruction = instruction
        self.generation = generation
        self.cartridge = KnowledgeCartridge(
            goal=instruction,
            generation=generation,
            parent_hash=parent_signature
        )
        self.excluded_sources = ["wikipedia.org", "content-farm.com"]

    def execute_research(self):
        """Perform full research workflow"""
        try:
            # Step 1: Conduct literature review
            papers = self.search_arxiv(max_results=10)
            
            # Step 2: Extract knowledge
            for paper in papers:
                findings = self.extract_knowledge(paper)
                for category, content in findings.items():
                    self.cartridge.add_finding(
                        category=category,
                        content=content,
                        source=paper.entry_id,
                        confidence=0.85  # Placeholder
                    )
            
            # Step 3: Formulate hypotheses
            self.generate_hypotheses()
            
            return self.cartridge
        except Exception as e:
            print(f"Research failed: {str(e)}")
            return None

    def search_arxiv(self, max_results=10):
        """Search arXiv for relevant papers"""
        search = arxiv.Search(
            query=self.instruction,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        return list(search.results())

    def extract_knowledge(self, paper):
        """Extract structured knowledge from paper"""
        prompt = f"""
        Research Instruction: {self.instruction}
        Paper Title: {paper.title}
        Abstract: {paper.summary}
        
        Extract knowledge as JSON with these keys:
        - supporting_evidence: Array of key findings
        - contradictions: Array of conflicting evidence
        - hypotheses: Array of testable hypotheses
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    
    # Additional methods...