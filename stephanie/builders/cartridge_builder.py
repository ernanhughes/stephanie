# stephanie/builders/cartridge_builder.py

from datetime import datetime

from stephanie.models.theorem import CartridgeORM


class CartridgeBuilder:
    def __init__(self, cfg, memory, prompt_loader, logger=None, call_llm=None):
        self.cfg = cfg    
        self.memory = memory
        self.prompt_loader = prompt_loader
        self.logger = logger
        self.call_llm = call_llm

    def build(self, doc: dict, goal: dict = None, context: dict = None) -> CartridgeORM:
        """
        Constructs a CartridgeORM object from a raw document dict.
        """
        context = context or {}
        doc_id = doc["id"]

        existing = self.memory.cartridges.get_by_source_uri(str(doc_id), "document")
        if existing:    
            if self.logger:
                self.logger.log("CartridgeAlreadyExists", {"source_uri": str(doc_id), "type": "document"})
            return existing

        title = doc.get("title", f"Document {doc_id}")
        summary = doc.get("summary", "")
        text = doc.get("content", doc.get("text", ""))
        goal_id = goal.get("id") if goal else None

        self.memory.embedding.get_or_create(text)
        embedding_vector_id = self.memory.embedding.get_id_for_text(text)
        if not embedding_vector_id:
            if self.logger:
                self.logger.log("EmbeddingNotFound", {"text": text[:100]})
            return None

        # Extract sections from the content
        sections = self._split_into_sections(text, goal, context)
        # Generate unified markdown content
        markdown_content = self.format_markdown(title, summary, sections)
        cartridge = self.memory.cartridges.add_cartridge({"goal_id":goal_id,
            "source_type":"document",
            "source_uri":str(doc_id),
            "title":title,
            "summary":summary,
            "sections":sections,
            "triples":[],
            "domain_tags":[],
            "embedding_id":embedding_vector_id,
            "markdown_content":markdown_content,
            "created_at":datetime.utcnow(),})
        return cartridge
                

    def _split_into_sections(self, text: str, goal: dict, context: dict) -> dict:
        """
        Extracts section points from the content using LLM.
        """
        merged_context = {"text": text, "goal": goal, **context}
        prompt = self.prompt_loader.load_prompt(self.cfg, merged_context)
        response = self.call_llm(prompt, context=merged_context)
        return self._parse_response(response)

    def _parse_response(self, markdown_text: str) -> list:
        """
        Extracts bullet points from markdown-formatted LLM output.
        """
        import re
        bullet_pattern = re.compile(r"^\s*([#*-]{1,2})\s+(.*)", re.MULTILINE)
        raw_lines = bullet_pattern.findall(markdown_text)
        bullet_points = [re.sub(r"\*\*(.*?)\*\*", r"\1", content).strip() for _, content in raw_lines]
        return bullet_points

    def format_markdown(self, title: str, summary: str, sections: list) -> str:
        """
        Constructs a clean markdown content string for scoring or embedding.
        """
        lines = []
        if title:
            lines.append(f"Title: {title}")
        if summary:
            lines.append(f"Summary: {summary}")
        for section in sections:
            lines.append(section)
        return "\n\n".join(lines).strip()
