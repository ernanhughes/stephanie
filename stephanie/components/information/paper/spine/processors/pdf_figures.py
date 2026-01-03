from typing import Dict, List

from .base import BaseSpineProcessor, ProcessorResult
from stephanie.components.information.data import DocumentElement


class PdfFiguresProcessor(BaseSpineProcessor):
    name = "pdf_figures"

    async def run(self, *, arxiv_id, pdf_path, elements, context):
        # Placeholder: actual extraction handled by PdfElementsTool
        return elements, ProcessorResult(name=self.name, ran=False)
